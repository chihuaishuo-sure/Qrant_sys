"""数据层：AKShare（东财 → 新浪 多源回退）+ 本地缓存 + CSV 导入 + 离线演示数据。

统一输出：以日期为索引、列为 open/high/low/close/volume 的 DataFrame（升序）。
"""
from __future__ import annotations

import io
import time
import zlib
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import is_etf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_TTL_HOURS = 8

_COLMAP = {
    "日期": "date", "date": "date", "Date": "date", "trade_date": "date", "时间": "date",
    "开盘": "open", "open": "open", "Open": "open", "开盘价": "open",
    "最高": "high", "high": "high", "High": "high", "最高价": "high",
    "最低": "low", "low": "low", "Low": "low", "最低价": "low",
    "收盘": "close", "close": "close", "Close": "close", "收盘价": "close",
    "成交量": "volume", "volume": "volume", "Volume": "volume", "vol": "volume",
}


class DataError(RuntimeError):
    pass


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """把各种来源的列名统一为 open/high/low/close/volume，并清洗。"""
    if df is None or len(df) == 0:
        raise DataError("数据为空")
    df = df.rename(columns={c: _COLMAP.get(str(c).strip(), str(c).strip()) for c in df.columns})
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        else:
            raise DataError("缺少日期列（需要 date 或 日期）")
    if "close" not in df.columns:
        raise DataError("缺少收盘价列（需要 close 或 收盘）")
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df[c] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    out = out.set_index("date")
    out.index.name = "date"
    # 开盘价缺失/为 0 时用收盘价代替；保证 high/low 包住 open/close
    out["open"] = out["open"].where(out["open"] > 0, out["close"])
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    if len(out) < 30:
        raise DataError(f"数据太少（{len(out)} 行），至少需要 30 个交易日")
    return out


def guess_kind(symbol: str) -> str:
    s = symbol.strip().lower()
    if s.startswith(("sh", "sz", "csi")) and len(s) > 6 and s[2:5] in {"000", "399", "930", "931"}:
        return "index"
    if s.startswith("sz399") or s.startswith("sh000"):
        return "index"
    return "etf" if is_etf(s) else "stock"


def _exchange_prefix(code: str) -> str:
    c = code[-6:]
    if c.startswith(("6", "5", "9")):
        return "sh" + c
    if c.startswith(("8", "4")):
        return "bj" + c
    return "sz" + c


def _fetch_akshare(symbol: str, kind: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    try:
        import akshare as ak  # 延迟导入：没有 akshare 也能用演示数据
    except ImportError as e:  # pragma: no cover
        raise DataError("未安装 akshare，请运行 启动.bat 自动安装，或 pip install akshare") from e

    code = symbol.strip().lower()
    errors: list[str] = []

    def attempt(fn, *a, **kw):
        try:
            d = fn(*a, **kw)
            if d is not None and len(d) > 0:
                return normalize(d)
            errors.append(f"{fn.__name__}: 返回空")
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{getattr(fn, '__name__', fn)}: {ex}")
        return None

    if kind == "index":
        sym = code if code[:2] in ("sh", "sz") else _exchange_prefix(code)
        for fn, kw in (
            (getattr(ak, "stock_zh_index_daily_em", None), {"symbol": sym}),
            (getattr(ak, "stock_zh_index_daily", None), {"symbol": sym}),
        ):
            if fn is not None:
                d = attempt(fn, **kw)
                if d is not None:
                    return d.loc[pd.Timestamp(start): pd.Timestamp(end)]
    elif kind == "etf":
        c6 = code[-6:]
        d = attempt(ak.fund_etf_hist_em, symbol=c6, period="daily",
                    start_date=start, end_date=end, adjust=adjust)
        if d is not None:
            return d
        fn = getattr(ak, "fund_etf_hist_sina", None)
        if fn is not None:
            d = attempt(fn, symbol=_exchange_prefix(c6))
            if d is not None:
                return d.loc[pd.Timestamp(start): pd.Timestamp(end)]
    else:
        c6 = code[-6:]
        d = attempt(ak.stock_zh_a_hist, symbol=c6, period="daily",
                    start_date=start, end_date=end, adjust=adjust)
        if d is not None:
            return d
        fn = getattr(ak, "stock_zh_a_daily", None)
        if fn is not None:
            d = attempt(fn, symbol=_exchange_prefix(c6), start_date=start,
                        end_date=end, adjust=adjust)
            if d is not None:
                return d
    raise DataError("所有数据接口均失败：\n" + "\n".join(errors[-4:]))


def _cache_path(symbol: str, kind: str, adjust: str) -> Path:
    return CACHE_DIR / f"{kind}_{symbol.strip().lower()}_{adjust or 'none'}.csv"


SPLIT_THRESHOLD = 0.35  # 单日涨跌超过 ±35%：超出任何 A 股/ETF 涨跌停，只可能是份额拆分/合并未复权


def repair_splits(df: pd.DataFrame, threshold: float = SPLIT_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    """修正 ETF 份额拆分/合并造成的价格断层（东财 ETF 前复权不处理份额折算）。

    做法：在断层日之前的所有价格乘以 (断层日开盘 / 前一日收盘)，使开盘无缺口；
    断层日当天“开盘→收盘”的真实涨跌保留。返回 (修正后数据, 修正记录)。
    """
    df = df.copy()
    notes: list[str] = []
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    for i in range(5, len(df)):  # 跳过上市头几天
        r = c[i] / c[i - 1] - 1
        if abs(r) > threshold:
            ratio = o[i] / c[i - 1] if o[i] > 0 else c[i] / c[i - 1]
            if abs(ratio - 1) < threshold:  # 开盘没有断层，说明不是折算
                continue
            df.iloc[:i, [df.columns.get_loc(k) for k in ("open", "high", "low", "close")]] *= ratio
            c = df["close"].to_numpy()
            o = df["open"].to_numpy()
            notes.append(f"{df.index[i]:%Y-%m-%d} 份额折算 ×{ratio:.4f}")
    return df, notes


def _finish(df: pd.DataFrame, start_s: str, end_s: str, src: str) -> tuple[pd.DataFrame, str]:
    df, notes = repair_splits(df)
    if notes:
        src += "；已修正：" + "，".join(notes)
    return df.loc[pd.Timestamp(start_s): pd.Timestamp(end_s)], src


def fetch(symbol: str, start: str | date = "20100101", end: str | date | None = None,
          kind: str = "auto", adjust: str = "qfq", force: bool = False) -> tuple[pd.DataFrame, str]:
    """获取日线。返回 (数据, 来源说明)。网络失败时自动回退到本地缓存。"""
    if kind == "auto":
        kind = guess_kind(symbol)
    start_s = pd.Timestamp(start).strftime("%Y%m%d")
    end_s = pd.Timestamp(end or datetime.now()).strftime("%Y%m%d")
    path = _cache_path(symbol, kind, adjust)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_HOURS * 3600
    if fresh and not force:
        cached = normalize(pd.read_csv(path))  # 缓存里保存的是全部历史（原始数据）
        return _finish(cached, start_s, end_s, "本地缓存")

    try:
        df = _fetch_akshare(symbol, kind, "19900101", end_s, adjust)
        df.reset_index().to_csv(path, index=False)
        return _finish(df, start_s, end_s, "AKShare 在线")
    except DataError as e:
        if path.exists():
            cached = normalize(pd.read_csv(path))
            return _finish(cached, start_s, end_s, f"本地缓存（在线失败：{str(e)[:60]}）")
        raise


def load_csv(file) -> pd.DataFrame:
    """读取用户上传的 CSV（支持中文/英文表头，UTF-8 或 GBK）。"""
    raw = file.read() if hasattr(file, "read") else Path(file).read_bytes()
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return normalize(pd.read_csv(io.BytesIO(raw), encoding=enc))
        except UnicodeDecodeError:
            continue
    raise DataError("无法识别 CSV 编码，请另存为 UTF-8")


# ---------------------------------------------------------------- 演示数据
_DEMO_PROFILES = {
    # 代码: (名称, 股票因子暴露, 自身漂移, 自身波动, 价格起点)
    "510300": ("沪深300ETF(模拟)", 1.0, 0.00, 0.06, 3.0),
    "510500": ("中证500ETF(模拟)", 1.15, 0.01, 0.10, 5.0),
    "159915": ("创业板ETF(模拟)", 1.35, 0.02, 0.14, 1.5),
    "518880": ("黄金ETF(模拟)", -0.05, 0.06, 0.14, 2.5),
    "511010": ("国债ETF(模拟)", -0.08, 0.03, 0.025, 100.0),
    "513100": ("纳指ETF(模拟)", 0.45, 0.08, 0.16, 1.0),
    "sh000300": ("沪深300指数(模拟)", 1.0, 0.00, 0.05, 3000.0),
}


def _market_factor(n: int, seed: int = 7) -> np.ndarray:
    """带牛/熊/震荡状态切换的市场因子日收益（波动聚集、熊市持续）。"""
    rng = np.random.default_rng(seed)
    # 状态：0 牛 1 震荡 2 熊
    mu = np.array([0.32, 0.04, -0.26]) / 252
    sd = np.array([0.17, 0.18, 0.30]) / np.sqrt(252)
    P = np.array([[0.992, 0.006, 0.002],
                  [0.004, 0.992, 0.004],
                  [0.004, 0.008, 0.988]])
    s = 1
    out = np.empty(n)
    for i in range(n):
        s = rng.choice(3, p=P[s])
        out[i] = rng.normal(mu[s], sd[s])
        if rng.random() < 0.002:  # 偶发跳空
            out[i] += rng.normal(-0.03, 0.02)
    return out


def demo_data(symbol: str = "510300", start: str = "20120101", end: str | None = None) -> pd.DataFrame:
    """生成可复现的模拟行情（仅用于离线体验功能，不代表真实市场）。"""
    sym = symbol.strip().lower()
    name, beta, drift, idio, p0 = _DEMO_PROFILES.get(sym, (sym + "(模拟)", 1.0, 0.0, 0.10, 10.0))
    idx = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end or "20261016"))
    n = len(idx)
    mkt = _market_factor(n)
    rng = np.random.default_rng(zlib.crc32(sym.encode()) if sym not in _DEMO_PROFILES
                                else list(_DEMO_PROFILES).index(sym) + 101)
    r = beta * mkt + drift / 252 + rng.normal(0, idio / np.sqrt(252), n)
    close = p0 * np.exp(np.cumsum(np.log1p(np.clip(r, -0.095, 0.095))))
    gap = rng.normal(0, 0.003, n)
    open_ = np.r_[p0, close[:-1]] * (1 + gap)
    rng_hl = np.abs(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * (1 + rng_hl)
    low = np.minimum(open_, close) * (1 - rng_hl)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    df.index.name = "date"
    return df


def demo_name(symbol: str) -> str:
    return _DEMO_PROFILES.get(symbol.strip().lower(), (symbol + "(模拟)",))[0]
