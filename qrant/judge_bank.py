"""裁判 A 的评测题库：干净的策略代码 + 植入前视 bug 的变体。

标准答案不靠人工标注，而是靠执行：把 t 日之后的行情随机扰动，t 日及以前的信号只要变了，
就说明代码用到了未来数据（与 tests/test_core.py 的无前视测试同一原理）。
设计意图（intent）与执行结果（oracle）不一致的题会被单元测试拦下，保证题库本身没出错。

每道题是一段自包含的代码，定义 signal(df) -> pd.Series（目标仓位），只依赖 pandas/numpy。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASES: dict[str, str] = {
    "均线趋势": '''
def signal(df):
    c = df["close"]
    fast = c.rolling(20, min_periods=20).mean()
    slow = c.rolling(120, min_periods=120).mean()
    return (fast > slow).astype(float).where(slow.notna(), 0.0)
''',
    "多周期动量": '''
def signal(df):
    c = df["close"]
    votes = []
    for n in (20, 60, 120, 250):
        votes.append((c / c.shift(n) - 1 > 0).astype(float).where(c.shift(n).notna()))
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0)
''',
    "唐奇安突破": '''
def signal(df):
    hi = df["high"].rolling(55, min_periods=55).max().shift(1)
    lo = df["low"].rolling(20, min_periods=20).min().shift(1)
    c, h, l = df["close"].to_numpy(), hi.to_numpy(), lo.to_numpy()
    pos = np.zeros(len(c))
    state = 0.0
    for i in range(len(c)):
        if state == 0 and not np.isnan(h[i]) and c[i] > h[i]:
            state = 1.0
        elif state == 1 and not np.isnan(l[i]) and c[i] < l[i]:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)
''',
    "Z分数回归": '''
def signal(df):
    c = df["close"]
    m = c.rolling(60, min_periods=60).mean()
    s = c.rolling(60, min_periods=60).std()
    z = (c - m) / s
    trend = c > c.rolling(200, min_periods=200).mean()
    return ((z < -1.0) & trend).astype(float)
''',
    "波动率缩放动量": '''
def signal(df):
    c = df["close"]
    r = c.pct_change()
    vol = r.rolling(20, min_periods=20).std() * np.sqrt(252)
    mom = c / c.shift(120) - 1
    w = (0.15 / vol).clip(upper=1.0)
    return w.where(mom > 0, 0.0).fillna(0.0)
''',
    "RSI回调": '''
def signal(df):
    c = df["close"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 6, adjust=False, min_periods=6).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 6, adjust=False, min_periods=6).mean()
    rsi = 100 - 100 / (1 + up / dn)
    trend = c > c.rolling(200, min_periods=200).mean()
    return ((rsi < 25) & trend).astype(float)
''',
    "通道位置": '''
def signal(df):
    c = df["close"]
    hi = df["high"].rolling(100, min_periods=100).max()
    lo = df["low"].rolling(100, min_periods=100).min()
    pos = (c - lo) / (hi - lo)
    return (pos > 0.8).astype(float)
''',
    "回撤过滤": '''
def signal(df):
    c = df["close"]
    peak = c.cummax()
    dd = c / peak - 1
    ema = c.ewm(span=50, adjust=False).mean()
    rising = ema.diff() > 0
    return ((c > ema) & rising & (dd > -0.2)).astype(float)
''',
    "周线趋势": '''
def signal(df):
    c = df["close"]
    wk = c.resample("W-FRI").last()
    wk_ma = wk.rolling(10, min_periods=10).mean()
    up = (wk > wk_ma).astype(float)
    return up.reindex(c.index, method="ffill").fillna(0.0)
''',
    "历史分位": '''
def signal(df):
    c = df["close"]
    r60 = c.pct_change(60)
    rank = r60.expanding(min_periods=250).rank(pct=True)
    return (rank > 0.5).astype(float)
''',
}

# (题目, 类型, 设计意图是否有前视, 原文, 替换)
VARIANTS: list[tuple[str, str, bool, str, str]] = [
    # ---- 植入的前视 bug
    ("多周期动量", "负向shift", True, "c / c.shift(n) - 1", "c / c.shift(-n) - 1"),
    ("均线趋势", "负向shift", True, "slow = c.rolling(120, min_periods=120).mean()",
     "slow = c.rolling(120, min_periods=120).mean().shift(-10)"),
    ("均线趋势", "居中滚动", True, "c.rolling(20, min_periods=20)", "c.rolling(20, min_periods=20, center=True)"),
    ("Z分数回归", "居中滚动", True, "s = c.rolling(60, min_periods=60).std()",
     "s = c.rolling(60, min_periods=60, center=True).std()"),
    ("Z分数回归", "全样本统计量", True, "z = (c - m) / s", "z = (c - c.mean()) / c.std()"),
    ("回撤过滤", "全样本统计量", True, "peak = c.cummax()", "peak = c.max()"),
    ("通道位置", "全样本统计量", True, 'hi = df["high"].rolling(100, min_periods=100).max()', 'hi = df["high"].max()'),
    ("周线趋势", "向后填充", True, 'method="ffill"', 'method="bfill"'),
    ("多周期动量", "向后填充", True, ".mean(axis=1).fillna(0.0)", ".mean(axis=1).bfill().fillna(0.0)"),
    ("波动率缩放动量", "插值", True, "mom = c / c.shift(120) - 1",
     "mom = (c / c.shift(120) - 1).where(c.index.dayofweek == 4).interpolate()"),
    ("RSI回调", "负向diff", True, "d = c.diff()", "d = -c.diff(-1)"),
    ("波动率缩放动量", "负向pct_change", True, "r = c.pct_change()", "r = c.pct_change(-1)"),
    ("唐奇安突破", "未来下标", True, "c[i] > h[i]", "i + 1 < len(c) and c[i + 1] > h[i]"),
    ("回撤过滤", "反转累计", True, "peak = c.cummax()", "peak = c[::-1].cummax()[::-1]"),
    ("回撤过滤", "两侧差分", True, "rising = ema.diff() > 0",
     "rising = pd.Series(np.gradient(ema.to_numpy()), index=c.index) > 0"),
    ("历史分位", "全样本统计量", True, "rank = r60.expanding(min_periods=250).rank(pct=True)",
     "rank = r60.rank(pct=True)"),
    ("均线趋势", "全样本统计量", True, ".where(slow.notna(), 0.0)",
     ".where(slow.notna(), 0.0) * (c > c.quantile(0.2)).astype(float)"),
    ("通道位置", "负向shift", True, "pos = (c - lo) / (hi - lo)", "pos = (c.shift(-5) - lo) / (hi - lo)"),
    ("RSI回调", "居中滚动", True, "trend = c > c.rolling(200, min_periods=200).mean()",
     "trend = c > c.rolling(200, min_periods=100, center=True).mean()"),
    ("Z分数回归", "向后填充", True, "m = c.rolling(60, min_periods=60).mean()",
     "m = c.rolling(60, min_periods=60).mean().where(c.index.dayofweek == 4).bfill()"),
    # ---- 看起来可疑、实际干净的“陷阱题”
    ("唐奇安突破", "陷阱:去掉shift", False, 'max().shift(1)', 'max()'),
    ("多周期动量", "陷阱:注释提到shift(-1)", False, "    votes = []",
     "    # 注意：这里必须用 shift(n) 而不是 shift(-n)，后者会偷看未来\n    votes = []"),
    ("均线趋势", "陷阱:expanding", False, "slow = c.rolling(120, min_periods=120).mean()",
     "slow = c.expanding(min_periods=120).mean()"),
    ("回撤过滤", "陷阱:adjust=True", False, "ema = c.ewm(span=50, adjust=False).mean()",
     "ema = c.ewm(span=50, adjust=True).mean()"),
    ("通道位置", "陷阱:rolling.apply", False, 'hi = df["high"].rolling(100, min_periods=100).max()',
     'hi = df["high"].rolling(100, min_periods=100).apply(lambda x: x[-1] * 0 + x.max(), raw=True)'),
    ("波动率缩放动量", "陷阱:前向填充", False, "w = (0.15 / vol).clip(upper=1.0)",
     "w = (0.15 / vol).clip(upper=1.0).ffill()"),
    ("唐奇安突破", "陷阱:i-1下标", False, "c[i] > h[i]", "i > 0 and c[i - 1] < c[i] and c[i] > h[i]"),
    ("历史分位", "陷阱:正向pct_change", False, "r60 = c.pct_change(60)", "r60 = c.pct_change(periods=60)"),
    ("Z分数回归", "陷阱:shift(1)防前视", False, "z = (c - m) / s", "z = (c - m.shift(1)) / s.shift(1)"),
    ("周线趋势", "陷阱:resample均值", False, 'wk = c.resample("W-FRI").last()',
     'wk = c.resample("W-FRI").last().rolling(2, min_periods=1).mean()'),
]


@dataclass
class Case:
    id: str
    base: str
    kind: str
    intent: bool  # 设计意图：是否有前视
    code: str
    planted: str = ""  # 植入（替换进去）的代码片段，用来检查裁判是否指对了位置


def build() -> list[Case]:
    cases = [Case(f"{b}/干净", b, "干净原版", False, code.strip() + "\n") for b, code in BASES.items()]
    for base, kind, intent, old, new in VARIANTS:
        src = BASES[base]
        if src.count(old) != 1:
            raise ValueError(f"{base}/{kind}：原文必须恰好出现一次：{old!r}")
        cases.append(Case(f"{base}/{kind}", base, kind, intent, src.replace(old, new).strip() + "\n", new))
    return cases


def run_signal(code: str, df: pd.DataFrame) -> pd.Series:
    ns: dict = {"pd": pd, "np": np}
    exec(code, ns)  # noqa: S102  题库代码是本模块自带的，不执行外部输入
    return pd.Series(ns["signal"](df), index=df.index, dtype=float)


def _perturbations(k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """对 cut 之后 k 天的价格乘数。要“够狠”：输出多是 0/1 信号，温和扰动很难让切点前的信号翻转，
    会漏判真实的前视。所以同时用暴涨、暴跌、逐日剧烈噪声和随机游走。"""
    return [np.full(k, 10.0), np.full(k, 0.1),
            np.exp(rng.normal(0, 0.5, k)),
            np.exp(np.cumsum(rng.normal(0, 0.1, k)))]


def oracle(code: str, df: pd.DataFrame, n_cuts: int = 15, seed: int = 0) -> bool:
    """执行判卷：扰动 cut 之后的行情，cut 及以前的信号变了 = 有前视。

    多个切点（含预热期，覆盖 bfill 这类只影响开头的 bug）× 多种扰动。
    这是充分不必要条件：判出前视一定是真的；没判出只说明在这些扰动下没暴露。
    """
    base = run_signal(code, df).to_numpy()
    rng = np.random.default_rng(seed)
    cols = [df.columns.get_loc(k) for k in ("open", "high", "low", "close")]
    cuts = sorted({10, 30, 60, *np.linspace(100, len(df) - 50, n_cuts).astype(int)})
    for cut in cuts:
        k = len(df) - cut - 1
        for factor in _perturbations(k, rng):
            d2 = df.copy()
            d2.iloc[cut + 1:, cols] = d2.iloc[cut + 1:, cols].to_numpy() * factor[:, None]
            s2 = run_signal(code, d2).to_numpy()
            if not np.allclose(base[:cut + 1], s2[:cut + 1], equal_nan=True):
                return True
    return False
