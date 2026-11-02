"""实盘记录与信号偏差跟踪：你的实际成交 vs 系统信号（纸面）vs 买入持有。

小额实盘验证的是“执行流程”而不是策略收益。这里回答三个问题：
1. 执行率：系统要求的每一笔交易，你有没有按时做（当天 / 延迟 / 没做）？有没有做计划外的交易？
2. 滑点：实际成交价比系统假设的开盘价差多少？
3. 收益差：实际账户 vs 按信号严格执行的纸面账户，差距多大、来自哪里？

价格口径：实际账户用不复权价格估值，分红作为现金收入记录（方向填“分红”，金额填到账现金）；
纸面账户用前复权价格（分红自动再投资），两者在总收益上可比。
数据存在 journal/（已在 .gitignore 中，个人交易记录不会上传到 GitHub）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as qdata
from .config import CostModel, RiskConfig, is_ashare_equity
from .portfolio import rebalance_dates
from .risk import apply_risk
from .strategies import STRATEGIES

DIR = Path(__file__).resolve().parent.parent / "journal"
COLS = ["日期", "代码", "方向", "数量", "成交价", "费用", "金额", "备注"]
SIDES = ("买入", "卖出", "分红", "未执行")
MATCH_DAYS = 2  # 信号后 2 个交易日内执行算“延迟”，超过算“未执行”


@dataclass
class Account:
    起始资金: float = 3000.0
    开始日期: str = field(default_factory=lambda: date.today().isoformat())
    跟踪标的: list[str] = field(default_factory=lambda: ["510300"])
    策略: str = "multi_momentum"
    最低佣金: float = 0.0  # 免五账户填 0；否则填 5
    佣金率: float = 0.00025
    现金年化: float = 0.0  # 空仓资金买了货币基金/逆回购才填（收益按“分红”记），否则纸面账户也不计息

    def cost(self, code: str) -> CostModel:
        return CostModel.for_symbol(code, min_commission=self.最低佣金, commission_rate=self.佣金率)


# ------------------------------------------------------------------ 存取
def _paths(folder: Path | None) -> tuple[Path, Path]:
    folder = folder or DIR  # 调用时再取 DIR，方便测试或迁移目录
    return folder / "账户.json", folder / "交易记录.csv"


def init(acc: Account, folder: Path | None = None) -> None:
    folder = folder or DIR
    folder.mkdir(parents=True, exist_ok=True)
    a, t = _paths(folder)
    a.write_text(json.dumps(asdict(acc), ensure_ascii=False, indent=2), encoding="utf-8")
    if not t.exists():
        pd.DataFrame(columns=COLS).to_csv(t, index=False, encoding="utf-8-sig")


def load_account(folder: Path | None = None) -> Account | None:
    a, _ = _paths(folder)
    return Account(**json.loads(a.read_text(encoding="utf-8"))) if a.exists() else None


def load_trades(folder: Path | None = None) -> pd.DataFrame:
    _, t = _paths(folder)
    if not t.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(t, dtype={"代码": str}, encoding="utf-8-sig")
    df["代码"] = df["代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    for k in ("数量", "成交价", "费用", "金额"):
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)
    df["备注"] = df["备注"].fillna("")
    return df.sort_values("日期", kind="stable").reset_index(drop=True)


def add_trade(day, code: str, side: str, qty: float = 0, price: float = 0, fee: float = 0,
              amount: float = 0, note: str = "", folder: Path | None = None) -> dict:
    """记一笔。买卖必须是 100 股整数倍；分红填金额；未执行只需日期、代码和原因。"""
    code = str(code).zfill(6)
    if side not in SIDES:
        raise ValueError(f"方向必须是 {SIDES} 之一")
    if side in ("买入", "卖出"):
        if qty <= 0 or qty % 100:
            raise ValueError("数量必须是 100 的正整数倍")
        if price <= 0:
            raise ValueError("成交价必须大于 0")
        amount = qty * price
    if side == "分红" and amount <= 0:
        raise ValueError("分红请填到账金额")
    row = {"日期": pd.Timestamp(day).date().isoformat(), "代码": code, "方向": side, "数量": qty,
           "成交价": price, "费用": fee, "金额": round(amount, 2), "备注": note}
    _, t = _paths(folder)
    if not t.exists():
        raise FileNotFoundError("还没有初始化账户，请先运行 python trade_log.py init")
    pd.DataFrame([row], columns=COLS).to_csv(t, mode="a", header=False, index=False, encoding="utf-8-sig")
    return row


# ------------------------------------------------------------------ 账户重建
def holdings(trades: pd.DataFrame, capital: float, until=None) -> tuple[float, dict[str, float]]:
    """截至某日（含）的现金和持股。"""
    t = trades if until is None else trades[trades["日期"] <= pd.Timestamp(until)]
    cash, sh = float(capital), {}
    for r in t.itertuples(index=False):
        if r.方向 == "买入":
            cash -= r.数量 * r.成交价 + r.费用
            sh[r.代码] = sh.get(r.代码, 0) + r.数量
        elif r.方向 == "卖出":
            cash += r.数量 * r.成交价 - r.费用
            sh[r.代码] = sh.get(r.代码, 0) - r.数量
        elif r.方向 == "分红":
            cash += r.金额
    return cash, {k: v for k, v in sh.items() if v}


def dividend_events(codes: list[str], start) -> pd.DataFrame:
    """开始日以来各标的的除息事件（每份分红），来自新浪复权因子；取不到时返回空表。"""
    rows = []
    for c in codes:
        try:
            f = qdata.sina_factors(qdata._exchange_prefix(c))
        except Exception:  # noqa: BLE001
            continue
        f["每份分红"] = f["u"].diff().fillna(f["u"])
        f = f[(f["date"] >= pd.Timestamp(start)) & (f["每份分红"] > 1e-9)]
        rows += [{"日期": d, "代码": c, "每份分红": v} for d, v in zip(f["date"], f["每份分红"])]
    return pd.DataFrame(rows, columns=["日期", "代码", "每份分红"])


def with_dividends(trades: pd.DataFrame, capital: float, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分红由交易所自动发放，很少有人记账：按除息前一天的实际持股自动补上。
    你自己记了的（同一标的、除息日前后 30 天内有“分红”记录）以你的为准。返回 (补全后的记录, 自动补的分红)。"""
    auto = []
    for e in events.itertuples(index=False):
        mine = trades[(trades["方向"] == "分红") & (trades["代码"] == e.代码)
                      & ((trades["日期"] - e.日期).abs() <= pd.Timedelta(days=30))]
        if len(mine):
            continue
        _, sh = holdings(trades, capital, e.日期 - pd.Timedelta(days=1))
        q = sh.get(e.代码, 0)
        if q > 0:
            auto.append({"日期": e.日期, "代码": e.代码, "方向": "分红", "数量": q, "成交价": 0.0, "费用": 0.0,
                         "金额": round(q * e.每份分红, 2), "备注": f"自动：每份 {e.每份分红:.3f} 元"})
    auto_df = pd.DataFrame(auto, columns=COLS)
    full = pd.concat([trades, auto_df], ignore_index=True) if len(auto_df) else trades
    return full.sort_values("日期", kind="stable").reset_index(drop=True), auto_df


def actual_equity(trades: pd.DataFrame, capital: float, raw_close: pd.DataFrame) -> pd.Series:
    """实际账户每日净值（不复权收盘估值 + 现金）。"""
    out = []
    for d in raw_close.index:
        cash, sh = holdings(trades, capital, d)
        out.append(cash + sum(q * raw_close.at[d, c] for c, q in sh.items() if c in raw_close))
    return pd.Series(out, index=raw_close.index, name="实际账户")


# ------------------------------------------------------------------ 纸面账户
def _decide(index: pd.DatetimeIndex, target: pd.Series, freq: str) -> pd.Series:
    """与 pipeline.run_target 相同：周度决策，但清仓信号次日立即执行。"""
    if freq != "W":
        return pd.Series(True, index=index)
    t = target.reindex(index).ffill().fillna(0.0)
    exit_now = (t <= 0.01) & (t.shift(1).fillna(0) > 0.01)
    return rebalance_dates(index, "W") | exit_now


def paper(acc: Account, adj: dict[str, pd.DataFrame], bench: pd.Series | None, cfg: RiskConfig | None = None) -> dict:
    """按信号严格执行的纸面账户：资金平分到每个跟踪标的，各自按默认风控回测。"""
    from . import backtest as bt

    from dataclasses import replace

    cfg = replace(cfg or RiskConfig(), cash_rate=acc.现金年化)
    start = pd.Timestamp(acc.开始日期)
    each = acc.起始资金 / len(acc.跟踪标的)
    eqs, bhs, trades, targets = [], [], [], {}
    for code in acc.跟踪标的:
        df = adj[code]
        raw = STRATEGIES[acc.策略](df)
        tgt = apply_risk(df, raw, cfg, bench if is_ashare_equity(code) else None).final
        targets[code] = tgt
        d = df.loc[start:]
        if len(d) == 0:
            continue
        r = bt.run({code: d}, tgt.reindex(d.index).to_frame(code), each, {code: acc.cost(code)},
                   cfg.rebalance_band, cfg, "纸面", _decide(d.index, tgt, cfg.decide_freq), cfg.cash_rate)
        eqs.append(r.equity)
        trades.append(r.trades)
        h = bt.run({code: d}, pd.Series(1.0, index=d.index).to_frame(code), each, {code: acc.cost(code)}, 0.05,
                   cash_rate=acc.现金年化)
        bhs.append(h.equity)
    if not eqs:
        return {"equity": pd.Series(dtype=float), "buy_hold": pd.Series(dtype=float),
                "trades": pd.DataFrame(), "targets": targets}
    return {"equity": pd.concat(eqs, axis=1).ffill().sum(axis=1).rename("纸面账户"),
            "buy_hold": pd.concat(bhs, axis=1).ffill().sum(axis=1).rename("买入持有"),
            "trades": pd.concat(trades, ignore_index=True), "targets": targets}


# ------------------------------------------------------------------ 偏差
def match(model: pd.DataFrame, actual: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """把系统的每一笔交易和你的实际交易配对：当天 = 按时，2 个交易日内 = 延迟，否则 = 未执行。"""
    model = model[model["方向"].isin(["买入", "卖出"])] if len(model) else model
    acts = actual[actual["方向"].isin(["买入", "卖出"])].copy()
    acts["已配对"] = False
    pos = {d: i for i, d in enumerate(calendar)}
    rows = []
    for m in model.itertuples(index=False):
        i0 = pos.get(pd.Timestamp(m.日期))
        cand = acts[(~acts["已配对"]) & (acts["代码"] == m.代码) & (acts["方向"] == m.方向)]
        hit = None
        for j, a in cand.iterrows():
            ia = pos.get(pd.Timestamp(a["日期"]))
            if i0 is not None and ia is not None and 0 <= ia - i0 <= MATCH_DAYS:
                hit = (j, ia - i0)
                break
        skip = actual[(actual["方向"] == "未执行") & (actual["代码"] == m.代码)]
        if i0 is not None:
            skip = skip[[0 <= pos.get(pd.Timestamp(d), -99) - i0 <= MATCH_DAYS for d in skip["日期"]]]
        row = {"信号成交日": pd.Timestamp(m.日期).date(), "代码": m.代码, "方向": m.方向, "系统数量": m.数量}
        if hit:
            j, lag = hit
            acts.at[j, "已配对"] = True
            a = acts.loc[j]
            row |= {"执行": "按时" if lag == 0 else f"延迟{lag}天", "实际日期": a["日期"].date(),
                    "实际数量": a["数量"], "实际价": a["成交价"], "_idx": j}
        else:
            row |= {"执行": "未执行", "备注": "；".join(skip["备注"]) if len(skip) else ""}
        rows.append(row)
    extra = acts[~acts["已配对"]]
    for _, a in extra.iterrows():
        rows.append({"信号成交日": None, "代码": a["代码"], "方向": a["方向"], "系统数量": 0,
                     "执行": "计划外", "实际日期": a["日期"].date(), "实际数量": a["数量"], "实际价": a["成交价"]})
    return pd.DataFrame(rows)


def slippage_bp(m: pd.DataFrame, raw_open: pd.DataFrame) -> pd.Series:
    """实际成交价相对当天不复权开盘价（系统假设的成交价）的偏离，单位基点；正数 = 对你不利。"""
    out = {}
    for i, r in m.iterrows():
        if r.get("执行", "") in ("未执行", "计划外") or pd.isna(r.get("实际价")):
            continue
        d, c = pd.Timestamp(r["实际日期"]), r["代码"]
        if c not in raw_open or d not in raw_open.index or not raw_open.at[d, c] > 0:
            continue
        diff = r["实际价"] / raw_open.at[d, c] - 1
        out[i] = (diff if r["方向"] == "买入" else -diff) * 1e4
    return pd.Series(out, dtype=float)


# ------------------------------------------------------------------ 汇总
def load_prices(codes: list[str], start: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """前复权全历史（给信号用）+ 开始日以来的不复权开盘/收盘（给实际账户用）。"""
    adj = {c: qdata.fetch(c, "20120101")[0] for c in codes}
    raw = {c: qdata.fetch(c, start, adjust="")[0] for c in codes}
    o = pd.DataFrame({c: r["open"] for c, r in raw.items()}).ffill()
    cl = pd.DataFrame({c: r["close"] for c, r in raw.items()}).ffill()
    return adj, o, cl


def analyze(acc: Account, trades: pd.DataFrame, adj: dict, raw_open: pd.DataFrame, raw_close: pd.DataFrame,
            bench: pd.Series | None = None, events: pd.DataFrame | None = None) -> dict:
    start = pd.Timestamp(acc.开始日期)
    raw_close = raw_close.loc[start:]
    pp = paper(acc, adj, bench)
    m = match(pp["trades"], trades, raw_close.index)
    slip = slippage_bp(m, raw_open)
    if events is None:
        events = dividend_events(acc.跟踪标的, start)
    full, auto_div = with_dividends(trades, acc.起始资金, events)
    act = actual_equity(full, acc.起始资金, raw_close)
    n_model = int((m["执行"] != "计划外").sum()) if len(m) else 0
    summary = {
        "交易日数": len(raw_close),
        "实际收益": act.iloc[-1] / acc.起始资金 - 1 if len(act) else np.nan,
        "纸面收益": pp["equity"].iloc[-1] / acc.起始资金 - 1 if len(pp["equity"]) else np.nan,
        "持有收益": pp["buy_hold"].iloc[-1] / acc.起始资金 - 1 if len(pp["buy_hold"]) else np.nan,
        "系统交易数": n_model,
        "按时": int((m["执行"] == "按时").sum()) if len(m) else 0,
        "延迟": int(m["执行"].str.startswith("延迟").sum()) if len(m) else 0,
        "未执行": int((m["执行"] == "未执行").sum()) if len(m) else 0,
        "计划外": int((m["执行"] == "计划外").sum()) if len(m) else 0,
        "平均滑点bp": float(slip.mean()) if len(slip) else np.nan,
        "实际费用": float(trades.loc[trades["方向"].isin(["买入", "卖出"]), "费用"].sum()),
        "纸面费用": float(pp["trades"]["费用"].sum()) if len(pp["trades"]) else 0.0,
        "自动补记分红": float(auto_div["金额"].sum()) if len(auto_div) else 0.0,
    }
    summary["执行差"] = summary["实际收益"] - summary["纸面收益"]
    if len(m):
        m["滑点bp"] = slip.reindex(m.index)
        m = m.drop(columns=["_idx"], errors="ignore")
    curves = pd.concat([act, pp["equity"], pp["buy_hold"]], axis=1).ffill()
    return {"summary": summary, "matches": m, "curves": curves, "paper": pp, "auto_dividends": auto_div}


def next_actions(acc: Account, trades: pd.DataFrame, adj: dict, raw_close: pd.DataFrame,
                 bench: pd.Series | None = None, cfg: RiskConfig | None = None) -> pd.DataFrame:
    """按今天收盘的信号和你的实际持仓，明天开盘该做什么（与回测引擎同一套规则）。"""
    cfg = cfg or RiskConfig()
    cash, sh = holdings(trades, acc.起始资金)
    last = raw_close.ffill().iloc[-1]
    equity = cash + sum(q * last[c] for c, q in sh.items() if c in last)
    each = equity / len(acc.跟踪标的)
    rows = []
    for code in acc.跟踪标的:
        df = adj[code]
        tgt = apply_risk(df, STRATEGIES[acc.策略](df), cfg, bench if is_ashare_equity(code) else None).final
        decide = bool(_decide(df.index, tgt, cfg.decide_freq).iloc[-1])
        tw, px, cur = float(tgt.iloc[-1]), float(last[code]), float(sh.get(code, 0))
        cur_w = cur * px / each if each > 0 else 0.0
        want = np.floor(tw * each / (px * 100)) * 100
        need = decide and ((tw <= 1e-9 < cur) or (cur == 0 and tw > 1e-9 and want > 0) or abs(tw - cur_w) > cfg.rebalance_band)
        act = "不动"
        if need and want != cur:
            act = f"买入 {int(want - cur)} 股" if want > cur else f"卖出 {int(cur - want)} 股"
        rows.append({"代码": code, "信号日": df.index[-1].date(), "目标仓位": tw, "当前持股": int(cur),
                     "目标持股": int(want) if need else int(cur), "明日开盘": act,
                     "说明": "" if decide else "非决策日（周度决策，只有清仓信号会立即执行）"})
    return pd.DataFrame(rows)


def report_markdown(acc: Account, res: dict, actions: pd.DataFrame | None = None) -> str:
    s = res["summary"]

    def p(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2%}"
    n = s["系统交易数"]
    slip = "—" if np.isnan(s["平均滑点bp"]) else f"{s['平均滑点bp']:.1f} 个基点（正数 = 对你不利）"
    lines = [f"# 实盘对比 · {date.today():%Y-%m-%d}", "",
             f"起始资金 {acc.起始资金:,.0f} 元，开始于 {acc.开始日期}，跟踪 {', '.join(acc.跟踪标的)}，策略 {acc.策略}，"
             f"最低佣金 {acc.最低佣金:g} 元。共 {s['交易日数']} 个交易日。", "",
             "| 指标 | 结果 |", "|:--|:--|",
             f"| 实际账户收益 | {p(s['实际收益'])} |",
             f"| 纸面账户收益（严格按信号） | {p(s['纸面收益'])} |",
             f"| 买入持有收益 | {p(s['持有收益'])} |",
             f"| **执行差**（实际 − 纸面） | **{p(s['执行差'])}** |",
             f"| 执行率 | 系统交易 {n} 笔：按时 {s['按时']}，延迟 {s['延迟']}，未执行 {s['未执行']} |",
             f"| 计划外交易 | {s['计划外']} 笔 |",
             f"| 平均滑点 | {slip} |",
             f"| 费用 | 实际 {s['实际费用']:.2f} 元，纸面 {s['纸面费用']:.2f} 元 |",
             f"| 自动补记的分红 | {s['自动补记分红']:.2f} 元（交易所自动发放，按除息前持股计算） |", ""]
    if actions is not None and len(actions):
        a = actions.copy()
        a["目标仓位"] = a["目标仓位"].map(lambda v: f"{v:.0%}")
        lines += ["## 明日开盘操作", "", a.to_markdown(index=False), ""]
    if len(res["matches"]):
        lines += ["## 信号与执行明细", "", res["matches"].to_markdown(index=False), ""]
    lines += ["> 小额实盘检验的是执行流程，不是策略收益；几个月的数据不足以判断策略好坏。"]
    return "\n".join(lines)
