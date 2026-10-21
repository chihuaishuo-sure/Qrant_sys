"""绩效与风险指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .indicators import drawdown


def _years(eq: pd.Series, ppy: int) -> float:
    return max(len(eq) - 1, 1) / ppy


def max_underwater(eq: pd.Series) -> int:
    """最长水下期（距离前高的最长持续周期数）。"""
    uw = (eq < eq.cummax()).to_numpy()
    best = cur = 0
    for u in uw:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def episodes(returns: pd.Series, exposure: pd.Series) -> list[float]:
    """把“持仓—空仓”分段，返回每段持仓的收益（用于胜率/盈亏比）。"""
    out, acc, inpos = [], 1.0, False
    for r, e in zip(returns.to_numpy(), exposure.shift(1).fillna(0).to_numpy()):
        if e > 1e-6:
            acc *= 1 + r
            inpos = True
        elif inpos:
            out.append(acc - 1)
            acc, inpos = 1.0, False
    if inpos:
        out.append(acc - 1)
    return out


def summary(eq: pd.Series, exposure: pd.Series | None = None, trades: pd.DataFrame | None = None,
            ppy: int = TRADING_DAYS, rf: float = 0.0) -> dict:
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    yrs = _years(eq, ppy)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(ppy)
    ex = r - rf / ppy
    sharpe = ex.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else np.nan
    down = r[r < 0]
    dstd = np.sqrt((down**2).sum() / max(len(r), 1)) * np.sqrt(ppy)
    sortino = (r.mean() * ppy - rf) / dstd if dstd > 0 else np.nan
    dd = drawdown(eq)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    ulcer = np.sqrt((dd**2).mean())
    q = np.quantile(r, 0.05) if len(r) > 20 else np.nan
    cvar = -r[r <= q].mean() if len(r) > 20 else np.nan
    monthly = eq.resample("ME").last().pct_change().dropna()
    out = {
        "累计收益": total,
        "年化收益": cagr,
        "年化波动": vol,
        "夏普比率": sharpe,
        "索提诺比率": sortino,
        "最大回撤": mdd,
        "卡玛比率": calmar,
        "Ulcer指数": ulcer,
        "最长水下(天)": max_underwater(eq),
        "日VaR95": -q if not np.isnan(q) else np.nan,
        "日CVaR95": cvar,
        "最差单日": r.min() if len(r) else np.nan,
        "月度胜率": (monthly > 0).mean() if len(monthly) else np.nan,
        "最差月份": monthly.min() if len(monthly) else np.nan,
    }
    if exposure is not None:
        out["平均仓位"] = exposure.reindex(eq.index).fillna(0).mean()
        ep = episodes(r.reindex(eq.index).fillna(0), exposure.reindex(eq.index).fillna(0))
        wins = [x for x in ep if x > 0]
        loss = [x for x in ep if x <= 0]
        out["持仓段数"] = len(ep)
        out["持仓胜率"] = len(wins) / len(ep) if ep else np.nan
        out["盈亏比"] = (np.mean(wins) / abs(np.mean(loss))) if wins and loss and np.mean(loss) != 0 else np.nan
    if trades is not None and len(trades):
        done = trades[trades["方向"].isin(["买入", "卖出"])]
        out["成交笔数"] = len(done)
        out["年均换手"] = done["金额"].sum() / eq.mean() / yrs if yrs > 0 else np.nan
    return out


PCT_KEYS = {"累计收益", "年化收益", "年化波动", "最大回撤", "Ulcer指数", "日VaR95", "日CVaR95",
            "最差单日", "月度胜率", "最差月份", "平均仓位", "持仓胜率"}


def fmt(k: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if k in PCT_KEYS:
        return f"{v * 100:.2f}%"
    if k in {"年均换手"}:
        return f"{v:.1f}x"
    if isinstance(v, (int, np.integer)) or k in {"最长水下(天)", "持仓段数", "成交笔数"}:
        return f"{int(v)}"
    return f"{v:.2f}"


def table(results: dict[str, dict]) -> pd.DataFrame:
    keys = list(next(iter(results.values())).keys())
    for v in results.values():
        for k in v:
            if k not in keys:
                keys.append(k)
    return pd.DataFrame({name: [fmt(k, m.get(k)) for k in keys] for name, m in results.items()}, index=keys)


def monthly_table(eq: pd.Series) -> pd.DataFrame:
    """年 × 月 收益矩阵，最后一列为全年。"""
    m = eq.resample("ME").last()
    first = eq.iloc[:1]
    m = pd.concat([first, m])
    m = m[~m.index.duplicated(keep="last")].pct_change().dropna()
    t = pd.DataFrame({"年": m.index.year, "月": m.index.month, "r": m.to_numpy()})
    pv = t.pivot_table(index="年", columns="月", values="r", aggfunc="sum")
    pv = pv.reindex(columns=range(1, 13))
    yearly = eq.resample("YE").last()
    yearly = pd.concat([first, yearly])
    yearly = yearly[~yearly.index.duplicated(keep="last")].pct_change().dropna()
    pv["全年"] = pd.Series(yearly.to_numpy(), index=yearly.index.year)
    pv.columns = [f"{c}月" if isinstance(c, int) else c for c in pv.columns]
    return pv
