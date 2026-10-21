"""多资产组合：反波动率/等权配置 + 单资产趋势过滤（Faber GTAA 思路）+ 组合波动率目标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .indicators import sma
from .strategies import multi_momentum


def _cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """单资产权重上限，超出部分按比例分给其他资产（迭代）。"""
    w = w.copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    cap = max(cap, 1.0 / max((w > 0).sum(), 1))
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = (~over) & (w > 0)
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.Series:
    """每个周期最后一个交易日为决策日。"""
    s = pd.Series(index, index=index)
    last = s.groupby(index.to_period("M" if freq == "ME" else "W")).max()
    return pd.Series(index.isin(last.to_numpy()), index=index)


def portfolio_targets(
    data: dict[str, pd.DataFrame],
    method: str = "inv_vol",
    trend: str = "momentum",
    trend_ma: int = 200,
    vol_lookback: int = 60,
    target_vol: float | None = 0.10,
    max_weight: float = 0.40,
    freq: str = "ME",
    ppy: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    """返回 (每日目标权重, 决策日掩码)。被趋势过滤掉的权重留作现金，不再分给其他资产。"""
    syms = list(data)
    close = pd.concat({s: data[s]["close"] for s in syms}, axis=1).sort_index().ffill()
    rets = close.pct_change()
    vols = rets.rolling(vol_lookback, min_periods=vol_lookback // 2).std() * np.sqrt(ppy)

    if trend == "ma":
        tf = pd.concat({s: (close[s] > sma(close[s], trend_ma)).astype(float) for s in syms}, axis=1)
    elif trend == "momentum":
        tf = pd.concat({s: multi_momentum(data[s].reindex(close.index).ffill()) for s in syms}, axis=1)
    else:
        tf = pd.DataFrame(1.0, index=close.index, columns=syms)

    decide = rebalance_dates(close.index, freq)
    W = pd.DataFrame(np.nan, index=close.index, columns=syms)
    for t in close.index[decide.to_numpy()]:
        v = vols.loc[t].to_numpy()
        avail = ~np.isnan(v) & ~np.isnan(close.loc[t].to_numpy())
        if not avail.any():
            W.loc[t] = 0.0
            continue
        if method == "inv_vol":
            base = np.where(avail, 1.0 / np.where(v > 0, v, np.nan), 0.0)
            base = np.nan_to_num(base)
        else:
            base = avail.astype(float)
        w = _cap_weights(base, max_weight)
        w = w * tf.loc[t].fillna(0.0).to_numpy()
        if target_vol:
            win = rets.loc[:t].tail(vol_lookback).fillna(0.0)
            cov = win.cov().to_numpy() * ppy
            pv = float(np.sqrt(max(w @ cov @ w, 0)))
            if pv > 0:
                w = w * min(1.0, target_vol / pv)
        W.loc[t] = w
    W = W.ffill().fillna(0.0)
    return W, decide
