"""策略库：每个策略返回 t 日收盘时决定的“目标仓位”（0~1，A股个人只做多）。

成交统一由回测引擎在 t+1 日开盘执行，因此这里可以放心使用 t 日收盘数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .indicators import donchian, rsi, sma


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def ma_trend(df: pd.DataFrame, fast: int = 20, slow: int = 120) -> pd.Series:
    """均线趋势：快线在慢线上方持有。fast=1 时即“价格站上慢线”。"""
    c = df["close"]
    f = c if fast <= 1 else sma(c, fast)
    s = sma(c, slow)
    return (f > s).astype(float).where(s.notna(), 0.0)


def multi_momentum(df: pd.DataFrame, short: int = 20, mid: int = 60, long: int = 120,
                   longest: int = 250) -> pd.Series:
    """多周期时间序列动量（TSMOM）：4 个回看期中动量为正的比例即仓位。

    比单一周期更稳健：不依赖某一个“最优参数”，信号是渐进的 0/25/50/75/100%。
    """
    c = df["close"]
    votes = []
    for n in (short, mid, long, longest):
        votes.append((c / c.shift(n) - 1 > 0).astype(float).where(c.shift(n).notna()))
    v = pd.concat(votes, axis=1)
    return v.mean(axis=1, skipna=True).fillna(0.0)


def donchian_breakout(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    """唐奇安通道突破（海龟）：收盘突破前 entry 日最高价买入，跌破前 exit 日最低价卖出。"""
    hi, _ = donchian(df, entry)
    _, lo = donchian(df, exit)
    c = df["close"].to_numpy()
    hi_a, lo_a = hi.to_numpy(), lo.to_numpy()
    pos = np.zeros(len(c))
    state = 0.0
    for i in range(len(c)):
        if state == 0 and not np.isnan(hi_a[i]) and c[i] > hi_a[i]:
            state = 1.0
        elif state == 1 and not np.isnan(lo_a[i]) and c[i] < lo_a[i]:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def rsi_reversion(df: pd.DataFrame, n: int = 6, buy_below: int = 25, sell_above: int = 60,
                  trend_ma: int = 200) -> pd.Series:
    """RSI 均值回归 + 长期趋势过滤：只在长期上升趋势中“逢跌买入”，避免熊市抄底。"""
    c = df["close"]
    r = rsi(c, n).to_numpy()
    up = (c > sma(c, trend_ma)).to_numpy()
    pos = np.zeros(len(c))
    state = 0.0
    for i in range(len(c)):
        if np.isnan(r[i]):
            pos[i] = 0.0
            continue
        if state == 0 and r[i] < buy_below and up[i]:
            state = 1.0
        elif state == 1 and (r[i] > sell_above or not up[i]):
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def trend_ensemble(df: pd.DataFrame) -> pd.Series:
    """三种趋势信号取平均：多周期动量 + 均线趋势 + 唐奇安突破。分散单一规则的打脸风险。"""
    return (multi_momentum(df) + ma_trend(df) + donchian_breakout(df)) / 3.0


@dataclass
class Param:
    key: str
    label: str
    default: int | float
    lo: int | float
    hi: int | float
    step: int | float = 1


@dataclass
class StrategySpec:
    key: str
    name: str
    fn: Callable[..., pd.Series]
    desc: str
    params: list[Param] = field(default_factory=list)

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def __call__(self, df: pd.DataFrame, **kw) -> pd.Series:
        args = self.defaults()
        args.update({k: v for k, v in kw.items() if k in args})
        return self.fn(df, **args).clip(0.0, 1.0)


STRATEGIES: dict[str, StrategySpec] = {
    s.key: s
    for s in [
        StrategySpec(
            "multi_momentum", "多周期动量（推荐）", multi_momentum,
            "20/60/120/250 日四个周期的动量投票，正动量越多仓位越高。参数鲁棒、换手低，"
            "学术证据最充分（Moskowitz 等 2012）。",
            [Param("short", "短周期", 20, 5, 60), Param("mid", "中周期", 60, 20, 120),
             Param("long", "长周期", 120, 60, 250), Param("longest", "最长周期", 250, 120, 500)],
        ),
        StrategySpec(
            "ma_trend", "均线趋势", ma_trend,
            "快线在慢线上方持有，否则空仓。fast=1 即经典“价格站上 N 日均线”（Faber 择时）。",
            [Param("fast", "快线", 20, 1, 100), Param("slow", "慢线", 120, 20, 300)],
        ),
        StrategySpec(
            "donchian", "唐奇安突破", donchian_breakout,
            "突破前 N 日最高价入场、跌破前 M 日最低价离场，海龟交易法核心。",
            [Param("entry", "入场周期", 55, 10, 120), Param("exit", "离场周期", 20, 5, 60)],
        ),
        StrategySpec(
            "rsi_reversion", "RSI 回调买入", rsi_reversion,
            "只在长期上升趋势中，短期超跌（RSI 低）时买入、反弹后卖出。",
            [Param("n", "RSI周期", 6, 2, 30), Param("buy_below", "买入阈值", 25, 5, 45),
             Param("sell_above", "卖出阈值", 60, 50, 90), Param("trend_ma", "趋势均线", 200, 50, 300)],
        ),
        StrategySpec(
            "ensemble", "趋势组合信号", trend_ensemble,
            "多周期动量、均线趋势、唐奇安突破三者取平均：信号更平滑，减少单一规则在震荡市的来回打脸。", [],
        ),
        StrategySpec("buy_hold", "买入持有", buy_hold, "满仓持有，作为基准。", []),
    ]
}
