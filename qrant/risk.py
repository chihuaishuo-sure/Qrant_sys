"""风控层：把策略的原始仓位改造成“更不容易大亏”的最终目标仓位。

顺序：原始信号 → 大盘趋势过滤 → ATR 跟踪止损 → 波动率目标缩放 → [0, 上限]。
回撤熔断依赖账户净值路径，放在回测引擎里执行。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RiskConfig, TRADING_DAYS
from .indicators import atr, realized_vol, sma


@dataclass
class RiskOutput:
    raw: pd.Series
    regime_cap: pd.Series
    stop_level: pd.Series
    vol_scale: pd.Series
    final: pd.Series
    realized_vol: pd.Series


def regime_cap(index: pd.DatetimeIndex, bench_close: pd.Series, ma: int, bear_exposure: float) -> pd.Series:
    b = bench_close.reindex(index.union(bench_close.index)).ffill().reindex(index)
    m = sma(bench_close, ma).reindex(index.union(bench_close.index)).ffill().reindex(index)
    cap = pd.Series(1.0, index=index)
    cap[(b < m) & m.notna()] = bear_exposure
    return cap


def atr_trailing_stop(df: pd.DataFrame, inpos: pd.Series, period: int, mult: float) -> tuple[pd.Series, pd.Series]:
    """基于收盘价的 ATR 跟踪止损。

    - 持仓期间记录最高收盘价，止损线 = 最高收盘 − mult × ATR
    - 收盘跌破止损线 → 当日出信号，次日开盘卖出（因此包含跳空风险，不美化止损成交价）
    - 止损后需原始信号仍在、且收盘重新突破止损前高点，才重新入场（避免来回打脸）
    返回 (允许持仓的 0/1 序列, 止损线)。
    """
    c = df["close"].to_numpy()
    a = atr(df, period).to_numpy()
    want = inpos.to_numpy() > 0
    allow = np.zeros(len(c))
    level = np.full(len(c), np.nan)
    peak = np.nan
    stopped = False
    stop_peak = np.nan
    for i in range(len(c)):
        if not want[i]:
            stopped, peak = False, np.nan
            continue
        if stopped:
            if c[i] > stop_peak:
                stopped, peak = False, c[i]
            else:
                continue
        peak = c[i] if np.isnan(peak) else max(peak, c[i])
        if np.isnan(a[i]):
            allow[i] = 1.0
            continue
        level[i] = peak - mult * a[i]
        if c[i] < level[i]:
            stopped, stop_peak = True, peak
        else:
            allow[i] = 1.0
    return pd.Series(allow, index=df.index), pd.Series(level, index=df.index)


def vol_scale(close: pd.Series, cfg: RiskConfig, ppy: int = TRADING_DAYS) -> tuple[pd.Series, pd.Series]:
    """波动率目标：仓位 = 目标波动 / 实际波动（取短、长期较大者，偏保守），不超过上限。"""
    v = pd.concat([realized_vol(close, cfg.vol_lookback_short, ppy),
                   realized_vol(close, cfg.vol_lookback_long, ppy)], axis=1).max(axis=1)
    scale = (cfg.target_vol / v.replace(0, np.nan)).clip(upper=cfg.max_exposure)
    return scale.fillna(0.0), v


def apply_risk(df: pd.DataFrame, raw: pd.Series, cfg: RiskConfig,
               bench_close: pd.Series | None = None, ppy: int = TRADING_DAYS) -> RiskOutput:
    raw = raw.reindex(df.index).fillna(0.0).clip(0, 1)
    if cfg.regime_on:
        cap = regime_cap(df.index, bench_close if bench_close is not None else df["close"],
                         cfg.regime_ma, cfg.bear_exposure)
    else:
        cap = pd.Series(1.0, index=df.index)
    x = np.minimum(raw, cap)

    if cfg.atr_stop_on:
        allow, level = atr_trailing_stop(df, x, cfg.atr_period, cfg.atr_mult)
        x = x * allow
    else:
        level = pd.Series(np.nan, index=df.index)

    if cfg.vol_target_on:
        scale, v = vol_scale(df["close"], cfg, ppy)
    else:
        scale = pd.Series(cfg.max_exposure, index=df.index)
        v = realized_vol(df["close"], cfg.vol_lookback_short, ppy)
    final = (x * scale).clip(0.0, cfg.max_exposure)
    return RiskOutput(raw, cap, level, scale, final, v)


# ---------------------------------------------------------------- 个人风控工具
def position_size(account: float, risk_pct: float, entry: float, stop: float,
                  max_position_pct: float = 0.3, lot: int = 100) -> dict:
    """按“单笔最多亏账户的 risk_pct”计算可买股数（整手）。"""
    if entry <= 0 or stop <= 0 or stop >= entry:
        return {"ok": False, "msg": "止损价必须低于买入价且都大于 0"}
    per_share_risk = entry - stop
    by_risk = account * risk_pct / per_share_risk
    by_cap = account * max_position_pct / entry
    shares = int(min(by_risk, by_cap) // lot * lot)
    limited_by = "单笔风险" if by_risk <= by_cap else "单只仓位上限"
    return {
        "ok": shares > 0,
        "shares": shares,
        "value": shares * entry,
        "position_pct": shares * entry / account if account else 0,
        "loss_if_stopped": shares * per_share_risk,
        "loss_pct": shares * per_share_risk / account if account else 0,
        "stop_distance_pct": per_share_risk / entry,
        "limited_by": limited_by,
        "msg": "" if shares > 0 else "按当前风险设置，连 1 手都不应买入",
    }


def var_cvar(returns: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """历史法 VaR / CVaR（返回正数表示亏损比例）。"""
    r = returns.dropna()
    if len(r) < 20:
        return float("nan"), float("nan")
    q = np.quantile(r, 1 - level)
    tail = r[r <= q]
    return float(-q), float(-tail.mean()) if len(tail) else float("nan")
