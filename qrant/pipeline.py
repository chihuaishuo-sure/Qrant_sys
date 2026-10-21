"""高层流程：界面和测试都调用这里。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import CostModel, RiskConfig, TRADING_DAYS, is_ashare_equity
from .metrics import summary
from .portfolio import portfolio_targets, rebalance_dates
from .risk import RiskOutput, apply_risk
from .strategies import STRATEGIES


@dataclass
class SingleAnalysis:
    symbol: str
    df: pd.DataFrame
    risk_out: RiskOutput
    results: dict[str, bt.BacktestResult]
    metrics: dict[str, dict]
    verdict: dict
    start: pd.Timestamp


def warmup_len(params: dict, cfg: RiskConfig) -> int:
    nums = [int(v) for v in params.values() if isinstance(v, (int, float))]
    return max(nums + [cfg.vol_lookback_long, 20])


def run_target(d: pd.DataFrame, target: pd.Series, symbol: str, capital: float, cost: CostModel | None,
               band: float, decide_freq: str = "W", risk: RiskConfig | None = None,
               name: str = "策略", cash_rate: float = 0.0) -> bt.BacktestResult:
    """按目标仓位回测。周度决策时“出场要快、进场可以慢”：清仓信号次日即执行，加仓等周度决策日。"""
    t = target.reindex(d.index).ffill().fillna(0.0)
    decide = None
    if decide_freq == "W":
        exit_now = (t <= 0.01) & (t.shift(1).fillna(0) > 0.01)
        decide = rebalance_dates(d.index, "W") | exit_now
    cost = cost or CostModel.for_symbol(symbol)
    return bt.run({symbol: d}, t.to_frame(symbol), capital, {symbol: cost}, band, risk, name, decide, cash_rate)


def analyze_single(df: pd.DataFrame, symbol: str, strategy: str, params: dict | None = None,
                   cfg: RiskConfig | None = None, capital: float = 100_000.0,
                   cost: CostModel | None = None, bench_close: pd.Series | None = None,
                   ppy: int = TRADING_DAYS, begin=None) -> SingleAnalysis:
    """单标的：买入持有 / 裸策略 / 风控策略 / 同等平均仓位基准 四线对比。"""
    spec = STRATEGIES[strategy]
    params = {**spec.defaults(), **(params or {})}
    cfg = cfg or RiskConfig()
    cost = cost or CostModel.for_symbol(symbol)

    if not is_ashare_equity(symbol):
        bench_close = None  # 债券/黄金/跨境资产用自身趋势，不跟随A股大盘
    raw = spec(df, **params)
    ro = apply_risk(df, raw, cfg, bench_close, ppy)

    w = min(warmup_len(params, cfg), len(df) - 30)
    start = df.index[max(w, 0)]
    if begin is not None and pd.Timestamp(begin) > start:
        start = df.index[df.index.searchsorted(pd.Timestamp(begin))]
    d = df.loc[start:]

    def go(target: pd.Series, name: str, band: float, risk: RiskConfig | None = None):
        return run_target(d, target, symbol, capital, cost, band, cfg.decide_freq, risk, name, cfg.cash_rate)

    res = {
        "买入持有": go(pd.Series(1.0, index=df.index), "买入持有", 0.05),
        "裸策略": go(raw, "裸策略", 0.10),
        "风控策略": go(ro.final, "风控策略", cfg.rebalance_band, cfg),
    }
    avg = float(res["风控策略"].exposure.mean())
    res["同仓位基准"] = go(pd.Series(avg, index=df.index), f"恒定{avg:.0%}仓位", 0.05)

    mets = {k: summary(r.equity, r.exposure, r.trades, ppy) for k, r in res.items()}
    verdict = judge(mets)
    return SingleAnalysis(symbol, df, ro, res, mets, verdict, start)


def judge(mets: dict[str, dict]) -> dict:
    """PRD v1.1 验收：风控策略必须在卡玛比率上同时战胜买入持有和同仓位基准，回撤也要更小。"""
    rk, bh, cp = mets["风控策略"], mets["买入持有"], mets["同仓位基准"]

    def g(m, k):
        v = m.get(k)
        return -np.inf if v is None or (isinstance(v, float) and np.isnan(v)) else v

    dd_cut = 1 - abs(rk["最大回撤"]) / abs(bh["最大回撤"]) if bh["最大回撤"] < 0 else np.nan
    checks = {
        "最大回撤小于买入持有": rk["最大回撤"] > bh["最大回撤"],
        "卡玛比率高于买入持有": g(rk, "卡玛比率") > g(bh, "卡玛比率"),
        "卡玛比率高于同仓位基准（真择时能力）": g(rk, "卡玛比率") > g(cp, "卡玛比率"),
    }
    return {"checks": checks, "passed": all(checks.values()), "dd_cut": dd_cut}


def analyze_portfolio(data: dict[str, pd.DataFrame], capital: float = 100_000.0,
                      method: str = "inv_vol", trend: str = "momentum", target_vol: float | None = 0.10,
                      max_weight: float = 0.40, freq: str = "ME", breaker: RiskConfig | None = None,
                      ppy: int = TRADING_DAYS, cash_rate: float = 0.012) -> dict:
    W, decide = portfolio_targets(data, method, trend, 200, 60, target_vol, max_weight, freq, ppy)
    warm = 250 if trend == "momentum" else 200
    start = W.index[min(warm, len(W) - 30)]
    Wd, dd = W.loc[start:], decide.loc[start:]
    costs = {s: CostModel.for_symbol(s) for s in data}

    W_eq, _ = portfolio_targets(data, "equal", "none", target_vol=None, max_weight=1.0, freq=freq, ppy=ppy)
    W_iv, _ = portfolio_targets(data, method, "none", target_vol=None, max_weight=max_weight, freq=freq, ppy=ppy)

    res = {
        "风控组合": bt.run(data, Wd, capital, costs, 0.05, breaker, "风控组合", dd, cash_rate),
        "等权买入持有": bt.run(data, W_eq.loc[start:], capital, costs, 0.02, None, "等权买入持有", dd, cash_rate),
        "配置不择时": bt.run(data, W_iv.loc[start:], capital, costs, 0.02, None, "配置不择时", dd, cash_rate),
    }
    mets = {k: summary(r.equity, r.exposure, r.trades, ppy) for k, r in res.items()}
    return {"targets": W, "results": res, "metrics": mets, "start": start}


def latest_signal(df: pd.DataFrame, symbol: str, strategy: str = "multi_momentum", params: dict | None = None,
                  cfg: RiskConfig | None = None, bench_close: pd.Series | None = None) -> dict:
    """今日（最新收盘）信号：用于“明天开盘该怎么做”。"""
    spec = STRATEGIES[strategy]
    cfg = cfg or RiskConfig()
    if not is_ashare_equity(symbol):
        bench_close = None
    raw = spec(df, **(params or {}))
    ro = apply_risk(df, raw, cfg, bench_close)
    last = df.index[-1]
    c = float(df["close"].iloc[-1])
    ma200 = df["close"].rolling(200).mean().iloc[-1]
    from .indicators import atr as _atr
    a = float(_atr(df, cfg.atr_period).iloc[-1])
    tgt = float(ro.final.iloc[-1])
    prev = float(ro.final.iloc[-2]) if len(ro.final) > 1 else 0.0
    if tgt <= 0.01:
        state = "防守·空仓"
    elif tgt < 0.5:
        state = "谨慎·轻仓"
    else:
        state = "进攻·持有"
    return {
        "代码": symbol,
        "数据日期": last,
        "收盘价": c,
        "原始信号": float(raw.iloc[-1]),
        "目标仓位": tgt,
        "较前日变化": tgt - prev,
        "状态": state,
        "年化波动": float(ro.realized_vol.iloc[-1]) if not np.isnan(ro.realized_vol.iloc[-1]) else np.nan,
        "距200日线": c / ma200 - 1 if ma200 and not np.isnan(ma200) else np.nan,
        "建议止损价": c - cfg.atr_mult * a if not np.isnan(a) else np.nan,
        "大盘过滤": "通过" if ro.regime_cap.iloc[-1] >= 1 else "熊市限仓",
    }


def with_overrides(cfg: RiskConfig, **kw) -> RiskConfig:
    return replace(cfg, **kw)
