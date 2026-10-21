"""回测引擎（日线，逐日撮合）。

关键约定（防止回测“虚胖”）：
1. t 日收盘产生目标仓位，t+1 日【开盘价】成交 —— 无前视偏差；
2. 100 股整手、佣金最低 5 元、过户费、卖出印花税（ETF 免）、双边滑点；
3. 开盘即涨停买不进、开盘即跌停卖不出；
4. 每天只在开盘交易一次，天然满足 T+1；
5. 回撤熔断：账户自高点回撤超过阈值 → 清仓冷却 N 日，之后价格站上均线才恢复。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostModel, RiskConfig
from .indicators import sma


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    exposure: pd.Series
    trades: pd.DataFrame
    breaker: pd.Series
    total_cost: float
    capital: float


def run(
    data: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    capital: float = 100_000.0,
    costs: dict[str, CostModel] | None = None,
    band: float = 0.10,
    risk: RiskConfig | None = None,
    name: str = "策略",
    decide: pd.Series | None = None,
    cash_rate: float = 0.0,
) -> BacktestResult:
    """decide: 可选的布尔序列，只有 decide[t]=True 的收盘决策会在 t+1 执行（用于月度再平衡）。
    cash_rate: 闲置现金年化收益（货币基金/国债逆回购），按交易日计息。"""
    syms = list(targets.columns)
    idx = targets.index
    for s in syms:
        idx = idx.union(data[s].index)
    idx = idx[(idx >= targets.index[0]) & (idx <= targets.index[-1])]
    n, m = len(idx), len(syms)

    has_bar = np.zeros((n, m), dtype=bool)
    O = np.full((n, m), np.nan)
    C = np.full((n, m), np.nan)
    for j, s in enumerate(syms):
        d = data[s].reindex(idx)
        has_bar[:, j] = d["close"].notna().to_numpy()
        O[:, j] = d["open"].to_numpy()
        C[:, j] = d["close"].to_numpy()
    Cf = pd.DataFrame(C, index=idx).ffill().to_numpy()  # 估值用（停牌沿用前收）
    T = targets.reindex(idx).ffill().fillna(0.0).to_numpy()
    D = (decide.reindex(idx).fillna(False).astype(bool).to_numpy() if decide is not None
         else np.ones(n, dtype=bool))
    costs = costs or {s: CostModel.for_symbol(s) for s in syms}

    # 熔断恢复参考：各资产归一化收盘的均值及其均线
    ref = pd.DataFrame(Cf, index=idx)
    ref = (ref / ref.bfill().iloc[0]).mean(axis=1)
    ref_ma = sma(ref, risk.breaker_resume_ma).to_numpy() if risk and risk.breaker_on else None
    ref = ref.to_numpy()

    cash = float(capital)
    shares = np.zeros(m)
    equity = np.zeros(n)
    W = np.zeros((n, m))
    brk = np.zeros(n)
    trades: list[tuple] = []
    total_cost = 0.0

    peak = capital
    active = False
    resumed = False
    until = -1

    daily_cash = cash_rate / 252.0
    for i in range(n):
        if i > 0 and cash > 0 and daily_cash > 0:
            cash *= 1 + daily_cash
        if i > 0 and (D[i - 1] or active or resumed):
            resumed = False
            desired = np.zeros(m) if active else T[i - 1]
            px_open = np.where(has_bar[i], O[i], Cf[i - 1])
            px_open = np.where(np.isnan(px_open), 0.0, px_open)
            eq_open = cash + float(np.nansum(shares * px_open))
            order = []
            for j in range(m):
                if not has_bar[i, j] or px_open[j] <= 0:
                    continue
                cur_w = shares[j] * px_open[j] / eq_open if eq_open > 0 else 0.0
                tw = float(desired[j])
                need = (tw <= 1e-9 < shares[j]) or (shares[j] == 0 and tw > 1e-9) or abs(tw - cur_w) > band
                if not need:
                    continue
                lot = costs[syms[j]].lot_size
                tgt_sh = np.floor(tw * eq_open / (px_open[j] * lot)) * lot
                if tgt_sh != shares[j]:
                    order.append((j, tgt_sh))
            # 先卖后买
            order.sort(key=lambda o: o[1] - shares[o[0]])
            for j, tgt_sh in order:
                cm = costs[syms[j]]
                prev_c = Cf[i - 1, j]
                o = O[i, j]
                reason = "熔断清仓" if active else "调仓"
                if tgt_sh < shares[j]:
                    if not np.isnan(prev_c) and o <= prev_c * (1 - cm.limit_pct) * 1.002:
                        trades.append((idx[i], syms[j], "卖出失败", o, 0, 0.0, 0.0, "开盘跌停"))
                        continue
                    qty = shares[j] - tgt_sh
                    px = o * (1 - cm.slippage)
                    val = qty * px
                    fee = cm.fee(val, "sell")
                    cash += val - fee
                    shares[j] = tgt_sh
                    total_cost += fee + qty * o * cm.slippage
                    trades.append((idx[i], syms[j], "卖出", px, qty, val, fee, reason))
                else:
                    if not np.isnan(prev_c) and o >= prev_c * (1 + cm.limit_pct) * 0.998:
                        trades.append((idx[i], syms[j], "买入失败", o, 0, 0.0, 0.0, "开盘涨停"))
                        continue
                    px = o * (1 + cm.slippage)
                    qty = tgt_sh - shares[j]
                    afford = np.floor(max(cash - cm.min_commission, 0) / (px * (1 + cm.commission_rate + cm.transfer_fee) * cm.lot_size)) * cm.lot_size
                    qty = min(qty, afford)
                    if qty <= 0:
                        continue
                    val = qty * px
                    fee = cm.fee(val, "buy")
                    cash -= val + fee
                    shares[j] += qty
                    total_cost += fee + qty * o * cm.slippage
                    trades.append((idx[i], syms[j], "买入", px, qty, val, fee, reason))

        px_close = np.where(np.isnan(Cf[i]), 0.0, Cf[i])
        pos_val = shares * px_close
        equity[i] = cash + pos_val.sum()
        W[i] = pos_val / equity[i] if equity[i] > 0 else 0.0

        if risk is not None and risk.breaker_on:
            if not active:
                peak = max(peak, equity[i])
                if equity[i] / peak - 1 <= -risk.breaker_dd:
                    active, until = True, i + risk.breaker_cooldown
            elif i >= until and ref_ma is not None and not np.isnan(ref_ma[i]) and ref[i] > ref_ma[i]:
                active, peak, resumed = False, equity[i], True
            brk[i] = 1.0 if active else 0.0

    eq = pd.Series(equity, index=idx, name=name)
    tr = pd.DataFrame(trades, columns=["日期", "代码", "方向", "成交价", "数量", "金额", "费用", "原因"])
    w = pd.DataFrame(W, index=idx, columns=syms)
    return BacktestResult(
        name=name, equity=eq, returns=eq.pct_change().fillna(0.0), weights=w,
        exposure=w.sum(axis=1), trades=tr, breaker=pd.Series(brk, index=idx),
        total_cost=total_cost, capital=capital,
    )


def run_single(df: pd.DataFrame, target: pd.Series, symbol: str = "asset", capital: float = 100_000.0,
               cost: CostModel | None = None, band: float = 0.10, risk: RiskConfig | None = None,
               name: str = "策略") -> BacktestResult:
    return run({symbol: df}, target.to_frame(symbol), capital,
               {symbol: cost or CostModel.for_symbol(symbol)}, band, risk, name)
