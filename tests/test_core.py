"""核心正确性测试：无前视、A股规则、成本、风控、稳健性模块。

运行：python -m pytest tests -q    （或直接 python tests/test_core.py）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrant import backtest as bt  # noqa: E402
from qrant import data, metrics, pipeline, validation  # noqa: E402
from qrant.config import CostModel, RiskConfig  # noqa: E402
from qrant.portfolio import portfolio_targets  # noqa: E402
from qrant.risk import apply_risk, position_size  # noqa: E402
from qrant.strategies import STRATEGIES  # noqa: E402


def _flat_df(n=60, price=10.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1e6}, index=idx)


def test_no_lookahead_in_signals_and_risk():
    df = data.demo_data("510300")
    cut = df.index[1500]
    df2 = df.copy()
    df2.loc[df2.index > cut, ["open", "high", "low", "close"]] *= 3.0  # 篡改未来
    for key, spec in STRATEGIES.items():
        a, b = spec(df), spec(df2)
        assert a.loc[:cut].equals(b.loc[:cut]), f"{key} 使用了未来数据"
    ra = apply_risk(df, STRATEGIES["multi_momentum"](df), RiskConfig(atr_stop_on=True)).final
    rb = apply_risk(df2, STRATEGIES["multi_momentum"](df2), RiskConfig(atr_stop_on=True)).final
    assert np.allclose(ra.loc[:cut], rb.loc[:cut])


def test_executes_next_open_with_slippage():
    df = _flat_df()
    df.iloc[10:, df.columns.get_loc("open")] = 11.0
    df.iloc[10:, df.columns.get_loc("close")] = 11.0
    tgt = pd.Series(0.0, index=df.index)
    tgt.iloc[5:] = 1.0  # 第 5 天收盘决定买入
    cm = CostModel(etf=True)
    r = bt.run_single(df, tgt, "510300", 100_000, cm, 0.1)
    first = r.trades.iloc[0]
    assert first["日期"] == df.index[6], "应在信号次日成交"
    assert abs(first["成交价"] - 10.0 * (1 + cm.slippage)) < 1e-9
    assert first["数量"] % 100 == 0


def test_costs_etf_vs_stock():
    etf, stk = CostModel(etf=True), CostModel(etf=False)
    assert etf.fee(1000, "buy") == 5.0 + 1000 * etf.transfer_fee  # 最低 5 元
    v = 1_000_000
    assert abs(stk.fee(v, "sell") - etf.fee(v, "sell") - v * 0.0005) < 1e-6  # 股票卖出多收印花税
    assert stk.fee(v, "buy") == etf.fee(v, "buy")  # 买入不收印花税


def test_limit_up_blocks_buy():
    df = _flat_df()
    df.iloc[6, df.columns.get_loc("open")] = 11.0  # 开盘即涨停
    tgt = pd.Series(0.0, index=df.index)
    tgt.iloc[5:] = 1.0
    r = bt.run_single(df, tgt, "510300", 100_000, CostModel(etf=True), 0.1)
    assert r.trades.iloc[0]["方向"] == "买入失败"
    assert r.trades.iloc[1]["方向"] == "买入" and r.trades.iloc[1]["日期"] == df.index[7]


def test_vol_target_never_levers():
    df = data.demo_data("511010")  # 低波动国债
    out = apply_risk(df, pd.Series(1.0, index=df.index), RiskConfig(target_vol=0.5, regime_on=False))
    assert out.final.max() <= 1.0 + 1e-12


def test_breaker_goes_flat():
    df = _flat_df(120)
    c = np.r_[np.full(20, 10.0), np.linspace(10, 7, 20), np.full(80, 7.0)]
    df["open"] = df["high"] = df["low"] = df["close"] = c
    tgt = pd.Series(1.0, index=df.index)
    cfg = RiskConfig(breaker_on=True, breaker_dd=0.15, breaker_cooldown=10, breaker_resume_ma=20)
    r = bt.run_single(df, tgt, "510300", 100_000, CostModel(etf=True, slippage=0), 0.05, cfg)
    assert r.breaker.max() == 1.0
    assert r.equity.min() / 100_000 - 1 > -0.25, "熔断应在 15% 附近止血"
    assert (r.trades["原因"] == "熔断清仓").any()


def test_position_size():
    p = position_size(100_000, 0.01, entry=10.0, stop=9.5, max_position_pct=0.5)
    assert p["shares"] == 2000 and abs(p["loss_if_stopped"] - 1000) < 1e-9
    p2 = position_size(100_000, 0.01, entry=10.0, stop=9.9, max_position_pct=0.3)
    assert p2["shares"] == 3000 and p2["limited_by"] == "单只仓位上限"


def test_pipeline_and_verdict_structure():
    df = data.demo_data("510300")
    a = pipeline.analyze_single(df, "510300", "multi_momentum", bench_close=data.demo_data("sh000300")["close"])
    assert set(a.results) == {"买入持有", "裸策略", "风控策略", "同仓位基准"}
    assert a.metrics["风控策略"]["最大回撤"] > a.metrics["买入持有"]["最大回撤"]
    assert set(a.verdict["checks"]) and isinstance(a.verdict["passed"], bool)
    tbl = metrics.table(a.metrics)
    assert "卡玛比率" in tbl.index


def test_portfolio_weights():
    d = {s: data.demo_data(s) for s in ["510300", "518880", "511010", "513100"]}
    W, decide = portfolio_targets(d, max_weight=0.4)
    assert (W.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (W.max(axis=1) <= 0.4 + 1e-9).all()
    res = pipeline.analyze_portfolio(d)
    assert res["metrics"]["风控组合"]["最大回撤"] > res["metrics"]["等权买入持有"]["最大回撤"]


def test_validation_tools():
    df = data.demo_data("510300")
    rets = {}
    for f, s in [(1, 100), (1, 200), (20, 120)]:
        tgt = STRATEGIES["ma_trend"](df, fast=f, slow=s)
        rets[f"{f}/{s}"] = bt.run_single(df, tgt, "510300").returns
    wf = validation.walk_forward(rets, 756, 252)
    assert len(wf["folds"]) > 5 and len(wf["oos_returns"]) > 1000
    sharpes = [metrics.summary(validation.equity_from_returns(r))["夏普比率"] for r in rets.values()]
    d = validation.deflated_sharpe(rets["1/200"], sharpes)
    assert 0 <= d["dsr"] <= 1
    mc = validation.block_bootstrap(rets["1/200"], n_sims=200)
    assert mc["max_dd"].shape == (200,) and (mc["max_dd"] <= 0).all()


def test_csv_loader_chinese_headers(tmp_path=None):
    import io
    csv = "日期,开盘,最高,最低,收盘,成交量\n" + "\n".join(
        f"2024-01-{i + 1:02d},10,10.5,9.5,10.{i % 10},1000" for i in range(31))
    df = data.load_csv(io.BytesIO(csv.encode("gbk")))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and len(df) == 31


def test_repair_splits():
    df = data.demo_data("513100")
    df.iloc[1500:, :4] = df.iloc[1500:, :4] / 5  # 模拟 1 拆 5
    fixed, notes = data.repair_splits(df)
    assert len(notes) == 1 and fixed["close"].pct_change().abs().max() < 0.2
    _, none = data.repair_splits(data.demo_data("510300"))
    assert none == []


def test_cash_interest():
    df = _flat_df(253)
    r = bt.run({"510300": df}, pd.DataFrame({"510300": 0.0}, index=df.index), 100_000, cash_rate=0.0252)
    assert abs(r.equity.iloc[-1] / 100_000 - (1 + 0.0252 / 252) ** 252) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"全部 {len(fns)} 项测试通过")
