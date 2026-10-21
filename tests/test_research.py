"""研究工具测试：数据质量、复权、过拟合统计、随机择时、报告对比、试验登记。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrant import compare, data, quality, registry, validation  # noqa: E402


def _px(close, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)


def test_quality_flags_unadjusted_split_and_stale_data():
    c = np.r_[np.full(30, 5.0), np.full(30, 1.0)]  # 1:5 拆分未复权
    df = _px(c)
    q = quality.check(df, "513100", today=df.index[-1])
    assert (q["级别"] == "错误").sum() == 1 and "拆分" in q.iloc[0]["问题"]
    q2 = quality.check(_px(np.full(60, 3.0)), "510300", today=pd.Timestamp("2030-01-01"))
    assert q2["问题"].str.contains("过期").any()
    assert quality.check(_px(np.full(60, 3.0)), "510300", today=pd.Timestamp("2024-03-26")).empty


def test_sina_factor_adjustment_removes_dividend_and_split_gaps():
    c = np.r_[np.full(20, 10.0), np.full(20, 9.5), np.full(20, 1.9)]  # 第20天分红 0.5，第40天 1:5 拆分
    df = _px(c)
    f = pd.DataFrame({"date": [df.index[20], df.index[40]], "s": [1.0, 5.0], "u": [0.5, 0.5]})
    adj, notes = data.adjust_with_sina_factors(df, f)
    r = adj["close"].pct_change().dropna()
    assert r.abs().max() < 1e-9, "除权后不应再有价格断层"
    assert abs(adj["close"].iloc[-1] - 1.9) < 1e-12, "前复权：最新价不变"
    assert len(notes) == 2


def test_pbo_detects_noise_and_rewards_real_edge():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 8)))
    assert 0.3 < validation.pbo_cscv(noise, S=10)["pbo"] < 0.8  # 纯噪声：挑最好的≈随机
    edge = noise.copy()
    edge[0] += 0.003  # 一个方案确实更好
    assert validation.pbo_cscv(edge, S=10)["pbo"] < 0.05


def test_paired_test_and_psr():
    rng = np.random.default_rng(2)
    b = rng.normal(0.3, 0.1, 40)
    t = validation.paired_test(b + 0.05, b)
    assert t["ci_low"] > 0 and t["win"] == 1.0
    t0 = validation.paired_test(b + rng.normal(0, 0.05, 40), b)
    assert t0["ci_low"] < 0 < t0["ci_high"]
    good = pd.Series(rng.normal(0.001, 0.01, 2500))
    assert validation.probabilistic_sharpe(good)["psr"] > 0.95
    assert validation.probabilistic_sharpe(-good)["psr"] < 0.05


def test_random_timing_separates_skill_from_luck():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0003, 0.012, 2500))
    oracle = (r.rolling(3, center=True).mean() > 0).astype(float)  # 偷看当天及前后行情 = 近乎完美择时
    assert validation.random_timing_test(r, oracle, n_sims=200)["p"] < 0.05
    coin = pd.Series((rng.random(2500) > 0.5).astype(float)).rolling(20).max().fillna(0)
    assert validation.random_timing_test(r, coin, n_sims=200)["p"] > 0.05


def _report(rows):
    cols = ["代码", "策略", "持有_年化", "持有_回撤", "风控_年化", "风控_回撤", "风控_卡玛", "同仓位_卡玛", "通过验收"]
    return pd.DataFrame(rows, columns=cols)


def test_compare_separates_data_change_from_regression():
    old = _report([["510300", "动量", 0.04, -0.46, 0.03, -0.20, 0.20, 0.10, True],
                   ["510500", "动量", 0.16, -0.63, 0.12, -0.20, 0.60, 0.30, True]])
    new = _report([["510300", "动量", 0.04, -0.46, 0.02, -0.25, 0.10, 0.10, False],  # 数据没变、结果变差
                   ["510500", "动量", 0.05, -0.63, 0.05, -0.20, 0.25, 0.30, False],  # 持有变了 = 数据修正
                   ["159915", "动量", 0.10, -0.70, 0.08, -0.15, 0.50, 0.20, True]])
    cmp = compare.compare(old, new)
    reg = compare.regressions(cmp)
    assert list(reg["代码"]) == ["510300"]
    s = compare.summary(cmp)
    assert s["数据变化行"] == 1 and s["新增"] == 1 and s["代码退化行"] == 1
    assert "需要排查" in compare.to_markdown(cmp, "a", "b")


def test_registry_counts_trials():
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
        _registry_roundtrip(Path(d) / "t.csv")


def _registry_roundtrip(p: Path):
    registry.log("A股验证", "x", 5, "ok", path=p)
    registry.log("样本外检验", "y", 4, "ok", path=p)
    assert registry.total_configs(path=p) == 9
    assert registry.total_configs("样本外检验", path=p) == 4
