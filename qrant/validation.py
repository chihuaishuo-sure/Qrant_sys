"""稳健性检验：参数敏感性、Walk-Forward 样本外、Deflated Sharpe、块自助蒙特卡洛。

目的只有一个：识别“回测好看、实盘亏钱”的过拟合策略。
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import TRADING_DAYS
from .indicators import drawdown
from .metrics import summary

EULER = 0.5772156649


def equity_from_returns(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def score(r: pd.Series, metric: str, ppy: int = TRADING_DAYS) -> float:
    eq = equity_from_returns(r)
    if len(eq) < 20:
        return float("nan")
    m = summary(eq, ppy=ppy)
    v = m.get(metric, np.nan)
    return float(v) if v is not None else float("nan")


def grid(params: dict[str, list]) -> list[dict]:
    keys = list(params)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[params[k] for k in keys])]


def walk_forward(returns_by_param: dict[str, pd.Series], train_days: int, test_days: int,
                 metric: str = "卡玛比率", ppy: int = TRADING_DAYS) -> dict:
    """滚动：在过去 train_days 上选指标最好的参数，在接下来 test_days 上检验。

    returns_by_param: {参数标签: 该参数在全历史上的日收益}（各参数的信号只用历史数据，
    因此在任意窗口截取都没有前视）。
    """
    labels = list(returns_by_param)
    R = pd.concat(returns_by_param, axis=1).fillna(0.0)
    idx = R.index
    folds, oos_parts = [], []
    start = train_days
    while start < len(idx):
        tr = R.iloc[start - train_days:start]
        te = R.iloc[start:start + test_days]
        scores = {lab: score(tr[lab], metric, ppy) for lab in labels}
        valid = {k: v for k, v in scores.items() if not math.isnan(v)}
        best = max(valid, key=valid.get) if valid else labels[0]
        oos_parts.append(te[best])
        folds.append({
            "训练期": f"{tr.index[0]:%Y-%m-%d} ~ {tr.index[-1]:%Y-%m-%d}",
            "检验期": f"{te.index[0]:%Y-%m-%d} ~ {te.index[-1]:%Y-%m-%d}",
            "选中参数": best,
            "样本内" + metric: valid.get(best, np.nan),
            "样本外" + metric: score(te[best], metric, ppy) if len(te) > 20 else np.nan,
            "样本外收益": float((1 + te[best]).prod() - 1),
        })
        start += test_days
    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    full_scores = {lab: score(R[lab], metric, ppy) for lab in labels}
    return {"folds": pd.DataFrame(folds), "oos_returns": oos, "full_sample_scores": full_scores}


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float], ppy: int = TRADING_DAYS) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio。

    在尝试了 N 组参数后，“最好的那个夏普”有多大概率是真的 > 0？
    trial_sharpes 为各试验的年化夏普。返回 DSR（概率，> 0.95 才算可信）。
    """
    r = r.dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return {"dsr": np.nan, "sr": np.nan, "sr0": np.nan, "n": len(trial_sharpes)}
    sr = r.mean() / r.std()  # 单期夏普
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    n = max(len(trial_sharpes), 1)
    srs = np.array([s for s in trial_sharpes if not np.isnan(s)]) / np.sqrt(ppy)
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    if n > 1 and var_sr > 0:
        sr0 = math.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e)))
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
    return {"dsr": dsr, "sr": sr * math.sqrt(ppy), "sr0": sr0 * math.sqrt(ppy), "n": n}


def block_bootstrap(r: pd.Series, n_sims: int = 1000, horizon: int = TRADING_DAYS, block: int = 20,
                    seed: int = 42) -> dict:
    """块自助法：随机抽取连续 block 天的历史收益拼成未来路径，保留波动聚集。"""
    x = r.dropna().to_numpy()
    if len(x) < block * 3:
        raise ValueError("历史数据太短，无法做蒙特卡洛")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(x) - block, size=(n_sims, nb))
    paths = np.stack([np.concatenate([x[s:s + block] for s in row])[:horizon] for row in starts])
    eq = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), eq], axis=1), axis=1)[:, 1:]
    mdd = (eq / peak - 1).min(axis=1)
    final = eq[:, -1] - 1
    pct = np.percentile(eq, [5, 25, 50, 75, 95], axis=0)
    return {"max_dd": mdd, "final": final, "fan": pct}


def prob_loss_exceeds(mdd: np.ndarray, thresholds=(0.1, 0.2, 0.3)) -> dict:
    return {f"回撤超过{int(t * 100)}%": float((mdd <= -t).mean()) for t in thresholds}


def _moments(r: pd.Series) -> tuple[float, float, float, int]:
    r = r.dropna()
    sd = r.std()
    if len(r) < 30 or sd == 0:
        return np.nan, np.nan, np.nan, len(r)
    z = (r - r.mean()) / sd
    return float(r.mean() / sd), float((z**3).mean()), float((z**4).mean()), len(r)


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0, ppy: int = TRADING_DAYS) -> dict:
    """Bailey & López de Prado (2012) PSR 与最短业绩记录长度 MinTRL。

    PSR = 真实夏普 > 基准夏普 的概率（考虑偏度、峰度和样本长度）；
    MinTRL = 想让 PSR 达到 95% 至少需要多少期数据 —— 回测比它短，夏普再高也说明不了问题。
    """
    sr, skew, kurt, T = _moments(r)
    if np.isnan(sr):
        return {"psr": np.nan, "min_trl_years": np.nan, "sr": np.nan}
    sr0 = sr_benchmark / math.sqrt(ppy)
    v = max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12)
    psr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(v)))
    min_trl = 1 + v * (norm.ppf(0.95) / (sr - sr0)) ** 2 if sr > sr0 else np.inf
    return {"psr": psr, "min_trl_years": min_trl / ppy, "sr": sr * math.sqrt(ppy), "years": T / ppy}


def pbo_cscv(R: pd.DataFrame, S: int = 16) -> dict:
    """Bailey, Borwein, López de Prado & Zhu (2015) 回测过拟合概率 PBO（组合对称交叉验证 CSCV）。

    R：T×N，每列是一个候选方案（参数组合）的日收益。把时间切成 S 块，枚举所有“一半块做样本内、
    另一半做样本外”的组合：样本内夏普最高的方案，在样本外排名落到中位数以下的比例 = PBO。
    PBO > 0.5 说明“挑最好的”这件事本身不比随机挑更好。
    """
    R = R.dropna(how="all").fillna(0.0)
    T, N = R.shape
    if N < 2 or T < S * 20:
        return {"pbo": np.nan, "logits": np.array([]), "n_splits": 0}
    blocks = np.array_split(np.arange(T), S)
    x = R.to_numpy()
    s1 = np.stack([x[b].sum(axis=0) for b in blocks])  # S×N
    s2 = np.stack([(x[b] ** 2).sum(axis=0) for b in blocks])
    cnt = np.array([len(b) for b in blocks], dtype=float)
    combos = np.array(list(itertools.combinations(range(S), S // 2)))
    mask = np.zeros((len(combos), S))
    mask[np.arange(len(combos))[:, None], combos] = 1.0

    def sharpe(m: np.ndarray) -> np.ndarray:
        n = (m @ cnt)[:, None]
        mu = (m @ s1) / n
        var = (m @ s2) / n - mu**2
        return mu / np.sqrt(np.clip(var, 1e-18, None))

    sr_is, sr_oos = sharpe(mask), sharpe(1.0 - mask)
    best = sr_is.argmax(axis=1)
    rank = (sr_oos < sr_oos[np.arange(len(best)), best][:, None]).sum(axis=1) + 1  # 1 = 最差
    w = rank / (N + 1)
    logits = np.log(w / (1 - w))
    return {"pbo": float((logits <= 0).mean()), "logits": logits, "n_splits": len(combos),
            "oos_sr_of_best": float(np.median(sr_oos[np.arange(len(best)), best]) * math.sqrt(TRADING_DAYS))}


def paired_test(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """配对比较“方案 A 相对方案 B”在同一批标的×策略上的差异。

    比通过率更可靠：用连续指标、配对消除标的差异；Wilcoxon 符号秩检验 + 中位数差的自助法 95% 置信区间。
    注意：A股 ETF 之间高度相关，有效样本数小于组数，p 值偏乐观。
    """
    from scipy.stats import wilcoxon

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 6:
        return {"n": len(d), "median_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "win": np.nan}
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"n": len(d), "median_diff": float(np.median(d)), "ci_low": float(np.percentile(meds, 2.5)),
            "ci_high": float(np.percentile(meds, 97.5)), "p": p, "win": float((d > 0).mean())}


def random_timing_test(asset_returns: pd.Series, exposure: pd.Series, n_sims: int = 500, min_shift: int = 63,
                       cash_rate: float = 0.0, seed: int = 42, ppy: int = TRADING_DAYS) -> dict:
    """随机择时检验：把真实仓位序列整体循环平移随机天数，保留“平均仓位、持仓时长、换手节奏”，
    只打乱它和行情的对应关系。真实择时的卡玛如果排不进随机择时的前 5%，说明择时能力可能只是运气。

    比“同仓位基准”更严格：同仓位基准只控制了平均仓位，这里还控制了仓位变化的模式。
    均为不含成本的毛收益，真实与随机一视同仁。
    """
    r = asset_returns.fillna(0.0)
    e = exposure.reindex(r.index).fillna(0.0).to_numpy()
    rc = cash_rate / ppy

    def calmar_of(expo: np.ndarray) -> float:
        g = expo * r.to_numpy() + (1 - expo) * rc
        eq = np.cumprod(1 + g)
        yrs = len(g) / ppy
        cagr = eq[-1] ** (1 / yrs) - 1 if yrs > 0 and eq[-1] > 0 else np.nan
        mdd = (eq / np.maximum.accumulate(eq) - 1).min()
        return cagr / abs(mdd) if mdd < 0 else np.nan

    actual = calmar_of(e)
    n = len(e)
    if n < min_shift * 3:
        return {"actual": actual, "p": np.nan, "pct_rank": np.nan, "sims": np.array([])}
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n - min_shift, size=n_sims)
    sims = np.array([calmar_of(np.roll(e, s)) for s in shifts])
    sims = sims[~np.isnan(sims)]
    p = (1 + (sims >= actual).sum()) / (1 + len(sims))
    return {"actual": actual, "p": float(p), "pct_rank": float((sims < actual).mean()), "sims": sims,
            "sim_median": float(np.median(sims)) if len(sims) else np.nan}


__all__ = ["walk_forward", "deflated_sharpe", "block_bootstrap", "grid", "score",
           "prob_loss_exceeds", "equity_from_returns", "drawdown", "probabilistic_sharpe", "pbo_cscv",
           "paired_test", "random_timing_test"]
