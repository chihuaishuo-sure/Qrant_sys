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


__all__ = ["walk_forward", "deflated_sharpe", "block_bootstrap", "grid", "score",
           "prob_loss_exceeds", "equity_from_returns", "drawdown"]
