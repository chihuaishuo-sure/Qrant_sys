"""技术指标。全部只使用 t 日及以前的数据（无未来函数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder ATR。"""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def realized_vol(close: pd.Series, n: int = 20, ppy: int = TRADING_DAYS) -> pd.Series:
    """年化已实现波动率。"""
    r = close.pct_change()
    return r.rolling(n, min_periods=max(5, n // 2)).std() * np.sqrt(ppy)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(up.notna())


def donchian(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """前 n 日（不含当日）最高价/最低价通道。"""
    hi = df["high"].rolling(n, min_periods=n).max().shift(1)
    lo = df["low"].rolling(n, min_periods=n).min().shift(1)
    return hi, lo


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0
