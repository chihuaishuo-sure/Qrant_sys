"""行情数据质量检查：在回测和出信号之前拦住坏数据。

教训：一次未处理的份额拆分让纳指ETF出现 −85% 的假暴跌，策略“躲过”它，验证通过率因此虚高。
检查项都只看数据本身，不依赖策略：
- 涨跌幅超过涨跌停（份额折算未修正、数据源错误）
- 非正价格、开高低收矛盾
- 长时间缺口（停牌或数据缺失）
- 连续零成交量
- 数据过期（最新日期离今天太远，信号不可用）
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .config import price_limit

LIMIT_SLACK = 0.02  # 涨跌停之外再留 2 个百分点（ETF 净值折溢价、四舍五入）
GAP_DAYS = 15  # 自然日缺口超过 15 天（春节最长约 9 天）
ZERO_VOL_RUN = 5
STALE_BDAYS = 3


def check(df: pd.DataFrame, symbol: str, kind: str = "auto", today: datetime | None = None) -> pd.DataFrame:
    """返回问题清单：级别(错误/警告)、日期、问题。空表 = 通过。"""
    issues: list[dict] = []

    def add(level: str, when, msg: str):
        issues.append({"代码": symbol, "级别": level, "日期": when, "问题": msg})

    if df is None or len(df) == 0:
        add("错误", None, "没有数据")
        return pd.DataFrame(issues)

    px = df[["open", "high", "low", "close"]]
    bad = (px <= 0).any(axis=1)
    for d in df.index[bad][:5]:
        add("错误", d, "价格 ≤ 0")
    incoherent = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | \
                 (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)
    for d in df.index[incoherent][:5]:
        add("错误", d, "最高价/最低价没有包住开盘收盘")

    if kind != "index":
        lim = price_limit(symbol) + LIMIT_SLACK
        r = df["close"].pct_change()
        for d, v in r[r.abs() > lim].items():
            if d == df.index[1]:
                continue  # 上市首日附近不设涨跌停
            add("错误", d, f"单日涨跌 {v:+.1%} 超过涨跌停 ±{lim - LIMIT_SLACK:.0%}，疑似拆分/折算未复权")

    gaps = df.index.to_series().diff().dt.days
    for d, g in gaps[gaps > GAP_DAYS].items():
        add("警告", d, f"距上一个交易日 {int(g)} 天，可能停牌或数据缺失")

    if "volume" in df and (df["volume"] > 0).any():
        z = (df["volume"] <= 0).astype(int)
        run = z.groupby((z != z.shift()).cumsum()).cumsum()
        ends = run[(run >= ZERO_VOL_RUN) & (run.shift(-1).fillna(0) < run)]
        for d, n in ends.items():
            add("警告", d, f"连续 {int(n)} 天成交量为 0")

    today = pd.Timestamp(today or datetime.now()).normalize()
    lag = len(pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), today))
    if lag > STALE_BDAYS:
        add("警告", df.index[-1], f"数据已过期 {lag} 个工作日，今日信号不可用")
    return pd.DataFrame(issues, columns=["代码", "级别", "日期", "问题"])


def check_many(datas: dict[str, pd.DataFrame], today: datetime | None = None) -> pd.DataFrame:
    frames = [check(df, c, today=today) for c, df in datas.items()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["代码", "级别", "日期", "问题"])
