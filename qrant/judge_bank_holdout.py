"""裁判 A 的留出题库：提示词检查清单里没有写到的前视写法，用来测“泛化能力”。

judge_bank.py 的 bug 类型和裁判提示词的 8 条清单一一对应，成绩是样本内的。这里的题分三类：
- 清单外：清单完全没提到的机制（merge_asof 向前对齐、按月 groupby、np.roll、reindex 到未来日期、
  按日期切片的未来窗口、resample 的 label/closed 语义），部分变量名还故意起得像“过去”（prev_week、lagged）
- 清单相邻：清单提到了同类问题，但换了清单没点名的写法（filtfilt、polyfit、idxmax、qcut、循环里的向前切片）
- 陷阱：与上面每道 bug 题配对的干净写法，且不在提示词的“安全写法”列表里

标准答案同样来自 judge_bank.oracle 的执行判卷。
"""
from __future__ import annotations

from .judge_bank import Case

# (题目, 类别, 设计意图是否有前视, 代码, 植入片段)
_CASES: list[tuple[str, str, bool, str, str]] = [
    # ------------------------------------------------------------ 清单外
    ("周线对齐/merge_asof向前", "清单外", True, '''
def signal(df):
    c = df["close"]
    wk = c.resample("W-FRI").last()
    up = (wk > wk.rolling(10, min_periods=10).mean()).astype(float)
    daily = pd.DataFrame({"date": c.index})
    weekly = pd.DataFrame({"date": up.index, "up": up.to_numpy()})
    m = pd.merge_asof(daily, weekly, on="date", direction="forward")
    return pd.Series(m["up"].fillna(0.0).to_numpy(), index=c.index)
''', 'direction="forward"'),
    ("月均线/groupby均值", "清单外", True, '''
def signal(df):
    c = df["close"]
    month_avg = c.groupby(c.index.to_period("M")).transform("mean")
    long_ma = c.rolling(120, min_periods=120).mean()
    return (month_avg > long_ma).astype(float)
''', 'transform("mean")'),
    ("月动量/groupby月末", "清单外", True, '''
def signal(df):
    c = df["close"]
    month_close = c.groupby(c.index.to_period("M")).transform("last")
    mom = month_close / c.rolling(60, min_periods=60).mean() - 1
    return (mom > 0).astype(float)
''', 'transform("last")'),
    ("周动量/np.roll负向", "清单外", True, '''
def signal(df):
    c = df["close"].to_numpy()
    prev_week = np.roll(c, -5)
    prev_week[-5:] = np.nan
    mom = c / prev_week - 1
    return pd.Series((mom > 0).astype(float), index=df.index)
''', "np.roll(c, -5)"),
    ("滞后均线/reindex未来日期", "清单外", True, '''
def signal(df):
    c = df["close"]
    lagged = c.reindex(c.index + pd.offsets.BDay(3))
    lagged.index = c.index
    return (c > lagged.rolling(20, min_periods=20).mean()).astype(float)
''', "c.index + pd.offsets.BDay(3)"),
    ("日历窗口/按日期切片", "清单外", True, '''
def signal(df):
    c = df["close"]
    out = []
    for d in c.index:
        window = c.loc[d - pd.Timedelta(days=30): d + pd.Timedelta(days=5)]
        out.append(float(c.loc[d] >= window.mean()))
    return pd.Series(out, index=c.index)
''', "d + pd.Timedelta(days=5)"),
    ("周均价/resample左标签", "清单外", True, '''
def signal(df):
    c = df["close"]
    wk = c.resample("W-MON", label="left", closed="left").mean()
    weekly_avg = wk.reindex(c.index, method="ffill")
    return (c > weekly_avg).astype(float)
''', 'label="left", closed="left"'),
    # ------------------------------------------------------------ 清单相邻（换了写法）
    ("平滑趋势/filtfilt", "清单相邻", True, '''
def signal(df):
    from scipy.signal import butter, filtfilt
    c = df["close"].to_numpy()
    b, a = butter(2, 0.05)
    smooth = filtfilt(b, a, c)
    slope = np.diff(smooth, prepend=smooth[0])
    return pd.Series((slope > 0).astype(float), index=df.index)
''', "filtfilt(b, a, c)"),
    ("去趋势/polyfit", "清单相邻", True, '''
def signal(df):
    c = np.log(df["close"].to_numpy())
    x = np.arange(len(c))
    k, b = np.polyfit(x, c, 1)
    resid = c - (k * x + b)
    return pd.Series((resid < 0).astype(float), index=df.index)
''', "np.polyfit(x, c, 1)"),
    ("见顶前持有/idxmax", "清单相邻", True, '''
def signal(df):
    c = df["close"]
    top = c.idxmax()
    trend = (c > c.rolling(60, min_periods=60).mean()).astype(float)
    return trend.where(c.index < top, 0.0)
''', "c.idxmax()"),
    ("近期均值/循环向前切片", "清单相邻", True, '''
def signal(df):
    c = df["close"].to_numpy()
    pos = np.zeros(len(c))
    for i in range(20, len(c)):
        recent = c[i:i + 20]
        pos[i] = float(c[i] > recent.mean())
    return pd.Series(pos, index=df.index)
''', "c[i:i + 20]"),
    ("动量分组/qcut", "清单相邻", True, '''
def signal(df):
    c = df["close"]
    r20 = c.pct_change(20)
    bucket = pd.qcut(r20, 5, labels=False)
    return (bucket >= 3).astype(float)
''', "pd.qcut(r20, 5, labels=False)"),
    # ------------------------------------------------------------ 陷阱（干净）
    ("周线对齐/merge_asof向后", "陷阱", False, '''
def signal(df):
    c = df["close"]
    wk = c.resample("W-FRI").last()
    up = (wk > wk.rolling(10, min_periods=10).mean()).astype(float)
    daily = pd.DataFrame({"date": c.index})
    weekly = pd.DataFrame({"date": up.index, "up": up.to_numpy()})
    m = pd.merge_asof(daily, weekly, on="date", direction="backward")
    return pd.Series(m["up"].fillna(0.0).to_numpy(), index=c.index)
''', ""),
    ("月均线/groupby累计均值", "陷阱", False, '''
def signal(df):
    c = df["close"]
    month_avg = c.groupby(c.index.to_period("M")).transform(lambda x: x.expanding().mean())
    long_ma = c.rolling(120, min_periods=120).mean()
    return (month_avg > long_ma).astype(float)
''', ""),
    ("周动量/np.roll正向", "陷阱", False, '''
def signal(df):
    c = df["close"].to_numpy()
    prev_week = np.roll(c, 5)
    prev_week[:5] = np.nan
    mom = c / prev_week - 1
    return pd.Series((mom > 0).astype(float), index=df.index)
''', ""),
    ("滞后均线/reindex过去日期", "陷阱", False, '''
def signal(df):
    c = df["close"]
    lagged = c.reindex(c.index - pd.offsets.BDay(3))
    lagged.index = c.index
    return (c > lagged.rolling(20, min_periods=20).mean()).astype(float)
''', ""),
    ("日历窗口/只切过去", "陷阱", False, '''
def signal(df):
    c = df["close"]
    out = []
    for d in c.index:
        window = c.loc[d - pd.Timedelta(days=35): d]
        out.append(float(c.loc[d] >= window.mean()))
    return pd.Series(out, index=c.index)
''', ""),
    ("周均价/resample默认标签", "陷阱", False, '''
def signal(df):
    c = df["close"]
    wk = c.resample("W-FRI").mean()
    weekly_avg = wk.reindex(c.index, method="ffill")
    return (c > weekly_avg).astype(float)
''', ""),
    ("平滑趋势/lfilter因果滤波", "陷阱", False, '''
def signal(df):
    from scipy.signal import butter, lfilter
    c = df["close"].to_numpy()
    b, a = butter(2, 0.05)
    smooth = lfilter(b, a, c)
    slope = np.diff(smooth, prepend=smooth[0])
    return pd.Series((slope > 0).astype(float), index=df.index)
''', ""),
    ("近期均值/循环向后切片", "陷阱", False, '''
def signal(df):
    c = df["close"].to_numpy()
    pos = np.zeros(len(c))
    for i in range(20, len(c)):
        recent = c[i - 20:i]
        pos[i] = float(c[i] > recent.mean())
    return pd.Series(pos, index=df.index)
''', ""),
    ("动量分组/滚动分位", "陷阱", False, '''
def signal(df):
    c = df["close"]
    r20 = c.pct_change(20)
    bucket = r20.rolling(250, min_periods=250).rank(pct=True)
    return (bucket >= 0.6).astype(float)
''', ""),
]


def build() -> list[Case]:
    return [Case(name, name.split("/")[0], kind, intent, code.strip() + "\n", planted)
            for name, kind, intent, code, planted in _CASES]
