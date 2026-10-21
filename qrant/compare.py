"""验证报告回归对比：新旧两份 A股验证 CSV 逐行对齐，区分“数据变了”和“策略变了”。

一次通过率下降，可能是代码退化，也可能是数据被修正（如份额折算）。两者结论完全不同，
所以先看买入持有的指标有没有变：持有变了 = 数据变化；持有没变而风控变了 = 代码/参数变化。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["代码", "策略"]
DATA_TOL = 0.003  # 持有年化/回撤变化超过 0.3 个百分点，或相对变化超过 5%，视为数据变化
DATA_REL = 0.05
CALMAR_DROP = 0.05  # 风控卡玛下降超过该值视为退化
DD_WORSE = 0.02  # 风控最大回撤加深超过 2 个百分点视为退化


def load_report(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"代码": str}, encoding="utf-8-sig")
    df["代码"] = df["代码"].str.zfill(6)
    return df


def latest_reports(folder: str | Path, pattern: str = "A股验证_*.csv", n: int = 2) -> list[Path]:
    """按文件名中的时间戳排序，返回最新的 n 份（旧 → 新）。演示数据报告不参与。"""
    files = sorted(p for p in Path(folder).glob(pattern) if "demo" not in p.stem)
    return files[-n:]


def compare(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    m = old.merge(new, on=KEYS, how="outer", suffixes=("_旧", "_新"), indicator=True)
    out = m[KEYS].copy()
    for k in ["持有_年化", "持有_回撤", "风控_年化", "风控_回撤", "风控_卡玛", "同仓位_卡玛"]:
        out[f"{k}_旧"], out[f"{k}_新"] = m[f"{k}_旧"], m[f"{k}_新"]
    out["通过_旧"] = m["通过验收_旧"].map(_bool)
    out["通过_新"] = m["通过验收_新"].map(_bool)
    out["卡玛变化"] = out["风控_卡玛_新"] - out["风控_卡玛_旧"]
    out["回撤变化"] = out["风控_回撤_新"] - out["风控_回撤_旧"]  # 负数 = 回撤加深
    out["数据变化"] = _changed(out, "持有_年化") | _changed(out, "持有_回撤")
    out["标记"] = [_flag(r, ind) for r, ind in zip(out.itertuples(index=False), m["_merge"])]
    return out


def _changed(df: pd.DataFrame, k: str) -> pd.Series:
    """买入持有指标变了 = 行情数据变了（修正、复权、新增交易日）。低收益资产用相对变化判断。"""
    d = (df[f"{k}_新"] - df[f"{k}_旧"]).abs()
    return (d > DATA_TOL) | (d > DATA_REL * df[f"{k}_旧"].abs())


def _bool(v) -> float:
    if isinstance(v, str):
        return {"True": True, "False": False, "✅": True, "❌": False}.get(v, np.nan)
    return v if v is None or isinstance(v, bool) or not np.isnan(v) else np.nan


def _flag(r, ind: str) -> str:
    if ind == "left_only":
        return "移除"
    if ind == "right_only":
        return "新增"
    tags = []
    if r.数据变化:
        tags.append("数据变化")
    if r.通过_旧 and not r.通过_新:
        tags.append("通过→未通过")
    elif r.通过_新 and not r.通过_旧:
        tags.append("未通过→通过")
    if r.卡玛变化 < -CALMAR_DROP:
        tags.append("卡玛下降")
    if r.回撤变化 < -DD_WORSE:
        tags.append("回撤加深")
    return "、".join(tags)


def regressions(cmp: pd.DataFrame) -> pd.DataFrame:
    """数据没变、但结果变差的行 —— 真正需要排查的代码/参数退化。"""
    bad = cmp["标记"].str.contains("通过→未通过|卡玛下降|回撤加深", regex=True)
    return cmp[bad & ~cmp["数据变化"]]


def summary(cmp: pd.DataFrame) -> dict:
    both = cmp[cmp["通过_旧"].notna() & cmp["通过_新"].notna()]
    return {
        "旧通过率": float(cmp["通过_旧"].dropna().mean()),
        "新通过率": float(cmp["通过_新"].dropna().mean()),
        "共同行数": len(both),
        "共同行旧通过率": float(both["通过_旧"].mean()) if len(both) else np.nan,
        "共同行新通过率": float(both["通过_新"].mean()) if len(both) else np.nan,
        "数据变化行": int(cmp["数据变化"].sum()),
        "代码退化行": len(regressions(cmp)),
        "新增": int((cmp["标记"] == "新增").sum()),
        "移除": int((cmp["标记"] == "移除").sum()),
    }


def to_markdown(cmp: pd.DataFrame, old_name: str, new_name: str) -> str:
    s = summary(cmp)
    reg = regressions(cmp)
    lines = [f"# 验证报告对比：{old_name} → {new_name}", "",
             f"- 通过率：{s['旧通过率']:.0%} → {s['新通过率']:.0%}"
             f"（两边都有的 {s['共同行数']} 行：{s['共同行旧通过率']:.0%} → {s['共同行新通过率']:.0%}）",
             f"- 数据变化（买入持有年化或回撤变动 > {DATA_TOL:.1%} 或相对 > {DATA_REL:.0%}）：{s['数据变化行']} 行",
             f"- **代码/参数退化（数据没变、结果变差）：{s['代码退化行']} 行**",
             f"- 新增 {s['新增']} 行，移除 {s['移除']} 行", ""]
    if len(reg):
        lines += ["## ⚠ 需要排查的退化", "", _fmt(reg).to_markdown(index=False), ""]
    changed = cmp[cmp["标记"] != ""]
    if len(changed):
        lines += ["## 全部有变化的行", "", _fmt(changed).to_markdown(index=False), ""]
    lines += ["判定：数据变化 = 买入持有本身变了（行情修正/数据源变动），这类行的结果变化不归咎于代码。"]
    return "\n".join(lines)


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    for k in [c for c in show.columns if c.startswith(("持有_", "风控_年化", "风控_回撤"))] + ["回撤变化"]:
        show[k] = show[k].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    for k in [c for c in show.columns if "卡玛" in c]:
        show[k] = show[k].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}" if k == "卡玛变化" else f"{v:.2f}")
    for k in ["通过_旧", "通过_新"]:
        show[k] = show[k].map({True: "✅", False: "❌"}).fillna("—")
    return show.drop(columns=["数据变化"])
