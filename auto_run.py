"""Qrant 自动化任务：由 Windows 任务计划（安装定时任务.bat）每个工作日收盘后调用。

用法：
    python auto_run.py auto        # 定时任务入口：每天跑 daily；周五加跑 weekly；季度首周的周五加跑 quarterly
    python auto_run.py daily       # 刷新行情 → 数据质量检查 → 今日信号 + 组合建议持仓 → 与昨日对比
    python auto_run.py weekly      # A股验证（自动与上一份报告对比，发现代码/参数退化）
    python auto_run.py quarterly   # 样本外检验（监控策略是否衰减）
    加 --demo 用演示数据（不联网，测试流程用）

只出信号、不下单。结果写入 reports/daily/，可选推送到微信/钉钉/飞书（见 qrant/notify.py）。
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from qrant import data, notify, pipeline, quality
from qrant.config import BENCHMARK_INDEX, DEFAULT_WATCHLIST, RiskConfig

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "reports" / "daily"
LOG = ROOT / "reports" / "auto.log"
STRATEGY = "multi_momentum"
CAPITAL = 200_000
ACT_THRESHOLD = 0.10  # 目标仓位变化超过 10 个百分点才算“需要操作”（与默认调仓阈值一致）


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _get(code: str, kind: str, demo: bool) -> pd.DataFrame:
    if demo:
        return data.demo_data(code, "20120101")
    return data.fetch(code, "20120101", kind=kind, force=True)[0]


def daily(demo: bool = False) -> dict:
    today = datetime.now()
    try:
        bench = _get(BENCHMARK_INDEX, "index", demo)["close"]
    except Exception as e:  # noqa: BLE001
        log(f"基准获取失败，大盘过滤改用各标的自身均线：{e}")
        bench = None
    datas, failed = {}, []
    for c in DEFAULT_WATCHLIST:
        try:
            datas[c] = _get(c, "auto", demo)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{c}：{str(e)[:60]}")
    qc = quality.check_many(datas, today=None if not demo else max(d.index[-1] for d in datas.values()))
    bad = set(qc.loc[qc["级别"] == "错误", "代码"])
    stale = set(qc.loc[qc["问题"].str.contains("过期"), "代码"])
    usable = {c: d for c, d in datas.items() if c not in bad}
    last = max(d.index[-1] for d in usable.values()) if usable else None

    sig = pd.DataFrame([{**pipeline.latest_signal(d, c, STRATEGY, cfg=RiskConfig(), bench_close=bench),
                         "名称": DEFAULT_WATCHLIST[c]} for c, d in usable.items()])
    pf_rows = []
    if len(usable) >= 2:
        pf = pipeline.analyze_portfolio({c: d.loc["20120101":] for c, d in usable.items()}, CAPITAL)
        w = pf["targets"].iloc[-1]
        for c, wt in w.items():
            px = float(usable[c]["close"].iloc[-1])
            pf_rows.append({"代码": c, "名称": DEFAULT_WATCHLIST[c], "目标权重": wt, "金额": CAPITAL * wt,
                            "约合股数": int(CAPITAL * wt / px // 100 * 100)})
        pf_rows.append({"代码": "现金", "名称": "现金/货币基金", "目标权重": 1 - w.sum(), "金额": CAPITAL * (1 - w.sum()),
                        "约合股数": 0})
    pf_df = pd.DataFrame(pf_rows)

    # 与上一份每日信号对比
    DAILY.mkdir(parents=True, exist_ok=True)
    tag = f"{last:%Y%m%d}" if last is not None else f"{today:%Y%m%d}"
    tag += "_demo" if demo else ""
    prev_files = sorted(p for p in DAILY.glob("信号_*.csv") if p.stem != f"信号_{tag}" and ("demo" in p.stem) == demo)
    actions = []
    if len(sig):
        cur = sig.set_index("代码")["目标仓位"]
        if prev_files:
            prev = pd.read_csv(prev_files[-1], dtype={"代码": str}).set_index("代码")["目标仓位"]
            for c, v in cur.items():
                p = float(prev.get(c, 0.0))
                if abs(v - p) > ACT_THRESHOLD:
                    actions.append(f"{c} {DEFAULT_WATCHLIST.get(c, c)}：{p:.0%} → {v:.0%}（{'加仓' if v > p else '减仓'}）")
        sig.to_csv(DAILY / f"信号_{tag}.csv", index=False, encoding="utf-8-sig")
        if len(pf_df):
            pf_df.to_csv(DAILY / f"组合_{tag}.csv", index=False, encoding="utf-8-sig")

    lines = [f"# Qrant 每日信号 · 数据截至 {last:%Y-%m-%d}" if last is not None else "# Qrant 每日信号", "",
             f"生成于 {today:%Y-%m-%d %H:%M}；策略 {STRATEGY}，默认风控；**明日开盘执行**，本系统不下单。", ""]
    if not qc.empty or failed:
        lines += ["## ⚠ 数据问题", ""] + [f"- {r.代码} {r.级别}：{r.问题}" for r in qc.itertuples()] + \
                 [f"- 获取失败 {f}" for f in failed] + [""]
    no_action = "- 无" if prev_files else "- 首次运行，没有上一交易日的信号可对比（明天起列出变化）"
    lines += ["## 需要操作（目标仓位变化 > 10 个百分点）", ""] + ([f"- {a}" for a in actions] or [no_action]) + [""]
    if len(sig):
        show = sig[["代码", "名称", "状态", "目标仓位", "收盘价", "建议止损价", "大盘过滤"]].copy()
        show["目标仓位"] = show["目标仓位"].map(lambda v: f"{v:.0%}")
        lines += ["## 单标的信号", "", show.to_markdown(index=False, floatfmt=".3f"), ""]
    if len(pf_df):
        show = pf_df.copy()
        show["目标权重"] = show["目标权重"].map(lambda v: f"{v:.1%}")
        show["金额"] = show["金额"].map(lambda v: f"{v:,.0f}")
        lines += [f"## 组合建议持仓（资金 {CAPITAL:,} 元，月度再平衡）", "", show.to_markdown(index=False), ""]
    lines += ["> 回测不代表未来收益，不构成投资建议。"]
    md = "\n".join(lines)
    (DAILY / f"每日信号_{tag}.md").write_text(md, encoding="utf-8")

    trading_day = last is not None and (demo or last.date() == today.date())
    if trading_day or bad or failed:
        res = notify.send(f"Qrant 每日信号 {tag}（操作 {len(actions)} 项）", md)
    else:
        res = "今日无新行情（非交易日或数据未更新），不推送"
    log(f"daily 完成：{len(sig)} 个信号，{len(actions)} 项操作，数据问题 {len(qc)} 个，{res}")
    return {"actions": actions, "issues": len(qc), "bad": sorted(bad), "stale": sorted(stale), "report": md}


def weekly(demo: bool = False) -> int:
    import validate_ashare
    code = validate_ashare.main(["--demo"] if demo else [])
    newest = sorted((ROOT / "reports").glob("对比_*.md"))
    summary = newest[-1].read_text(encoding="utf-8")[:1500] if newest and not demo else ""
    res = notify.send("Qrant 周度验证", summary or "周度验证完成，详见 reports/") if summary else "无对比报告"
    log(f"weekly 完成（返回码 {code}），{res}")
    return code


def quarterly(demo: bool = False) -> int:
    import validate_oos
    code = validate_oos.main(["--demo"] if demo else [])
    log(f"quarterly 完成（返回码 {code}）")
    return code


def main(argv: list[str]) -> int:
    demo = "--demo" in argv
    cmd = next((a for a in argv if not a.startswith("--")), "auto")
    now = datetime.now()
    try:
        if cmd == "daily":
            daily(demo)
        elif cmd == "weekly":
            return weekly(demo)
        elif cmd == "quarterly":
            return quarterly(demo)
        elif cmd == "auto":
            daily(demo)
            if now.weekday() == 4:
                weekly(demo)
                if now.month in (1, 4, 7, 10) and now.day <= 7:
                    quarterly(demo)
        else:
            print(__doc__)
            return 2
    except Exception as e:  # noqa: BLE001
        log(f"{cmd} 失败：{e}\n{traceback.format_exc()}")
        notify.send("Qrant 自动任务失败", f"{cmd}：{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
