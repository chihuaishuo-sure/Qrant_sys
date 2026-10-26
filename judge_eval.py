"""评测裁判 A：在植入 bug 的题库上算召回率、误报率、定位准确率。

用法：
    python judge_eval.py                         # 本地 qwen3.5:9b，不开思考
    python judge_eval.py --think                 # 开思考（慢很多）
    python judge_eval.py --votes 3               # 采样 3 次取最严
    python judge_eval.py --model qwen3.6:35b-a3b-coding-mtp-q4_K_M

标准答案来自执行判卷（qrant/judge_bank.oracle），不是人工标注。
“标记” = 裁决为 前视 或 不确定（不确定会转人工复查，所以对召回率算作抓到、对误报率算作误报）。
"""
from __future__ import annotations

import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from qrant import data, judge, judge_bank

OUT = Path(__file__).resolve().parent / "reports"


def located(findings: list[dict], planted: str) -> bool:
    """有效引用是否指向植入的那段代码（最长公共子串 ≥ 8 个非空白字符，或互相包含）。"""
    p = judge._norm(planted)
    for f in findings:
        q = judge._norm(str(f.get("quote", "")))
        if q and (q in p or p in q):
            return True
        if q and SequenceMatcher(None, q, p).find_longest_match(0, len(q), 0, len(p)).size >= 8:
            return True
    return False


def rate(num: int, den: int) -> str:
    return f"{num / den:.0%}（{num}/{den}）" if den else "—"


def main(argv: list[str]) -> int:
    think = "--think" in argv
    votes = int(argv[argv.index("--votes") + 1]) if "--votes" in argv else 1
    model = argv[argv.index("--model") + 1] if "--model" in argv else judge.DEFAULT_MODEL
    mode = f"{model}｜{'思考' if think else '不思考'}｜{'单次 temperature=0' if votes == 1 else f'{votes} 次采样取最严'}"
    print(f"[裁判A评测] {mode}")

    df = data.demo_data("510300")
    cases = judge_bank.build()
    reuse = {}
    if "--retry-from" in argv:  # 沿用上次已有答案的题，只重跑调用出错的题
        prev = pd.read_csv(argv[argv.index("--retry-from") + 1], encoding="utf-8-sig")
        reuse = {r["题目"]: r for r in prev.to_dict("records") if r["裁决"] != "错误"}
        print(f"  沿用 {len(reuse)} 题，重跑 {len(cases) - len(reuse)} 题")
    rows = []
    for i, c in enumerate(cases, 1):
        if c.id in reuse:
            rows.append({k: (None if pd.isna(v) else v) for k, v in reuse[c.id].items()})
            continue
        truth = judge_bank.oracle(c.code, df)
        v = judge.review_votes(c.code, n=votes, model=model, think=think)
        flagged = v.verdict in ("前视", "不确定")
        rows.append({"题目": c.id, "类型": c.kind, "真实有前视": truth, "裁决": v.verdict, "标记": flagged,
                     "正确": flagged == truth, "定位正确": located(v.findings, c.planted) if truth else None,
                     "引用": " | ".join(f["quote"] for f in v.findings)[:80],
                     "作废引用": " | ".join(v.invalid_quotes)[:60], "置信度": v.confidence,
                     "秒": round(v.seconds, 1), "tokens": v.tokens, "错误": v.error})
        print(f"  [{i:2d}/{len(cases)}] {c.id:28s} 真实={'前视' if truth else '干净'}  裁决={v.verdict}"
              f"  {'✓' if flagged == truth else '✗'}  {v.seconds:.1f}s", flush=True)
    r = pd.DataFrame(rows)
    r["真实有前视"] = r["真实有前视"].astype(bool)
    r["标记"] = r["标记"].astype(bool)
    errors = int((r["裁决"] == "错误").sum())
    r_all, r = r, r[r["裁决"] != "错误"]  # 调用出错的题不算答对也不算答错，单独报告

    pos, neg = r[r["真实有前视"]], r[~r["真实有前视"]]
    tp, fp = int(pos["标记"].sum()), int(neg["标记"].sum())
    tp_strict = int((pos["裁决"] == "前视").sum())
    fp_strict = int((neg["裁决"] == "前视").sum())
    loc = int(pos["定位正确"].fillna(False).sum())
    unsure = int((r["裁决"] == "不确定").sum())
    traps = neg[neg["类型"].str.startswith("陷阱")]

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"{model.replace(':', '-')}_{'think' if think else 'nothink'}_v{votes}"
    lines = [f"# 裁判 A（代码前视审查）评测 · {datetime.now():%Y-%m-%d %H:%M}", "",
             f"模式：{mode}；提示词版本 {judge.PROMPT_VERSION}；题库 {len(r_all)} 题，有效作答 {len(r)} 题"
             f"（有前视 {len(pos)}，干净 {len(neg)}，其中陷阱题 {len(traps)}）。标准答案来自执行判卷。", ""]
    if errors:
        lines += [f"⚠ **{errors} 题调用出错，未计入下列指标**；用 `--retry-from 本报告.csv` 只重跑这些题。", ""]
    lines += [
             "| 指标 | 结果 | 含义 |", "|:--|:--|:--|",
             f"| **召回率** | **{rate(tp, len(pos))}** | 有前视的题里，被标记（前视或不确定）的比例 |",
             f"| **误报率** | **{rate(fp, len(neg))}** | 干净的题里，被误标记的比例 |",
             f"| 召回率（只算“前视”） | {rate(tp_strict, len(pos))} | 不把“不确定”算作抓到 |",
             f"| 误报率（只算“前视”） | {rate(fp_strict, len(neg))} | |",
             f"| 定位准确率 | {rate(loc, len(pos))} | 有前视的题里，引用指向了植入 bug 那一行 |",
             f"| 陷阱题误报 | {rate(int(traps['标记'].sum()), len(traps))} | 看起来可疑、实际干净的题 |",
             f"| 不确定 / 调用错误 | {unsure} / {errors} | 不确定 = 说有前视但拿不出代码原文 |",
             f"| 耗时中位数 | {r['秒'].median():.1f} 秒/题 | 平均 {r['tokens'].mean():.0f} tokens |", ""]
    by_kind = r.groupby("类型").agg(题数=("正确", "size"), 答对=("正确", "sum")).reset_index()
    lines += ["## 按题型", "", by_kind.to_markdown(index=False), ""]
    wrong = r[~r["正确"].astype(bool)]
    if len(wrong):
        lines += ["## 答错的题", "", wrong[["题目", "真实有前视", "裁决", "引用", "作废引用"]].to_markdown(index=False), ""]
    if errors:
        failed = r_all[r_all["裁决"] == "错误"]
        lines += ["## 调用出错的题", "", failed[["题目", "真实有前视", "秒", "错误"]].to_markdown(index=False), ""]
    lines += ["## 全部明细", "", r_all.drop(columns=["错误"]).to_markdown(index=False), ""]
    OUT.mkdir(exist_ok=True)
    md = OUT / f"裁判A评测_{stamp}_{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    r_all.to_csv(md.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print("\n".join(lines[:16]))
    print(f"\n报告已保存：{md}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
