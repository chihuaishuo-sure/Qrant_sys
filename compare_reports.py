"""对比两份 A股验证报告，找出代码/参数退化（区分于数据修正带来的变化）。

用法：
    python compare_reports.py                     # 自动取 reports/ 下最新两份
    python compare_reports.py 旧.csv 新.csv
    python compare_reports.py --strict            # 有退化时返回码 1（供自动化/CI 使用）
"""
from __future__ import annotations

import sys
from pathlib import Path

from qrant import compare

OUT = Path(__file__).resolve().parent / "reports"


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    files = [Path(a) for a in argv if not a.startswith("--")]
    if not files:
        files = compare.latest_reports(OUT)
    if len(files) != 2:
        print("需要两份报告：reports/ 下至少要有两份 A股验证_*.csv，或手动指定 旧.csv 新.csv")
        return 2
    old, new = files
    cmp = compare.compare(compare.load_report(old), compare.load_report(new))
    md = compare.to_markdown(cmp, old.stem, new.stem)
    path = OUT / f"对比_{old.stem.split('_', 1)[-1]}_vs_{new.stem.split('_', 1)[-1]}.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已保存：{path}")
    n_reg = compare.summary(cmp)["代码退化行"]
    return 1 if strict and n_reg else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
