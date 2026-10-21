"""试验登记簿：记录每一次“试过什么”。

多重检验的核心问题是：试了 100 次挑出最好的，和只试 1 次，同一个夏普的可信度完全不同。
人会忘记自己试过多少次，登记簿不会。Deflated Sharpe、PBO 都需要这个“试验次数”。
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent.parent / "reports" / "试验登记.csv"
COLS = ["时间", "类型", "说明", "方案数", "结果摘要", "代码版本"]


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PATH.parent.parent,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "-"
    except Exception:  # noqa: BLE001
        return "-"


def log(kind: str, desc: str, n_configs: int, result: str, path: Path = PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([[f"{datetime.now():%Y-%m-%d %H:%M}", kind, desc, n_configs, result, _commit()]], columns=COLS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def load(path: Path = PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=COLS)


def total_configs(kind: str | None = None, path: Path = PATH) -> int:
    df = load(path)
    if kind:
        df = df[df["类型"] == kind]
    return int(df["方案数"].sum()) if len(df) else 0
