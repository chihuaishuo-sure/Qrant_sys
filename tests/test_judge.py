"""裁判 A 的离线测试：题库有效性、证据核对、裁决规则。不需要 Ollama。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qrant import data, judge, judge_bank  # noqa: E402


def test_bank_labels_match_execution_oracle():
    df = data.demo_data("510300")
    cases = judge_bank.build()
    assert sum(c.intent for c in cases) >= 15 and sum(not c.intent for c in cases) >= 15
    wrong = [c.id for c in cases if judge_bank.oracle(c.code, df) != c.intent]
    assert not wrong, f"题库设计意图与执行判卷不一致：{wrong}"


def test_oracle_catches_subtle_and_early_lookahead():
    df = data.demo_data("510300")
    early_bfill = 'def signal(df):\n    c = df["close"]\n    return (c / c.shift(20) - 1 > 0).astype(float).where(c.shift(20).notna()).bfill()\n'
    assert judge_bank.oracle(early_bfill, df)  # 只影响前 20 天
    clean = 'def signal(df):\n    c = df["close"]\n    return (c > c.rolling(50).mean()).astype(float)\n'
    assert not judge_bank.oracle(clean, df)


def test_quotes_must_come_from_code_not_comments():
    code = "def signal(df):\n    # 不要写 c.shift(-1)\n    c = df['close']\n    return c.shift(1)\n"
    assert judge.quote_in_code("c.shift(1)", code)
    assert judge.quote_in_code("return   c.shift(1)", code)  # 忽略空白
    assert not judge.quote_in_code("c.shift(-1)", code)  # 只出现在注释里
    assert not judge.quote_in_code("c.s", code)  # 太短不算证据


def test_decide_downgrades_unsupported_claims():
    code = "def signal(df):\n    return df['close'].shift(-1)\n"
    ok = judge.decide({"has_lookahead": True, "findings": [{"quote": "shift(-1)", "reason": "x"}]}, code)
    assert ok.verdict == "前视"
    made_up = judge.decide({"has_lookahead": True, "findings": [{"quote": "rolling(center=True)", "reason": "x"}]}, code)
    assert made_up.verdict == "不确定" and made_up.invalid_quotes
    assert judge.decide({"has_lookahead": False, "findings": []}, code).verdict == "干净"


def test_votes_take_strictest(monkeypatch):
    seq = iter(["干净", "前视", "不确定"])
    monkeypatch.setattr(judge, "review", lambda code, **kw: judge.Verdict(next(seq), None, seconds=1.0))
    v = judge.review_votes("x", n=3)
    assert v.verdict == "前视" and v.seconds == 3.0
