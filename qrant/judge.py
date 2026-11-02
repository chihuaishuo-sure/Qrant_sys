"""LLM 裁判 A：审查策略代码有没有前视偏差（用到 t 日之后的数据）。

只能“加严”：它报出的问题需要人工复查，但它说“没问题”不能替代 tests/ 里的执行检验。
默认调用本机 Ollama（不联网、不花钱）；模型可换，题库评测见 judge_eval.py。

防幻觉：判定有前视时，每条发现都必须逐字引用代码原文；引用对不上代码（或只来自注释）的发现作废，
全部作废的“有前视”降级为“不确定”。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:9b"
PROMPT_VERSION = "A-1"

SYSTEM = """你是量化回测代码审查员，只判断一件事：函数 signal(df) 在计算第 t 天的值时，是否用到了第 t 天之后的数据（前视偏差 / 未来函数）。

背景：df 是按日期升序排列的日线行情（open/high/low/close/volume），索引是交易日。回测引擎在 t+1 日开盘按 t 日的信号成交，所以使用 t 日当天的收盘价、最高价、最低价都不算前视。

逐项检查：
1. shift / diff / pct_change 的参数是否为负（负数 = 取未来的值）
2. rolling 是否 center=True（居中窗口包含未来）
3. 是否用整列的统计量处理每一天，如 c.mean()、c.std()、c.max()、c.quantile()、c.rank()（全样本 = 包含未来）
4. 是否有 bfill / backfill / interpolate 等用后面的值填前面的空缺
5. 循环里是否读取 i+1 等未来下标
6. 是否先反转序列再做累计，如 [::-1].cummax()[::-1]
7. np.gradient、居中平滑等两侧计算（会用到下一天）
8. resample 后对齐回日线时是否用了 bfill

以下只用过去数据，不算前视：shift(正数)、rolling 默认右对齐窗口、expanding、cummax/cummin、ewm（无论 adjust 取值）、ffill、rolling(...).apply、循环里读 i-1。代码注释里的文字不算代码。

如果判定有前视，findings 里每一条的 quote 必须逐字复制代码中的一段原文（不要改写），reason 用一句中文说明为什么用到了未来。没有前视时 findings 为空数组。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "has_lookahead": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"quote": {"type": "string"}, "reason": {"type": "string"}},
                      "required": ["quote", "reason"]},
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["has_lookahead", "findings", "confidence"],
}


@dataclass
class Verdict:
    verdict: str  # "前视" / "干净" / "不确定" / "错误"
    raw_has_lookahead: bool | None
    findings: list[dict] = field(default_factory=list)
    invalid_quotes: list[str] = field(default_factory=list)
    confidence: str = ""
    seconds: float = 0.0
    tokens: int = 0
    error: str = ""


def strip_comments(code: str) -> str:
    """去掉 # 注释（粗略处理：不考虑字符串里的 #，题库和策略代码里没有这种情况）。"""
    return "\n".join(line.split("#", 1)[0] for line in code.splitlines())


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def quote_in_code(quote: str, code: str) -> bool:
    """引用是否真实出现在代码（不含注释）里；忽略空白差异。太短的引用（< 4 个字符）不算证据。"""
    q = _norm(quote)
    return len(q) >= 4 and q in _norm(strip_comments(code))


def decide(raw: dict, code: str) -> Verdict:
    """把模型原始输出变成裁决，并核对证据。"""
    has = bool(raw.get("has_lookahead"))
    findings = [f for f in raw.get("findings", []) if isinstance(f, dict)]
    valid = [f for f in findings if quote_in_code(str(f.get("quote", "")), code)]
    invalid = [str(f.get("quote", "")) for f in findings if f not in valid]
    if not has:
        v = "干净"
    elif valid:
        v = "前视"
    else:
        v = "不确定"  # 说有前视却拿不出代码原文作证据
    return Verdict(v, has, valid, invalid, str(raw.get("confidence", "")))


def review(code: str, model: str = DEFAULT_MODEL, think: bool = False, temperature: float = 0.0,
           seed: int = 0, url: str = OLLAMA_URL, timeout: float = 1800, max_tokens: int = 8192) -> Verdict:
    body = {
        "model": model, "stream": False, "think": think, "format": SCHEMA,
        # num_predict 封顶：思考模式偶尔陷入循环，不封顶会一直生成到超时；num_ctx 留足空间给思考 + JSON
        "options": {"temperature": temperature, "seed": seed, "num_ctx": max(16384, max_tokens + 4096),
                    "num_predict": max_tokens},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"请审查下面的代码：\n```python\n{code}\n```"}],
    }
    t0 = time.time()
    try:
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        if out.get("done_reason") == "length":
            raise ValueError("生成达到长度上限（多半是思考陷入循环），没有给出完整答案")
        raw = json.loads(out["message"]["content"])
    except Exception as e:  # noqa: BLE001
        return Verdict("错误", None, error=str(e)[:200], seconds=time.time() - t0)
    v = decide(raw, code)
    v.seconds = time.time() - t0
    v.tokens = int(out.get("eval_count", 0))
    return v


def review_votes(code: str, n: int = 3, **kw) -> Verdict:
    """多次采样取最严：任意一次“前视”即前视；否则任意一次“不确定”即不确定。"""
    if n <= 1:
        return review(code, **kw)
    kw.setdefault("temperature", 0.7)
    runs = [review(code, seed=i, **{k: v for k, v in kw.items() if k != "seed"}) for i in range(n)]
    order = {"前视": 0, "不确定": 1, "错误": 2, "干净": 3}
    best = min(runs, key=lambda v: order[v.verdict])
    best.seconds = sum(v.seconds for v in runs)
    best.tokens = sum(v.tokens for v in runs)
    return best
