"""可选的消息推送：把每日信号/验证结果推到手机。

不配置就不推送，只写本地报告。配置方式（二选一）：
1. 项目根目录放 notify.json（已在 .gitignore 中，不会上传）：
   {"type": "serverchan", "key": "SCTxxxx"}                       Server酱 → 微信
   {"type": "wecom",      "url": "https://qyapi.weixin.qq.com/..."}  企业微信群机器人
   {"type": "dingtalk",   "url": "https://oapi.dingtalk.com/..."}    钉钉群机器人（关键词需含 Qrant）
   {"type": "feishu",     "url": "https://open.feishu.cn/..."}       飞书群机器人
2. 环境变量 QRANT_NOTIFY='同上 JSON'
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "notify.json"


def load_config() -> dict | None:
    raw = os.environ.get("QRANT_NOTIFY") or (CONFIG.read_text(encoding="utf-8") if CONFIG.exists() else "")
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def send(title: str, markdown: str, cfg: dict | None = None) -> str:
    """发送一条消息，返回结果说明。未配置时直接返回，不报错。"""
    cfg = cfg if cfg is not None else load_config()
    if not cfg:
        return "未配置推送（只写本地报告）"
    import requests

    t = cfg.get("type")
    text = markdown[:3800]  # 各家机器人都有长度限制
    try:
        if t == "serverchan":
            r = requests.post(f"https://sctapi.ftqq.com/{cfg['key']}.send", data={"title": title, "desp": text}, timeout=15)
        elif t == "wecom":
            r = requests.post(cfg["url"], json={"msgtype": "markdown", "markdown": {"content": f"**{title}**\n{text}"}},
                              timeout=15)
        elif t == "dingtalk":
            r = requests.post(cfg["url"], json={"msgtype": "markdown", "markdown": {"title": title, "text": f"### {title}\n{text}"}},
                              timeout=15)
        elif t == "feishu":
            r = requests.post(cfg["url"], json={"msg_type": "text", "content": {"text": f"{title}\n{text}"}}, timeout=15)
        else:
            return f"未知推送类型：{t}"
        return f"推送 {t}：HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return f"推送失败：{str(e)[:80]}"
