"""实盘记录：记下每一笔实际成交，和系统信号（纸面账户）、买入持有对比。

用法：
    python trade_log.py init --capital 3000 --codes 510300,159915 [--min-commission 0] [--start 2026-11-03]
    python trade_log.py buy  2026-11-04 510300 700 4.085 [--fee 0.71] [--note 备注]
    python trade_log.py sell 2026-11-20 510300 700 4.210 [--fee 0.74]
    python trade_log.py skip 2026-11-04 510300 --note 没时间看盘        # 信号要求操作但没做
    python trade_log.py dividend 2027-01-19 510300 86.10                # 可不记：系统会按除息前持股自动补上
    python trade_log.py today                                           # 明天开盘该做什么
    python trade_log.py report                                          # 实际 vs 纸面 vs 持有 + 执行明细

记录存在 journal/（不会上传到 GitHub），交易记录.csv 也可以直接用 Excel 编辑。
"""
from __future__ import annotations

import sys
from datetime import date

from qrant import data, journal
from qrant.config import BENCHMARK_INDEX


def _opt(argv: list[str], key: str, default=None):
    return argv[argv.index(key) + 1] if key in argv else default


def _positional(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a.startswith("--"):
            skip = True
        else:
            out.append(a)
    return out


def _bench():
    try:
        return data.fetch(BENCHMARK_INDEX, "20120101", kind="index")[0]["close"]
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    pos = _positional(rest)
    if cmd == "init":
        if journal.load_account() and "--force" not in rest:
            print("账户已存在（journal/账户.json）。要重建请加 --force（不会删除交易记录）。")
            return 1
        acc = journal.Account(起始资金=float(_opt(rest, "--capital", 3000)),
                              开始日期=_opt(rest, "--start", date.today().isoformat()),
                              跟踪标的=[c.strip().zfill(6) for c in _opt(rest, "--codes", "510300").split(",")],
                              策略=_opt(rest, "--strategy", "multi_momentum"),
                              最低佣金=float(_opt(rest, "--min-commission", 0)))
        journal.init(acc)
        print(f"已建立账户：{acc}")
        return 0

    acc = journal.load_account()
    if acc is None:
        print("还没有账户，先运行：python trade_log.py init --capital 3000 --codes 510300")
        return 1
    note = _opt(rest, "--note", "")
    if cmd in ("buy", "sell"):
        d, code, qty, px = pos[:4]
        fee = float(_opt(rest, "--fee", 0))
        row = journal.add_trade(d, code, "买入" if cmd == "buy" else "卖出", float(qty), float(px), fee, note=note)
        print(f"已记录：{row}")
    elif cmd == "skip":
        d, code = pos[:2]
        print(f"已记录：{journal.add_trade(d, code, '未执行', note=note)}")
    elif cmd == "dividend":
        d, code, amt = pos[:3]
        print(f"已记录：{journal.add_trade(d, code, '分红', amount=float(amt), note=note)}")
    elif cmd in ("today", "report"):
        bench = _bench()
        adj, ro, rc = journal.load_prices(acc.跟踪标的, acc.开始日期)
        trades = journal.load_trades()
        actions = journal.next_actions(acc, trades, adj, rc, bench)
        if cmd == "today":
            print(actions.to_string(index=False))
            return 0
        res = journal.analyze(acc, trades, adj, ro, rc, bench)
        md = journal.report_markdown(acc, res, actions)
        path = journal.DIR / f"实盘对比_{date.today():%Y%m%d}.md"
        path.write_text(md, encoding="utf-8")
        print(md)
        print(f"\n已保存：{path}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
