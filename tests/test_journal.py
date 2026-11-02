"""实盘记录与偏差跟踪的离线测试（演示数据，不联网）。"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrant import data, journal  # noqa: E402

CODE = "510300"
START = "2024-01-02"
NO_DIV = pd.DataFrame(columns=["日期", "代码", "每份分红"])


@pytest.fixture()
def env():
    df = data.demo_data(CODE)
    adj = {CODE: df}
    ro = df[["open"]].rename(columns={"open": CODE}).loc[START:]
    rc = df[["close"]].rename(columns={"close": CODE}).loc[START:]
    acc = journal.Account(起始资金=3000, 开始日期=START, 跟踪标的=[CODE], 最低佣金=0)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
        folder = Path(d)
        journal.init(acc, folder)
        yield acc, adj, ro, rc, folder


def _model_trades(acc, adj):
    t = journal.paper(acc, adj, None)["trades"]
    return t[t["方向"].isin(["买入", "卖出"])].reset_index(drop=True)


def _follow(model, ro, folder, skip=(), delay=(), worse_bp=0.0):
    cal = list(ro.index)
    for i, t in model.iterrows():
        if i in skip:
            continue
        d = pd.Timestamp(t.日期)
        if i in delay:
            d = cal[cal.index(d) + 1]
        px = float(ro.at[d, CODE]) * (1 + (worse_bp if t.方向 == "买入" else -worse_bp) / 1e4)
        journal.add_trade(d, CODE, t.方向, t.数量, round(px, 4), 0.1, folder=folder)


def test_add_trade_validates(env):
    acc, adj, ro, rc, folder = env
    with pytest.raises(ValueError):
        journal.add_trade(START, CODE, "买入", 150, 4.0, folder=folder)  # 非整手
    with pytest.raises(ValueError):
        journal.add_trade(START, CODE, "加仓", 100, 4.0, folder=folder)
    journal.add_trade(START, CODE, "未执行", note="没看盘", folder=folder)
    assert journal.load_trades(folder).iloc[0]["方向"] == "未执行"


def test_perfect_follower_matches_paper(env):
    acc, adj, ro, rc, folder = env
    model = _model_trades(acc, adj)
    assert len(model) >= 4
    _follow(model, ro, folder)
    res = journal.analyze(acc, journal.load_trades(folder), adj, ro, rc, events=NO_DIV)
    s = res["summary"]
    assert s["按时"] == s["系统交易数"] == len(model) and s["未执行"] == s["计划外"] == 0
    assert abs(s["平均滑点bp"]) < 0.1  # 价格取 4 位小数
    assert abs(s["执行差"]) < 0.01  # 只差纸面假设的滑点和费用取整


def test_deviations_are_classified(env):
    acc, adj, ro, rc, folder = env
    model = _model_trades(acc, adj)
    _follow(model, ro, folder, skip={1}, delay={2}, worse_bp=20)
    journal.add_trade(model.iloc[0]["日期"], CODE, "未执行", note="测试原因", folder=folder)  # 与跳过的无关，不影响
    extra_day = ro.index[-5]
    journal.add_trade(extra_day, CODE, "买入", 100, float(ro.iloc[-5][CODE]), folder=folder)
    m = journal.analyze(acc, journal.load_trades(folder), adj, ro, rc, events=NO_DIV)["matches"]
    assert m.loc[1, "执行"] == "未执行"
    assert m.loc[2, "执行"] == "延迟1天"
    assert (m["执行"] == "计划外").sum() == 1
    on_time = m[m["执行"] == "按时"]
    assert ((on_time["滑点bp"] - 20).abs() < 0.2).all()  # 买贵卖便宜都记为正（对你不利）；价格取 4 位小数有微小误差


def test_dividends_auto_added_unless_recorded(env):
    acc, adj, ro, rc, folder = env
    d0, d1 = ro.index[10], ro.index[30]
    journal.add_trade(d0, CODE, "买入", 300, float(ro.at[d0, CODE]), folder=folder)
    ev = pd.DataFrame([{"日期": d1, "代码": CODE, "每份分红": 0.1}])
    full, auto = journal.with_dividends(journal.load_trades(folder), 3000, ev)
    assert len(auto) == 1 and auto.iloc[0]["金额"] == 30.0
    journal.add_trade(d1 + pd.Timedelta(days=3), CODE, "分红", amount=29.5, folder=folder)
    _, auto2 = journal.with_dividends(journal.load_trades(folder), 3000, ev)
    assert auto2.empty  # 自己记了就以自己的为准


def test_next_actions_in_whole_lots(env):
    acc, adj, ro, rc, folder = env
    a = journal.next_actions(acc, journal.load_trades(folder), adj, rc)
    assert len(a) == 1 and a.iloc[0]["目标持股"] % 100 == 0


def test_journal_folder_is_git_ignored():
    r = subprocess.run(["git", "check-ignore", "-q", "journal/交易记录.csv"], cwd=ROOT)
    assert r.returncode == 0, "journal/ 必须被 .gitignore 忽略，个人交易记录不能上传"
