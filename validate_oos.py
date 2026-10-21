"""样本外检验：熊市留仓（方案 C）/ 熊市留仓 + 目标波动 20%（方案 E）是否真的优于当前默认（方案 B）。

C、E 是在 6 只默认 ETF × 2 个策略、2013–2026 全样本上比出来的，存在“挑参数”偏差。这里用三种样本外视角复核：
  1. 标的外：不参与挑选的 ETF；策略外：不参与挑选的均线趋势、唐奇安突破
  2. 时间切分：2013–2019 / 2020–今 两段分别验收
  3. Walk-Forward：每年只用过去 3 年数据在 B/C/D/E 中选卡玛最高者，下一年实盘检验

用法：python validate_oos.py [--demo]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qrant import data, pipeline, validation
from qrant.config import BENCHMARK_INDEX, DEFAULT_WATCHLIST, RiskConfig
from qrant.metrics import summary

START, SPLIT = "20130101", "20200101"
BASE = RiskConfig()
CFGS = {
    "B": BASE,
    "C": replace(BASE, bear_exposure=0.3),
    "D": replace(BASE, target_vol=0.20),
    "E": replace(BASE, bear_exposure=0.3, target_vol=0.20),
}
STRATS = {"multi_momentum": "多周期动量", "ensemble": "趋势组合信号", "ma_trend": "均线趋势", "donchian": "唐奇安突破"}
PICKED_STRATS = {"multi_momentum", "ensemble"}  # 挑选 C/E 时用过的策略
EXTRA = {"510050": "上证50ETF", "512100": "中证1000ETF", "159949": "创业板50ETF",
         "512880": "证券ETF", "512010": "医药ETF", "510880": "红利ETF"}
OUT = Path(__file__).resolve().parent / "reports"


def load(demo: bool) -> tuple[pd.Series | None, dict[str, pd.DataFrame]]:
    get = (lambda c, k: data.demo_data(c, "20120101")) if demo else (lambda c, k: data.fetch(c, "20120101", kind=k)[0])
    try:
        bench = get(BENCHMARK_INDEX, "index")["close"]
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 基准获取失败：{e}")
        bench = None
    out = {}
    for c in [*DEFAULT_WATCHLIST, *EXTRA]:
        try:
            out[c] = get(c, "auto")
            print(f"  {c}: {out[c].index[0]:%Y-%m-%d} ~ {out[c].index[-1]:%Y-%m-%d}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {c} 跳过：{str(e)[:80]}")
    return bench, out


def calmar(r: pd.Series) -> float:
    return float(summary(validation.equity_from_returns(r))["卡玛比率"])


def main(argv: list[str]) -> int:
    demo = "--demo" in argv
    print("[1/4] 行情 …")
    bench, datas = load(demo)

    print("[2/4] 全样本与分段回测 …")
    rows, rets = [], {}
    for c, df in datas.items():
        for s, sn in STRATS.items():
            for k, cfg in CFGS.items():
                periods = {"全样本": (df, START), "2013-2019": (df.loc[:"2019-12-31"], START),
                           "2020-今": (df, SPLIT)}
                for p, (d, beg) in periods.items():
                    if len(d.loc[beg:]) < 250:
                        continue
                    a = pipeline.analyze_single(d, c, s, cfg=cfg, capital=200_000, bench_close=bench, begin=beg)
                    m = a.metrics
                    rows.append({"代码": c, "名称": DEFAULT_WATCHLIST.get(c) or EXTRA[c], "策略": sn, "方案": k,
                                 "区间": p, "挑选用过": c in DEFAULT_WATCHLIST and s in PICKED_STRATS,
                                 "通过": a.verdict["passed"], "风控年化": m["风控策略"]["年化收益"],
                                 "风控回撤": m["风控策略"]["最大回撤"], "风控卡玛": m["风控策略"]["卡玛比率"],
                                 "收益差": m["风控策略"]["年化收益"] - m["买入持有"]["年化收益"],
                                 "平均仓位": m["风控策略"]["平均仓位"]})
                    if p == "全样本":
                        rets[(c, sn, k)] = a.results["风控策略"].returns
                        rets[(c, sn, "持有")] = a.results["买入持有"].returns
        print(f"  {c} 完成")
    res = pd.DataFrame(rows)

    def agg(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("方案").agg(组数=("通过", "size"), 通过率=("通过", "mean"), 卡玛中位数=("风控卡玛", "median"),
                                 回撤中位数=("风控回撤", "median"), 收益差中位数=("收益差", "median"))
        return g

    full = res[res["区间"] == "全样本"]
    views = {
        "挑选所用样本（6只默认ETF × 动量/组合信号）": full[full["挑选用过"]],
        "策略外：6只默认ETF × 均线/唐奇安": full[full["代码"].isin(list(DEFAULT_WATCHLIST)) & ~full["挑选用过"]],
        "标的外：新增ETF × 4个策略": full[full["代码"].isin(list(EXTRA))],
        "时间段 2013-2019（全部标的×策略）": res[res["区间"] == "2013-2019"],
        "时间段 2020-今（全部标的×策略）": res[res["区间"] == "2020-今"],
    }

    print("[3/4] Walk-Forward …")
    wf_rows, picks = [], []
    for c in datas:
        for sn in STRATS.values():
            cand = {k: rets[(c, sn, k)] for k in CFGS if (c, sn, k) in rets}
            if len(cand) < len(CFGS) or len(cand["B"]) < 756 + 252:
                continue
            wf = validation.walk_forward(cand, 756, 252, "卡玛比率")
            oos = wf["oos_returns"]
            span = oos.index
            row = {"代码": c, "策略": sn, "WF选择": calmar(oos)}
            for k in ["B", "C", "E"]:
                row[f"固定{k}"] = calmar(cand[k].reindex(span).fillna(0))
            row["持有"] = calmar(rets[(c, sn, "持有")].reindex(span).fillna(0))
            wf_rows.append(row)
            picks += list(wf["folds"]["选中参数"])
    wf = pd.DataFrame(wf_rows)

    print("[4/4] 熊市年份代价 …")
    yr_rows = []
    for (c, sn, k), r in rets.items():
        if k == "持有":
            continue
        y = (1 + r).groupby(r.index.year).prod() - 1
        bh = rets[(c, sn, "持有")]
        ybh = (1 + bh).groupby(bh.index.year).prod() - 1
        for yr, v in y.items():
            yr_rows.append({"代码": c, "策略": sn, "方案": k, "年": yr, "收益": v, "持有": ybh.get(yr, np.nan)})
    yr = pd.DataFrame(yr_rows).pivot_table(index=["代码", "策略", "年", "持有"], columns="方案", values="收益").reset_index()
    for k in ["C", "E"]:
        yr[f"{k}-B"] = yr[k] - yr["B"]
    bear = yr[yr["持有"] < -0.10]
    bull = yr[yr["持有"] > 0.10]

    # -------------------------------------------------------------- 报告
    lines = [f"# 方案 C / E 样本外检验（{datetime.now():%Y-%m-%d %H:%M}）", "",
             "B = 当前默认；C = B + 熊市留 30% 仓位；D = B + 目标波动 20%；E = C + 目标波动 20%。",
             f"数据：{'演示数据' if demo else 'AKShare 真实行情（前复权）'}；标的：默认 6 只 + 新增 {len([c for c in datas if c in EXTRA])} 只；"
             "通过 = 回撤小于持有，且卡玛同时高于持有和同仓位基准。", ""]
    for name, v in views.items():
        g = agg(v)
        for col in ["通过率", "回撤中位数", "收益差中位数"]:
            g[col] = g[col].map(lambda x: f"{x:.1%}")
        g["卡玛中位数"] = g["卡玛中位数"].map(lambda x: f"{x:.2f}")
        lines += [f"## {name}", "", g.to_markdown(), ""]

    if len(wf):
        s = wf[["WF选择", "固定B", "固定C", "固定E", "持有"]]
        lines += ["## Walk-Forward（训练 3 年 → 检验 1 年，滚动；只看样本外拼接段）", "",
                  f"组数 {len(wf)}；每年从 B/C/D/E 中按过去 3 年卡玛选一个。", "",
                  "| | WF选择 | 固定B | 固定C | 固定E | 买入持有 |", "|:--|--:|--:|--:|--:|--:|",
                  "| 样本外卡玛中位数 | " + " | ".join(f"{x:.2f}" for x in s.median()) + " |",
                  f"| 胜过固定B的组数占比 | {(s['WF选择'] > s['固定B']).mean():.0%} | — | "
                  f"{(s['固定C'] > s['固定B']).mean():.0%} | {(s['固定E'] > s['固定B']).mean():.0%} | — |", "",
                  "各方案被选中的次数：" + "，".join(f"{k} {v}" for k, v in pd.Series(picks).value_counts().items()), ""]

    def yr_line(df: pd.DataFrame, title: str) -> list[str]:
        return [f"| {title} | {len(df)} | {df['持有'].median():.1%} | {df['B'].median():.1%} | {df['C'].median():.1%} | "
                f"{df['E'].median():.1%} | {df['C-B'].median():+.1%} | {df['E-B'].median():+.1%} | "
                f"{df['C-B'].min():+.1%} | {df['E-B'].min():+.1%} |"]
    lines += ["## 代价：熊市年份 vs 牛市年份（年度收益中位数）", "",
              "| 年份类型 | 样本 | 持有 | B | C | E | C−B | E−B | C−B 最差 | E−B 最差 |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
              *yr_line(bear, "熊市年（持有 < −10%）"), *yr_line(bull, "牛市年（持有 > +10%）"), ""]
    lines += ["> 回测不代表未来收益，本报告不构成投资建议。"]

    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md = OUT / f"样本外检验_{stamp}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    res.to_csv(OUT / f"样本外检验_{stamp}.csv", index=False, encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"\n报告已保存：{md}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
