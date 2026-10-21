"""一键 A 股验证：用真实行情批量回测，生成报告（reports/ 文件夹）。

用法：
    python validate_ashare.py                    # 默认 6 只 ETF
    python validate_ashare.py 510300 600519      # 指定代码
    python validate_ashare.py --demo             # 离线演示数据（测试脚本本身）
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from qrant import data, metrics, pipeline
from dataclasses import replace

from qrant.config import BENCHMARK_INDEX, DEFAULT_WATCHLIST, RiskConfig

START = "20130101"
STRATS = ["multi_momentum", "ma_trend", "donchian", "ensemble"]
NAMES = {"multi_momentum": "多周期动量", "ma_trend": "均线趋势", "donchian": "唐奇安突破", "ensemble": "趋势组合信号"}
BASE = RiskConfig()
VARIANTS = {
    "A 旧默认（现金不计息）": replace(BASE, cash_rate=0.0),
    "B 新默认（现金1.2%）": BASE,
    "C B+熊市留30%仓位": replace(BASE, bear_exposure=0.3),
    "D B+目标波动20%": replace(BASE, target_vol=0.20),
    "E B+熊市留30%+目标波动20%": replace(BASE, bear_exposure=0.3, target_vol=0.20),
}
VAR_STRATS = ["multi_momentum", "ensemble"]
NOTES: list[str] = []
OUT = Path(__file__).resolve().parent / "reports"


def get(code: str, kind: str, demo: bool) -> pd.DataFrame:
    if demo:
        return data.demo_data(code, "20120101")
    df, src = data.fetch(code, "20120101", kind=kind)
    if "已修正" in src:
        NOTES.append(f"{code}：{src.split('已修正：')[1]}")
    print(f"  {code}: {len(df)} 行，{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}（{src}）")
    return df


def main(argv: list[str]) -> int:
    demo = "--demo" in argv
    codes = [a for a in argv if not a.startswith("--")] or list(DEFAULT_WATCHLIST)
    print(f"[Qrant] 一键验证：{len(codes)} 个标的 × {len(STRATS)} 个策略，起始 {START}" + ("（演示数据）" if demo else ""))

    print("[1/3] 下载行情 …")
    try:
        bench = get(BENCHMARK_INDEX, "index", demo)["close"]
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 基准指数获取失败，大盘过滤改用各标的自身均线：{e}")
        bench = None
    datas = {}
    for c in codes:
        try:
            datas[c] = get(c, "auto", demo)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {c} 获取失败，跳过：{str(e)[:120]}")
    if not datas:
        print("没有可用数据，请检查网络后重试。")
        return 1

    print("[2/3] 回测 …")
    rows = []
    for c, df in datas.items():
        for s in STRATS:
            try:
                a = pipeline.analyze_single(df, c, s, cfg=RiskConfig(), capital=200_000, bench_close=bench,
                                            begin=START)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {c} {s}: {e}")
                continue
            m = a.metrics
            rows.append({
                "代码": c, "名称": DEFAULT_WATCHLIST.get(c, c), "策略": NAMES[s],
                "区间": f"{a.start:%Y-%m}~{df.index[-1]:%Y-%m}",
                "持有_年化": m["买入持有"]["年化收益"], "持有_回撤": m["买入持有"]["最大回撤"],
                "持有_卡玛": m["买入持有"]["卡玛比率"],
                "风控_年化": m["风控策略"]["年化收益"], "风控_回撤": m["风控策略"]["最大回撤"],
                "风控_卡玛": m["风控策略"]["卡玛比率"], "同仓位_卡玛": m["同仓位基准"]["卡玛比率"],
                "平均仓位": m["风控策略"]["平均仓位"], "成交笔数": m["风控策略"].get("成交笔数", 0),
                "回撤降低": a.verdict["dd_cut"], "通过验收": a.verdict["passed"],
            })
            print(f"  {c} {NAMES[s]:6s} 回撤 {m['买入持有']['最大回撤']:.1%} → {m['风控策略']['最大回撤']:.1%}  "
                  f"卡玛 {m['买入持有']['卡玛比率']:.2f} → {m['风控策略']['卡玛比率']:.2f}  "
                  f"{'✅通过' if a.verdict['passed'] else '❌未通过'}")
    res = pd.DataFrame(rows)

    print("[2b] 改进方案对比 …")
    vrows = []
    for vn, cfg in VARIANTS.items():
        for c, df in datas.items():
            for s in VAR_STRATS:
                try:
                    a = pipeline.analyze_single(df, c, s, cfg=cfg, capital=200_000, bench_close=bench, begin=START)
                except Exception:  # noqa: BLE001
                    continue
                m = a.metrics
                vrows.append({"方案": vn, "代码": c, "策略": NAMES[s], "通过": a.verdict["passed"],
                              "回撤降低": a.verdict["dd_cut"], "风控年化": m["风控策略"]["年化收益"],
                              "收益差": m["风控策略"]["年化收益"] - m["买入持有"]["年化收益"],
                              "风控卡玛": m["风控策略"]["卡玛比率"], "平均仓位": m["风控策略"]["平均仓位"]})
    vr = pd.DataFrame(vrows)
    vsum = vr.groupby("方案", sort=False).agg(
        通过率=("通过", "mean"), 回撤降低中位数=("回撤降低", "median"), 年化中位数=("风控年化", "median"),
        相对持有收益差中位数=("收益差", "median"), 卡玛中位数=("风控卡玛", "median"), 平均仓位=("平均仓位", "mean"))
    for k in ["通过率", "回撤降低中位数", "年化中位数", "相对持有收益差中位数", "平均仓位"]:
        vsum[k] = vsum[k].map(lambda v: f"{v:.1%}")
    vsum["卡玛中位数"] = vsum["卡玛中位数"].map(lambda v: f"{v:.2f}")
    print(vsum.to_string())

    pf_tbl = None
    if len(datas) >= 2:
        try:
            pf = pipeline.analyze_portfolio({c: d.loc["20120101":] for c, d in datas.items()}, 200_000)
            pf_tbl = metrics.table(pf["metrics"]).loc[["年化收益", "年化波动", "最大回撤", "卡玛比率", "平均仓位", "成交笔数"]]
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ 组合回测失败：{e}")

    print("[3/3] 生成报告 …")
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = OUT / f"A股验证_{stamp}.csv"
    md_path = OUT / f"A股验证报告_{stamp}.md"
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    show = res.copy()
    for k in ["持有_年化", "持有_回撤", "风控_年化", "风控_回撤", "平均仓位", "回撤降低"]:
        show[k] = show[k].map(lambda v: f"{v:.1%}")
    for k in ["持有_卡玛", "风控_卡玛", "同仓位_卡玛"]:
        show[k] = show[k].map(lambda v: f"{v:.2f}")
    show["通过验收"] = show["通过验收"].map({True: "✅", False: "❌"})
    by_s = res.groupby("策略").agg(通过率=("通过验收", "mean"), 回撤降低中位数=("回撤降低", "median"),
                                 风控卡玛中位数=("风控_卡玛", "median"), 持有卡玛中位数=("持有_卡玛", "median"))
    lines = [
        f"# Qrant A股验证报告（{datetime.now():%Y-%m-%d %H:%M}）", "",
        f"数据：{'演示数据（模拟）' if demo else 'AKShare 真实行情（前复权）'}；起始 {START}；默认风控参数"
        f"（空仓资金按年化 {BASE.cash_rate:.1%} 计息，所有对比线相同）；"
        "t 日收盘出信号、t+1 开盘成交，含 A 股全部交易成本。", "",
        "数据修正：" + ("；".join(NOTES) if NOTES else "无"), "",
        "## 总览（按策略）", "",
        by_s.to_markdown(floatfmt=".2f") if _has_tabulate() else by_s.round(2).to_string(), "",
        f"**总体通过率：{res['通过验收'].mean():.0%}**（{int(res['通过验收'].sum())}/{len(res)}）。"
        "通过 = 回撤更小，且卡玛比率同时高于买入持有和同等平均仓位基准。", "",
        "## 明细", "",
        show.to_markdown(index=False) if _has_tabulate() else show.to_string(index=False), "",
    ]
    lines += ["## 改进方案对比（多周期动量 + 趋势组合信号，全部标的）", "",
              vsum.to_markdown() if _has_tabulate() else vsum.to_string(), "",
              "相对持有收益差 = 风控年化 − 买入持有年化（负数即“保险费”）。", ""]
    if pf_tbl is not None:
        lines += ["## 多资产风控组合", "",
                  pf_tbl.to_markdown() if _has_tabulate() else pf_tbl.to_string(), ""]
    lines += ["> 回测不代表未来收益，本报告不构成投资建议。"]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n总体通过率 {res['通过验收'].mean():.0%}；回撤降低中位数 {res['回撤降低'].median():.0%}")
    print(f"报告已保存：\n  {md_path}\n  {csv_path}")
    return 0


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
