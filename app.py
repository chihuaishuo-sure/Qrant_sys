"""Qrant —— 先求少亏的个人量化风控系统（Streamlit 界面）。

启动：双击 启动.bat（Windows）或运行  streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qrant import __version__, data, metrics, pipeline, validation
from qrant import backtest as bt
from qrant.config import (BENCHMARK_INDEX, BENCHMARK_NAME, DEFAULT_WATCHLIST, CostModel, RiskConfig,
                          is_ashare_equity, is_etf, price_limit)
from qrant.indicators import atr, sma
from qrant.risk import apply_risk, position_size, var_cvar
from qrant.strategies import STRATEGIES
from qrant.ui import (BLUE, CSS, DOWN, GOLD, GREY, UP, card, chart, drawdown_fig, equity_fig, hero,
                      monthly_heatmap, num, pct, style, table)

st.set_page_config(page_title="Qrant 量化风控", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

KIND_LABEL = {"自动识别": "auto", "ETF/场内基金": "etf", "A股个股": "stock", "指数": "index"}
PAGES = ["🏠 今日信号", "📈 策略回测", "🧺 组合配置", "🔬 稳健性检验", "🛡️ 风控工具箱", "📘 使用说明"]

# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.markdown(f"## 🛡️ Qrant\n<span class='q-pill'>v{__version__}</span><span class='q-pill'>先求少亏</span>",
                unsafe_allow_html=True)
    page = st.radio("导航", PAGES, label_visibility="collapsed")
    st.divider()
    source = st.radio("数据来源", ["AKShare 在线", "演示数据（模拟）"], index=0,
                      help="在线数据来自东方财富/新浪（免费），自动缓存到 data/cache；无网络时可用演示数据体验功能。")
    start_date = st.date_input("回测起始日", value=date(2013, 1, 1), min_value=date(2005, 1, 1),
                               max_value=date.today())
    capital = st.number_input("账户资金（元）", min_value=10_000, max_value=100_000_000, value=200_000,
                              step=10_000)
    if st.button("🔄 刷新行情缓存"):
        st.cache_data.clear()
        st.session_state["force_refresh"] = True
        st.toast("已清除缓存，下次加载将重新下载")
    st.divider()
    st.caption("⚠️ 本系统为风险管理与研究工具，不构成投资建议。历史回测不代表未来收益。")


# ------------------------------------------------------------------ 数据加载
@st.cache_data(ttl=3600, show_spinner=False)
def _load(symbol: str, kind: str, start: str, src: str, force: bool = False) -> tuple[pd.DataFrame, str]:
    if src.startswith("演示"):
        return data.demo_data(symbol, start), "演示数据（模拟，非真实行情）"
    return data.fetch(symbol, start, kind=kind, force=force)


def load(symbol: str, kind: str = "auto", warmup_days: int = 420) -> tuple[pd.DataFrame | None, str]:
    """多取一段历史用于指标预热（不进入回测区间）。"""
    s = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
    try:
        df, src = _load(symbol.strip(), kind, s, source, st.session_state.pop("force_refresh", False))
        return df, src
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ {symbol} 数据获取失败：{e}\n\n可尝试：① 检查网络 ② 侧边栏切换到“演示数据” ③ 稍后重试")
        return None, ""


def load_bench() -> pd.Series | None:
    try:
        df, _ = _load(BENCHMARK_INDEX, "index",
                      (pd.Timestamp(start_date) - pd.Timedelta(days=420)).strftime("%Y%m%d"), source)
        return df["close"]
    except Exception:  # noqa: BLE001
        return None


def name_of(code: str) -> str:
    return DEFAULT_WATCHLIST.get(code, data.demo_name(code) if source.startswith("演示") else code)


def bt_start(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start_date) - pd.Timedelta(days=420):]


def params_form(spec, prefix: str) -> dict:
    vals = {}
    if not spec.params:
        return vals
    cols = st.columns(len(spec.params))
    for c, p in zip(cols, spec.params):
        vals[p.key] = c.number_input(p.label, min_value=p.lo, max_value=p.hi, value=p.default, step=p.step,
                                     key=f"{prefix}_{spec.key}_{p.key}")
    return vals


def risk_form(prefix: str, default: RiskConfig | None = None) -> RiskConfig:
    d = default or RiskConfig()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vt = st.toggle("波动率目标", d.vol_target_on, key=f"{prefix}vt",
                       help="波动越大仓位越低。学术证据：显著降低左尾风险与最大回撤（Harvey 等 2018）。")
        tv = st.slider("目标年化波动", 0.05, 0.40, d.target_vol, 0.01, key=f"{prefix}tv", disabled=not vt)
    with c2:
        rg = st.toggle("大盘趋势过滤", d.regime_on, key=f"{prefix}rg",
                       help=f"{BENCHMARK_NAME} 跌破长期均线时限制仓位，避开系统性熊市。")
        rma = st.slider("大盘均线(日)", 60, 300, d.regime_ma, 10, key=f"{prefix}rma", disabled=not rg)
        bear = st.slider("熊市仓位上限", 0.0, 1.0, d.bear_exposure, 0.1, key=f"{prefix}bear", disabled=not rg)
    with c3:
        ast = st.toggle("ATR 跟踪止损", d.atr_stop_on, key=f"{prefix}ast",
                        help="个股建议开启；宽基 ETF 上会增加打脸交易。止损按次日开盘价成交（含跳空风险）。")
        am = st.slider("ATR 倍数", 1.5, 6.0, d.atr_mult, 0.5, key=f"{prefix}am", disabled=not ast)
    with c4:
        br = st.toggle("回撤熔断", d.breaker_on, key=f"{prefix}br",
                       help="账户自高点回撤超过阈值即清仓冷却，冷却期满且价格站上均线后恢复。")
        bdd = st.slider("熔断回撤阈值", 0.05, 0.40, d.breaker_dd, 0.01, key=f"{prefix}bdd", disabled=not br)
        bcd = st.slider("冷却交易日", 5, 60, d.breaker_cooldown, 5, key=f"{prefix}bcd", disabled=not br)
    c5, c6, c7 = st.columns([1, 1.2, 1])
    band = c5.slider("调仓阈值（仓位偏离超过才交易）", 0.02, 0.30, d.rebalance_band, 0.01, key=f"{prefix}band")
    cash = c7.number_input("空仓资金年化收益 %", 0.0, 5.0, d.cash_rate * 100, 0.1, key=f"{prefix}cash",
                           help="空仓的钱买货币基金/国债逆回购的收益。所有对比线都按同一利率计息，比较公平。") / 100
    freq = c6.radio("决策频率", ["每周（推荐，换手低）", "每日"], horizontal=True, key=f"{prefix}freq",
                    help="清仓信号无论如何都会在次日执行；“每周”只让加仓/建仓等到周五决策。")
    return RiskConfig(vol_target_on=vt, target_vol=tv, regime_on=rg, regime_ma=rma, bear_exposure=bear,
                      atr_stop_on=ast, atr_mult=am, breaker_on=br, breaker_dd=bdd, breaker_cooldown=bcd,
                      rebalance_band=band, decide_freq="W" if freq.startswith("每周") else "D", cash_rate=cash)


# ================================================================== 页面：今日信号
def page_signals():
    hero("🏠 今日信号", "每天收盘后看一眼：哪些该持有、哪些该降仓 —— 次日开盘执行。")
    bench = load_bench()
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        if bench is not None and len(bench) > 200:
            b, m = bench.iloc[-1], sma(bench, 200).iloc[-1]
            bull = b > m
            card(f"市场状态 · {BENCHMARK_NAME}", "🟥 多头趋势" if bull else "🟩 空头趋势",
                 f"距 200 日均线 {pct(b / m - 1, 1, True)} · 数据 {bench.index[-1]:%Y-%m-%d}",
                 "q-bad" if bull else "q-good")
        else:
            card("市场状态", "未知", "基准指数数据不可用，大盘过滤将使用各标的自身均线")
    with c2:
        strat_key = st.selectbox("信号策略", list(STRATEGIES)[:-1], format_func=lambda k: STRATEGIES[k].name)
    with c3:
        codes_txt = st.text_input("自选代码（逗号分隔）", ",".join(DEFAULT_WATCHLIST))
    codes = [c.strip() for c in codes_txt.replace("，", ",").split(",") if c.strip()]
    cfg = RiskConfig()

    rows, frames = [], {}
    prog = st.progress(0.0, text="正在计算信号…")
    for i, code in enumerate(codes):
        df, src = load(code)
        prog.progress((i + 1) / max(len(codes), 1), text=f"正在计算 {code}…")
        if df is None or len(df) < 260:
            continue
        frames[code] = df
        s = pipeline.latest_signal(df, code, strat_key, cfg=cfg, bench_close=bench)
        s["名称"] = name_of(code)
        s["数据来源"] = src
        rows.append(s)
    prog.empty()
    if not rows:
        st.warning("没有可用的数据。")
        return

    t = pd.DataFrame(rows)
    stale = (pd.Timestamp(datetime.now().date()) - pd.to_datetime(t["数据日期"])).dt.days > 5
    t["数据日期"] = pd.to_datetime(t["数据日期"]).dt.strftime("%Y-%m-%d") + np.where(stale, " ⚠️过期", "")
    t["目标仓位%"] = (t["目标仓位"] * 100).round(0)
    t["变化"] = t["较前日变化"].map(lambda v: "—" if abs(v) < 0.005 else (f"▲ 加仓 {v:.0%}" if v > 0 else f"▼ 减仓 {-v:.0%}"))
    show = t[["代码", "名称", "状态", "目标仓位%", "变化", "收盘价", "建议止损价", "距200日线", "年化波动", "大盘过滤", "数据日期"]].copy()
    show["距200日线"] = show["距200日线"].map(lambda v: pct(v, 1, True))
    show["年化波动"] = show["年化波动"].map(lambda v: pct(v, 1))

    n_att = int((t["目标仓位"] >= 0.5).sum())
    n_def = int((t["目标仓位"] <= 0.01).sum())
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card("进攻（≥50%仓位）", f"{n_att}", f"共 {len(t)} 个标的", "q-bad")
    with k2:
        card("防守（空仓）", f"{n_def}", "趋势转弱或大盘限仓", "q-good")
    with k3:
        avg = t["目标仓位"].mean()
        card("平均建议仓位", pct(avg, 0), "按单标的视角计算", "q-gold")
    with k4:
        card("今日需操作", f"{int((t['较前日变化'].abs() > 0.1).sum())}", "仓位变化超过 10% 的标的")

    st.markdown("#### 📋 信号明细")
    from streamlit import column_config as cc
    table(show, hide_index=True, column_config={
        "目标仓位%": cc.ProgressColumn("目标仓位", min_value=0, max_value=100, format="%d%%"),
        "收盘价": cc.NumberColumn(format="%.3f"),
        "建议止损价": cc.NumberColumn(format="%.3f", help=f"收盘价 − {cfg.atr_mult}×ATR"),
    })
    st.caption("规则：今日收盘计算 → 明日开盘执行。“目标仓位”是该标的单独持有时的建议比例；多标的配置请看「组合配置」。")

    st.markdown("#### 🔍 单标的走势")
    pick = st.selectbox("选择标的", list(frames), format_func=lambda c: f"{c} {name_of(c)}")
    df = frames[pick].iloc[-500:]
    ro = apply_risk(frames[pick], STRATEGIES[strat_key](frames[pick]), cfg,
                    bench if is_ashare_equity(pick) else None).final.loc[df.index]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="K线",
                                 increasing_line_color=UP, decreasing_line_color=DOWN))
    fig.add_trace(go.Scatter(x=df.index, y=sma(frames[pick]["close"], 200).loc[df.index], name="200日均线",
                             line=dict(color=GOLD, width=1.4)))
    fig.add_trace(go.Scatter(x=df.index, y=sma(frames[pick]["close"], 60).loc[df.index], name="60日均线",
                             line=dict(color=BLUE, width=1.1)))
    fig.add_trace(go.Scatter(x=ro.index, y=ro, name="目标仓位", yaxis="y2", fill="tozeroy",
                             line=dict(color="rgba(232,176,75,.6)", width=1), fillcolor="rgba(232,176,75,.12)"))
    style(fig, 460)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      yaxis2=dict(overlaying="y", side="right", range=[0, 3], tickformat=".0%", showgrid=False))
    chart(fig)


# ================================================================== 页面：策略回测
def page_backtest():
    hero("📈 策略回测", "四线对比：买入持有 · 裸策略 · 风控策略 · 同等平均仓位基准 —— 只有战胜后者才是真本事。")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.4])
        code = c1.text_input("代码", "510300", help="ETF 如 510300；个股如 600519；指数如 sh000300")
        kind = KIND_LABEL[c2.selectbox("类型", list(KIND_LABEL))]
        sk = c3.selectbox("策略", list(STRATEGIES)[:-1], format_func=lambda k: STRATEGIES[k].name)
        spec = STRATEGIES[sk]
        st.caption("💡 " + spec.desc)
        params = params_form(spec, "bt")
        with st.expander("🛡️ 风控设置（默认值已按研究结论设置）", expanded=False):
            cfg = risk_form("bt_")
        with st.expander("💰 交易成本", expanded=False):
            k1, k2, k3 = st.columns(3)
            comm = k1.number_input("佣金率（万分之）", 0.5, 30.0, 2.5, 0.5) / 10000
            slip = k2.number_input("滑点（千分之，单边）", 0.0, 10.0, 0.5, 0.1) / 1000
            minc = k3.number_input("最低佣金（元）", 0.0, 10.0, 5.0, 1.0)
        go_btn = st.button("🚀 开始回测", type="primary")

    if go_btn:
        df, src = load(code, kind)
        if df is None:
            return
        bench = load_bench()
        cost = CostModel(commission_rate=comm, slippage=slip, min_commission=minc,
                         etf=is_etf(code) or kind == "index", limit_pct=price_limit(code))
        with st.spinner("回测中…"):
            a = pipeline.analyze_single(df, code, sk, params, cfg, capital, cost, bench, begin=start_date)
        a.df = a.df.loc[pd.Timestamp(start_date):]
        st.session_state["bt"] = (a, src, code, kind)

    if "bt" not in st.session_state:
        st.info("👆 设置好参数后点击「开始回测」。第一次加载需要从网络下载行情，稍等几秒。")
        return
    a, src, code, kind = st.session_state["bt"]
    if kind != "etf" and not is_etf(code):
        st.warning("⚠️ 个股回测存在幸存者偏差（你选它往往因为它活到了今天），结果会偏乐观；建议优先用宽基 ETF。")
    res, m = a.results, a.metrics
    st.caption(f"标的 **{code} {name_of(code)}** · 区间 {a.start:%Y-%m-%d} ~ {a.df.index[-1]:%Y-%m-%d} · 数据：{src}")

    v = a.verdict
    items = "".join(f"<div>{'✅' if ok else '❌'} {k}</div>" for k, ok in v["checks"].items())
    if v["passed"]:
        st.markdown(f'<div class="q-verdict pass"><b>✅ 通过验收：风控策略具有真实的风险调整后优势</b>'
                    f'<div style="margin-top:6px">{items}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="q-verdict fail"><b>⚠️ 未完全通过：这套设置在该标的上的“优势”不充分</b>'
                    f'<div style="margin-top:6px">{items}</div>'
                    f'<div style="margin-top:6px;color:#8A94A6">建议：换用多周期动量、开启波动率目标，或改用宽基ETF/组合；'
                    f'不要为了通过而反复调参（那正是过拟合）。</div></div>', unsafe_allow_html=True)

    rk, bh = m["风控策略"], m["买入持有"]
    c = st.columns(5)
    with c[0]:
        card("最大回撤（风控 / 持有）", f"{pct(rk['最大回撤'])}", f"买入持有 {pct(bh['最大回撤'])}", "q-good")
    with c[1]:
        card("回撤降低", pct(v["dd_cut"], 0), "相对买入持有", "q-gold")
    with c[2]:
        card("年化收益", pct(rk["年化收益"]), f"买入持有 {pct(bh['年化收益'])}",
             "q-bad" if rk["年化收益"] > 0 else "q-good")
    with c[3]:
        card("卡玛比率", num(rk["卡玛比率"]), f"持有 {num(bh['卡玛比率'])} · 同仓位 {num(m['同仓位基准']['卡玛比率'])}")
    with c[4]:
        card("平均仓位 / 交易笔数", f"{pct(rk['平均仓位'], 0)}", f"{rk.get('成交笔数', 0)} 笔 · 成本 {res['风控策略'].total_cost:,.0f} 元")

    eqs = {k: r.equity for k, r in res.items()}
    t1, t2, t3, t4, t5, t6 = st.tabs(["净值对比", "回撤", "仓位与交易", "月度收益", "指标明细", "交易记录"])
    with t1:
        log = st.toggle("对数坐标", False)
        chart(equity_fig(eqs, log))
    with t2:
        chart(drawdown_fig({k: eqs[k] for k in ["风控策略", "买入持有", "同仓位基准"]}))
        st.caption("水下曲线：离历史最高点跌了多少。面积越小，持有体验越好、越不容易被迫割肉。")
    with t3:
        d = a.df
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d.close, name="收盘价", line=dict(color="#C9D1E0", width=1.2)))
        tr = res["风控策略"].trades
        for side, col, sym in [("买入", UP, "triangle-up"), ("卖出", DOWN, "triangle-down")]:
            x = tr[tr["方向"] == side]
            fig.add_trace(go.Scatter(x=x["日期"], y=x["成交价"], mode="markers", name=side,
                                     marker=dict(color=col, symbol=sym, size=8)))
        fig.add_trace(go.Scatter(x=res["风控策略"].exposure.index, y=res["风控策略"].exposure, name="实际仓位",
                                 yaxis="y2", fill="tozeroy", line=dict(color="rgba(232,176,75,.7)", width=1),
                                 fillcolor="rgba(232,176,75,.15)"))
        style(fig, 480)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 2.5], tickformat=".0%", showgrid=False))
        chart(fig)
        if res["风控策略"].breaker.max() > 0:
            st.info(f"回撤熔断共触发 {int((res['风控策略'].breaker.diff() == 1).sum())} 次。")
    with t4:
        chart(monthly_heatmap(metrics.monthly_table(eqs["风控策略"])))
        st.caption("风控策略月度收益（%）：红色盈利、绿色亏损（A股习惯）。")
    with t5:
        tbl = metrics.table(m)
        tbl.columns = [res[k].name for k in tbl.columns]
        table(tbl)
        st.caption("卡玛比率 = 年化收益 / 最大回撤，是“少亏”目标最核心的指标。Ulcer 指数衡量回撤的深度与持续时间。")
    with t6:
        tr = res["风控策略"].trades.copy()
        table(tr.style.format({"成交价": "{:.3f}", "金额": "{:,.0f}", "费用": "{:.2f}"}), hide_index=True)
        st.download_button("⬇️ 下载交易记录 CSV", tr.to_csv(index=False).encode("utf-8-sig"),
                           f"trades_{code}.csv", "text/csv")


# ================================================================== 页面：组合配置
def page_portfolio():
    hero("🧺 组合配置", "不把鸡蛋放在一个篮子里：按风险分配资金 + 每个资产单独做趋势过滤 + 组合波动率目标。")
    with st.form("pf"):
        opts = list(DEFAULT_WATCHLIST)
        c1, c2 = st.columns([2, 1])
        sel = c1.multiselect("资产池", opts, default=opts, format_func=lambda c: f"{c} {name_of(c)}")
        extra = c2.text_input("追加代码（可选，逗号分隔）", "")
        c3, c4, c5 = st.columns(3)
        method = c3.radio("配置方法", ["反波动率（风险均衡）", "等权"], horizontal=True)
        trend = c4.radio("择时过滤", ["多周期动量", "200日均线", "不择时"], horizontal=True)
        freq = c5.radio("再平衡", ["每月", "每周"], horizontal=True)
        c6, c7, c8 = st.columns(3)
        tv = c6.slider("组合目标波动", 0.04, 0.25, 0.10, 0.01)
        mw = c7.slider("单资产权重上限", 0.15, 1.0, 0.40, 0.05)
        brk = c8.toggle("回撤熔断（12%）", False)
        go_btn = st.form_submit_button("🚀 计算组合", type="primary")

    if go_btn:
        codes = sel + [c.strip() for c in extra.replace("，", ",").split(",") if c.strip()]
        if len(codes) < 2:
            st.error("至少选择 2 个资产")
            return
        datas = {}
        for c in codes:
            df, _ = load(c)
            if df is not None and len(df) > 300:
                datas[c] = df
        if len(datas) < 2:
            return
        with st.spinner("计算中…"):
            out = pipeline.analyze_portfolio(
                datas, capital, "inv_vol" if method.startswith("反") else "equal",
                {"多周期动量": "momentum", "200日均线": "ma", "不择时": "none"}[trend], tv, mw,
                "ME" if freq == "每月" else "W", RiskConfig(breaker_on=True, breaker_dd=0.12) if brk else None)
        st.session_state["pf"] = (out, datas)

    if "pf" not in st.session_state:
        st.info("👆 选择资产后点击「计算组合」。默认 6 只 ETF 覆盖 A股大/中/小盘、黄金、国债、美股。")
        return
    out, datas = st.session_state["pf"]
    res, m = out["results"], out["metrics"]
    rk, bh = m["风控组合"], m["等权买入持有"]
    c = st.columns(4)
    with c[0]:
        card("最大回撤", pct(rk["最大回撤"]), f"等权持有 {pct(bh['最大回撤'])}", "q-good")
    with c[1]:
        card("年化收益", pct(rk["年化收益"]), f"等权持有 {pct(bh['年化收益'])}", "q-bad" if rk["年化收益"] > 0 else "q-good")
    with c[2]:
        card("卡玛比率", num(rk["卡玛比率"]), f"等权持有 {num(bh['卡玛比率'])}", "q-gold")
    with c[3]:
        card("年化波动", pct(rk["年化波动"]), f"等权持有 {pct(bh['年化波动'])}")

    t1, t2, t3, t4 = st.tabs(["净值与回撤", "当前建议持仓", "历史权重", "指标明细"])
    with t1:
        eqs = {k: r.equity for k, r in res.items()}
        chart(equity_fig(eqs))
        chart(drawdown_fig(eqs, 260))
    with t2:
        w = out["targets"].iloc[-1]
        rows = []
        for s, wt in w.items():
            px = float(datas[s]["close"].iloc[-1])
            amt = capital * wt
            rows.append({"代码": s, "名称": name_of(s), "目标权重": wt, "金额(元)": amt,
                         "约合股数": int(amt / px // 100 * 100), "最新价": px})
        cash = 1 - w.sum()
        rows.append({"代码": "现金", "名称": "现金/货币基金", "目标权重": cash, "金额(元)": capital * cash,
                     "约合股数": 0, "最新价": np.nan})
        alloc = pd.DataFrame(rows)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            table(alloc.style.format({"目标权重": "{:.1%}", "金额(元)": "{:,.0f}", "最新价": "{:.3f}"}), hide_index=True)
            st.caption(f"决策日 {out['targets'].index[-1]:%Y-%m-%d}。被择时过滤掉的权重留作现金（可买货币基金/国债逆回购）。")
        with c2:
            fig = go.Figure(go.Pie(labels=alloc["名称"], values=alloc["目标权重"].clip(lower=0), hole=.55,
                                   textinfo="label+percent"))
            chart(style(fig, 340).update_layout(showlegend=False))
    with t3:
        W = out["targets"].loc[out["start"]:]
        fig = go.Figure()
        for s in W.columns:
            fig.add_trace(go.Scatter(x=W.index, y=W[s], name=name_of(s), stackgroup="one", mode="lines",
                                     line=dict(width=.5)))
        chart(style(fig, 420, pct_y=True))
    with t4:
        table(metrics.table(m))


# ================================================================== 页面：稳健性检验
def _grid_vals(p, n=6):
    vals = np.linspace(max(p.lo, p.default * 0.5), min(p.hi, p.default * 1.6), n)
    vals = {int(round(v)) if isinstance(p.default, int) else round(float(v), 2) for v in vals}
    vals.add(p.default)
    vals = sorted(vals)
    return vals


def page_robust():
    hero("🔬 稳健性检验", "回测好看 ≠ 实盘赚钱。这里专门找过拟合：参数敏感性、滚动样本外、Deflated Sharpe、蒙特卡洛。")
    with st.form("rb"):
        c1, c2, c3 = st.columns(3)
        code = c1.text_input("代码", "510300")
        sk = c2.selectbox("策略", [k for k in STRATEGIES if STRATEGIES[k].params and len(STRATEGIES[k].params) >= 2],
                          format_func=lambda k: STRATEGIES[k].name)
        metric = c3.selectbox("评价指标", ["卡玛比率", "夏普比率", "索提诺比率"])
        c4, c5, c6 = st.columns(3)
        train_y = c4.slider("训练窗口（年）", 2, 6, 3)
        test_y = c5.slider("检验窗口（年）", 1, 3, 1)
        horizon = c6.selectbox("蒙特卡洛期限", ["1年", "3年"])
        go_btn = st.form_submit_button("🔬 开始检验（约 10~30 秒）", type="primary")
    if not go_btn and "rb" not in st.session_state:
        st.info("👆 选择标的与策略后开始检验。系统会自动在默认参数附近构造 6×6 参数网格。")
        return
    if go_btn:
        df, src = load(code)
        if df is None:
            return
        bench = load_bench()
        spec = STRATEGIES[sk]
        p1, p2 = spec.params[0], spec.params[1]
        v1, v2 = _grid_vals(p1), _grid_vals(p2)
        cfg = RiskConfig()
        start = pd.Timestamp(start_date)
        rets, grid_score = {}, pd.DataFrame(index=v1, columns=v2, dtype=float)
        prog = st.progress(0.0, text="参数扫描中…")
        k = 0
        for a in v1:
            for b in v2:
                k += 1
                prog.progress(k / (len(v1) * len(v2)), text=f"参数扫描 {k}/{len(v1) * len(v2)}")
                prm = {**spec.defaults(), p1.key: a, p2.key: b}
                ro = apply_risk(df, spec(df, **prm), cfg, bench if is_ashare_equity(code) else None)
                r = pipeline.run_target(df.loc[start:], ro.final, code, capital, None, cfg.rebalance_band, cfg.decide_freq)
                lab = f"{p1.label}={a}, {p2.label}={b}"
                rets[lab] = r.returns
                grid_score.loc[a, b] = validation.score(r.returns, metric)
        prog.empty()
        bh = bt.run_single(df.loc[start:], pd.Series(1.0, index=df.loc[start:].index), code, capital).returns
        default_lab = f"{p1.label}={p1.default}, {p2.label}={p2.default}"
        wf = validation.walk_forward(rets, train_y * 252, test_y * 252, metric)
        sharpes = [validation.score(r, "夏普比率") for r in rets.values()]
        best_lab = max(wf["full_sample_scores"], key=lambda k: -np.inf if np.isnan(wf["full_sample_scores"][k]) else wf["full_sample_scores"][k])
        dsr = validation.deflated_sharpe(rets[best_lab], sharpes)
        hz = 252 if horizon == "1年" else 756
        ref = rets.get(default_lab, rets[best_lab])
        mc_s = validation.block_bootstrap(ref, 1000, hz)
        mc_b = validation.block_bootstrap(bh, 1000, hz)
        st.session_state["rb"] = dict(code=code, sk=sk, metric=metric, grid=grid_score, p1=p1, p2=p2, wf=wf,
                                      rets=rets, bh=bh, best=best_lab, dsr=dsr, mc_s=mc_s, mc_b=mc_b, hz=hz,
                                      default=default_lab if default_lab in rets else best_lab)
    R = st.session_state["rb"]
    t1, t2, t3, t4 = st.tabs(["参数敏感性", "Walk-Forward 样本外", "Deflated Sharpe", "蒙特卡洛回撤"])
    with t1:
        g = R["grid"]
        fig = go.Figure(go.Heatmap(z=g.to_numpy(dtype=float), x=[str(c) for c in g.columns], y=[str(i) for i in g.index],
                                   colorscale="Viridis", text=np.round(g.to_numpy(dtype=float), 2), texttemplate="%{text}",
                                   hovertemplate=f"{R['p1'].label}=%{{y}} {R['p2'].label}=%{{x}}<br>{R['metric']}=%{{z:.2f}}<extra></extra>"))
        style(fig, 460)
        fig.update_xaxes(title=R["p2"].label, type="category")
        fig.update_yaxes(title=R["p1"].label, type="category")
        fig.update_layout(hovermode="closest")
        chart(fig)
        z = g.to_numpy(dtype=float)
        frac = np.nanmean(z > 0) if np.isfinite(z).any() else np.nan
        cv = np.nanstd(z) / abs(np.nanmean(z)) if np.isfinite(z).any() and np.nanmean(z) != 0 else np.nan
        st.markdown(f"**解读**：{pct(frac, 0)} 的参数组合{R['metric']}为正；离散系数 {num(cv)}。"
                    "热力图是一大片“高原”说明策略稳健；只有一个孤立“尖峰”说明是过拟合，千万别用那个尖峰参数。")
    with t2:
        wf = R["wf"]
        oos = wf["oos_returns"]
        if len(oos) > 20:
            idx = oos.index
            eqs = {
                "样本外（Walk-Forward）": validation.equity_from_returns(oos),
                "全样本最优（过拟合）": validation.equity_from_returns(R["rets"][R["best"]].loc[idx]),
                "买入持有": validation.equity_from_returns(R["bh"].reindex(idx).fillna(0)),
            }
            chart(equity_fig(eqs))
            ms = {k: metrics.summary(v) for k, v in eqs.items()}
            c = st.columns(3)
            for i, (k, mm) in enumerate(ms.items()):
                with c[i]:
                    card(k, num(mm[R["metric"]]), f"年化 {pct(mm['年化收益'])} · 回撤 {pct(mm['最大回撤'])}")
            st.caption("“全样本最优”用了未来信息挑参数，现实中不可能拿到；“样本外”才是你真实能得到的。两者差距越大，过拟合越严重。")
        if len(wf["folds"]):
            f = wf["folds"]
            table(f.style.format({c: "{:.2f}" for c in f.columns if "比率" in c} | {"样本外收益": "{:.1%}"}),
                  hide_index=True)
        else:
            st.warning("数据长度不足以做滚动检验，请把回测起始日调早或缩短训练窗口。")
    with t3:
        d = R["dsr"]
        ok = d["dsr"] >= 0.95 if not np.isnan(d["dsr"]) else False
        c = st.columns(3)
        with c[0]:
            card("Deflated Sharpe（真实概率）", pct(d["dsr"], 1), "≥95% 才可信", "q-good" if ok else "q-bad")
        with c[1]:
            card("最优参数年化夏普", num(d["sr"]), R["best"])
        with c[2]:
            card("运气门槛夏普", num(d["sr0"]), f"试了 {d['n']} 组参数后，纯靠运气也能达到")
        st.markdown(
            "Bailey & López de Prado (2014)：**你试的参数越多，最好那组的夏普里“运气”成分越大。** "
            "DSR 扣除了多重测试、偏度和肥尾的影响，给出“真实夏普 > 0”的概率。"
            + ("✅ 当前结果可信度高。" if ok else "⚠️ 当前结果不足以排除运气，建议使用默认参数、多资产组合，并降低预期。"))
    with t4:
        mc_s, mc_b = R["mc_s"], R["mc_b"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mc_b["max_dd"], name="买入持有", marker_color=GREY, opacity=.6, nbinsx=50))
        fig.add_trace(go.Histogram(x=mc_s["max_dd"], name="风控策略(默认参数)", marker_color=GOLD, opacity=.75, nbinsx=50))
        style(fig, 380, f"未来 {R['hz'] // 252} 年最大回撤分布（块自助法 1000 次模拟）")
        fig.update_layout(barmode="overlay", hovermode="closest")
        fig.update_xaxes(tickformat=".0%")
        chart(fig)
        ps, pb = validation.prob_loss_exceeds(mc_s["max_dd"]), validation.prob_loss_exceeds(mc_b["max_dd"])
        tbl = pd.DataFrame({"风控策略": ps, "买入持有": pb}).map(lambda v: pct(v, 1))
        tbl.loc["回撤中位数"] = [pct(np.median(mc_s["max_dd"])), pct(np.median(mc_b["max_dd"]))]
        tbl.loc["最差5%情形回撤"] = [pct(np.percentile(mc_s["max_dd"], 5)), pct(np.percentile(mc_b["max_dd"], 5))]
        table(tbl)


# ================================================================== 页面：风控工具箱
def page_tools():
    hero("🛡️ 风控工具箱", "系统再好，执行不到位也会亏。这里把纪律变成数字。")
    t1, t2, t3, t4 = st.tabs(["仓位计算器", "持仓风险(VaR)", "交易前检查清单", "亏损恢复表"])
    with t1:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            acct = st.number_input("账户总资金", 10_000, 100_000_000, int(capital), 10_000, key="ps_acct")
            risk_pct = st.slider("单笔最多亏账户的", 0.002, 0.03, 0.01, 0.001, format="%.3f",
                                 help="职业交易员通常 0.5%~1%。连亏 10 次也只亏约 10%。")
            maxpos = st.slider("单只最大仓位", 0.05, 1.0, 0.30, 0.05)
            code = st.text_input("代码（可选：自动按 ATR 算止损）", "")
            entry_default, stop_default = 10.0, 9.3
            if code.strip():
                df, _ = load(code)
                if df is not None:
                    entry_default = float(df["close"].iloc[-1])
                    stop_default = round(entry_default - 2.5 * float(atr(df, 20).iloc[-1]), 3)
                    st.caption(f"最新收盘 {entry_default:.3f}，建议止损 = 收盘 − 2.5×ATR(20) = {stop_default:.3f}")
            entry = st.number_input("计划买入价", 0.001, 100000.0, entry_default, 0.01, format="%.3f")
            stop = st.number_input("止损价", 0.001, 100000.0, stop_default, 0.01, format="%.3f")
        with c2:
            r = position_size(acct, risk_pct, entry, stop, maxpos)
            if not r["ok"]:
                st.error(r["msg"])
            else:
                k = st.columns(2)
                with k[0]:
                    card("建议买入", f"{r['shares']:,} 股", f"{r['shares'] // 100} 手 · 受限于{r['limited_by']}", "q-gold")
                with k[1]:
                    card("占用资金", f"{r['value']:,.0f} 元", f"占账户 {pct(r['position_pct'])}")
                k = st.columns(2)
                with k[0]:
                    card("若触发止损亏损", f"{r['loss_if_stopped']:,.0f} 元", f"账户的 {pct(r['loss_pct'], 2)}", "q-good")
                with k[1]:
                    card("止损距离", pct(r["stop_distance_pct"]), "距离太近易被洗出，太远仓位会很小")
                st.markdown("> **核心思想**：先决定“最多亏多少”，再倒推“买多少”。这样任何一笔交易都不可能让你伤筋动骨。")
    with t2:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            code = st.text_input("持仓代码", "510300", key="var_code")
            hold = st.number_input("持仓市值（元）", 1000, 100_000_000, 100_000, 1000)
            days = st.selectbox("持有期", [1, 5, 10, 20], index=0, format_func=lambda d: f"{d} 个交易日")
        df, _ = load(code)
        if df is not None:
            r = df["close"].pct_change().dropna().iloc[-750:]
            rn = (1 + r).rolling(days).apply(np.prod, raw=True).dropna() - 1 if days > 1 else r
            v95, c95 = var_cvar(rn, 0.95)
            v99, c99 = var_cvar(rn, 0.99)
            with c2:
                k = st.columns(2)
                with k[0]:
                    card(f"VaR 95%（{days}日）", f"{hold * v95:,.0f} 元", f"20 次里约有 1 次亏损超过 {pct(v95)}", "q-good")
                    card(f"VaR 99%（{days}日）", f"{hold * v99:,.0f} 元", f"100 次里约有 1 次超过 {pct(v99)}", "q-good")
                with k[1]:
                    card("CVaR 95%（尾部平均）", f"{hold * c95:,.0f} 元", "最坏 5% 情况下的平均亏损", "q-bad")
                    card("CVaR 99%", f"{hold * c99:,.0f} 元", "极端情形平均亏损", "q-bad")
            fig = go.Figure(go.Histogram(x=rn, nbinsx=80, marker_color=BLUE, opacity=.8))
            fig.add_vline(x=-v95, line_color=GOLD, annotation_text="VaR95")
            fig.add_vline(x=-v99, line_color=UP, annotation_text="VaR99")
            style(fig, 300, f"近 3 年 {days} 日收益分布")
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(hovermode="closest")
            chart(fig)
            st.caption("如果 CVaR 金额让你睡不着觉，说明仓位太重 —— 按比例减仓直到可以接受。")
    with t3:
        st.markdown("每次下单前逐条确认，**全部勾选**才执行：")
        items = [
            "信号来自系统（收盘后计算），而不是盘中情绪或消息",
            "已用仓位计算器确定股数，单笔最大亏损 ≤ 账户 1%",
            "已设定止损价，并决定触发后无条件执行",
            "该笔买入后，单一标的仓位不超过上限（如 30%）",
            "本月账户亏损未触及红线（如 −6%）；触及则本月停止开新仓",
            "不是为了“回本”而加仓摊平亏损头寸",
            "已检查数据日期是最新交易日，没有用过期信号",
        ]
        done = [st.checkbox(t, key=f"ck{i}") for i, t in enumerate(items)]
        if all(done):
            st.success("✅ 全部确认，可以按计划执行。")
        else:
            st.warning(f"还有 {len(items) - sum(done)} 项未确认 —— 先别下单。")
    with t4:
        losses = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        need = 1 / (1 - losses) - 1
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"-{x:.0%}" for x in losses], y=losses, name="亏损", marker_color=DOWN))
        fig.add_trace(go.Bar(x=[f"-{x:.0%}" for x in losses], y=need, name="回本所需涨幅", marker_color=UP,
                             text=[pct(v, 0) for v in need], textposition="outside"))
        style(fig, 400, "亏损越深，回本越难：亏 50% 需要涨 100%", pct_y=True)
        fig.update_layout(barmode="group")
        chart(fig)
        st.markdown("这就是本系统“先求少亏”的数学原因：**把最大回撤从 −50% 压到 −20%，回本所需涨幅从 100% 降到 25%。**")


# ================================================================== 页面：使用说明
def page_help():
    hero("📘 使用说明", "三分钟上手，以及你需要知道的方法论与局限。")
    st.markdown("""
### 快速上手
1. **今日信号**：每个交易日收盘后（15:30 以后）打开，查看自选 ETF 的目标仓位，次日开盘按信号调整。
2. **策略回测**：输入代码 → 选择策略 → 开始回测。重点看 **最大回撤** 与 **卡玛比率**，以及顶部的验收结论。
3. **组合配置**：多资产按风险分配，给出每只 ETF 应买多少钱、多少股。**推荐作为主力用法。**
4. **稳健性检验**：上线任何新参数前必须跑一遍，确认不是过拟合。
5. **风控工具箱**：下单前用仓位计算器和检查清单。

### 系统为什么能减少损失
| 机制 | 作用 | 证据 |
|---|---|---|
| 趋势/动量过滤 | 长期下跌趋势中空仓，避开主跌段 | Faber 择时；Moskowitz 等 TSMOM（2012）|
| 波动率目标 | 市场恐慌（高波动）时自动降仓 | Harvey 等（2018）：降低左尾和最大回撤 |
| 风险均衡配置 | 股、债、金互补，单一资产暴跌影响有限 | 风险平价 |
| 次日开盘成交 + 完整成本 | 回测不虚胖，所见即所得 | — |
| 同仓位基准 + Walk-Forward + DSR | 识别“假优势”与过拟合 | Bailey & López de Prado |

### 内置真实数据验证（开发时完成）
标普500 指数 1871–2026 月度数据（价格指数，不含分红）上，“价格站上10个月均线 + 波动率目标”：
最大回撤 **−84.8% → −29.6%**，年化 4.9% → 6.5%，卡玛比率 0.06 → 0.22（同等仓位基准仅 0.05）。

### 必须知道的局限
- 趋势策略在**震荡市**会连续小亏（来回打脸），这是换取“避开大跌”的保险费。
- 回测不包含现金利息与分红；空仓资金可买货币基金/国债逆回购，实盘略好于回测。
- 前复权价格用于计算收益，整手股数按复权价估算，早年数据的手数会有偏差。
- **任何历史结果都不保证未来**。本系统不构成投资建议，请独立判断、自负盈亏。

详细调研、PRD 与评审记录见项目 `docs/` 文件夹。
""")


{PAGES[0]: page_signals, PAGES[1]: page_backtest, PAGES[2]: page_portfolio, PAGES[3]: page_robust,
 PAGES[4]: page_tools, PAGES[5]: page_help}[page]()
