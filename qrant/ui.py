"""界面通用组件：主题、图表样式、跨版本兼容的 Streamlit 调用。"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 中国习惯：红涨绿跌
UP = "#F0524F"
DOWN = "#17B26A"
GOLD = "#E8B04B"
BLUE = "#4C8DF6"
GREY = "#8A94A6"
PURPLE = "#A78BFA"
LINE_COLORS = {"风控策略": GOLD, "买入持有": GREY, "裸策略": BLUE, "同仓位基准": PURPLE,
               "风控组合": GOLD, "等权买入持有": GREY, "配置不择时": BLUE,
               "样本外（Walk-Forward）": GOLD, "全样本最优（过拟合）": UP}

CSS = """
<style>
:root { --gold:#E8B04B; --card:#121826; --line:#232B3D; --muted:#8A94A6; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: .5px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0E1422 0%,#0A0F1A 100%); border-right:1px solid var(--line); }
.q-hero { padding: 18px 22px; border-radius: 16px; border:1px solid var(--line);
  background: radial-gradient(1200px 200px at 0% 0%, rgba(232,176,75,.14), transparent 60%), #0F1523; margin-bottom: 14px; }
.q-hero h1 { margin: 0; font-size: 1.65rem; }
.q-hero p { margin: 4px 0 0; color: var(--muted); }
.q-card { background: var(--card); border:1px solid var(--line); border-radius: 14px; padding: 14px 16px; height: 100%; }
.q-card .lbl { color: var(--muted); font-size: .82rem; }
.q-card .val { font-size: 1.55rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
.q-card .sub { color: var(--muted); font-size: .8rem; margin-top: 2px; }
.q-good { color: #17B26A; } .q-bad { color: #F0524F; } .q-gold { color: var(--gold); }
.q-verdict { border-radius: 14px; padding: 14px 18px; margin: 6px 0 14px; border:1px solid; }
.q-verdict.pass { background: rgba(23,178,106,.08); border-color: rgba(23,178,106,.45); }
.q-verdict.fail { background: rgba(240,82,79,.07); border-color: rgba(240,82,79,.45); }
.q-verdict b { font-size: 1.05rem; }
.q-pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; margin-right:6px;
  border:1px solid var(--line); background:#0F1523; }
.q-state { font-size: 1.25rem; font-weight: 700; }
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#0F1523; border-radius: 10px 10px 0 0; padding: 6px 14px; }
footer { visibility: hidden; }
</style>
"""


def _width_kw(fn) -> dict:
    """兼容新旧版 Streamlit：新版用 width="stretch"，旧版用 use_container_width=True。"""
    try:
        p = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    w = p.get("width")
    if w is not None and isinstance(w.default, str):
        return {"width": "stretch"}
    if "use_container_width" in p:
        return {"use_container_width": True}
    return {}


def chart(fig: go.Figure, key: str | None = None):
    kw = _width_kw(st.plotly_chart)
    st.plotly_chart(fig, config={"displaylogo": False, "displayModeBar": False}, key=key, **kw)


def table(df, **kw):
    st.dataframe(df, **_width_kw(st.dataframe), **kw)


def style(fig: go.Figure, h: int = 420, title: str | None = None, pct_y: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=h, title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, PingFang SC, Noto Sans SC, sans-serif", color="#D7DCE5", size=12),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1C2333", zeroline=False)
    fig.update_yaxes(gridcolor="#1C2333", zeroline=False, tickformat=".0%" if pct_y else None)
    return fig


def card(label: str, value: str, sub: str = "", tone: str = ""):
    st.markdown(f'<div class="q-card"><div class="lbl">{label}</div>'
                f'<div class="val {tone}">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def hero(title: str, sub: str):
    st.markdown(f'<div class="q-hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def pct(v, d=1, sign=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def num(v, d=2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{d}f}"


def equity_fig(series: dict[str, pd.Series], log: bool = False, h: int = 430) -> go.Figure:
    fig = go.Figure()
    for name, s in series.items():
        s = s / s.iloc[0]
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines",
                                 line=dict(width=2.6 if name in ("风控策略", "风控组合") else 1.6,
                                           color=LINE_COLORS.get(name))))
    style(fig, h)
    fig.update_yaxes(type="log" if log else "linear", title="净值（起点=1）")
    return fig


def drawdown_fig(series: dict[str, pd.Series], h: int = 300) -> go.Figure:
    fig = go.Figure()
    for name, s in series.items():
        dd = s / s.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, mode="lines",
                                 fill="tozeroy" if name in ("风控策略", "风控组合", "买入持有", "等权买入持有") else None,
                                 line=dict(width=1.4, color=LINE_COLORS.get(name)), opacity=.9))
    return style(fig, h, pct_y=True)


def monthly_heatmap(mt: pd.DataFrame, h: int = 380) -> go.Figure:
    z = mt.to_numpy(dtype=float)
    text = [[("" if np.isnan(v) else f"{v * 100:.1f}") for v in row] for row in z]
    lim = np.nanmax(np.abs(z[:, :12])) if np.isfinite(z[:, :12]).any() else 0.1
    fig = go.Figure(go.Heatmap(
        z=z, x=list(mt.columns), y=[str(y) for y in mt.index], text=text, texttemplate="%{text}",
        colorscale=[[0, DOWN], [0.5, "#151B28"], [1, UP]], zmid=0, zmin=-lim, zmax=lim,
        showscale=False, hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
    style(fig, max(h, 26 * len(mt) + 60))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(hovermode="closest")
    return fig
