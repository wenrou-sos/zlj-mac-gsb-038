# -*- coding: utf-8 -*-
"""城市垃圾分类成效分析看板 (Streamlit + Plotly)

数据: data/ 目录下三张模拟 CSV
  communities.csv    小区/站点基础信息
  daily_records.csv  每日四类垃圾分出量 + 正确/混投量 + 参与户数
  contamination.csv  混投成分分析(小区 × 月)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
PLOTLY_TEMPLATE = "plotly_white"
FONT = "Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"
CAT_COLORS = {"厨余垃圾": "#2E8B57", "可回收物": "#1f77b4", "有害垃圾": "#d62728", "其他垃圾": "#7f7f7f"}

st.set_page_config(page_title="城市垃圾分类成效分析看板", page_icon="🗂️", layout="wide")


# ---------------- 数据加载 ----------------
@st.cache_data
def load_data():
    communities = pd.read_csv(DATA_DIR / "communities.csv")
    daily = pd.read_csv(DATA_DIR / "daily_records.csv", parse_dates=["date"])
    contamination = pd.read_csv(DATA_DIR / "contamination.csv")
    return communities, daily, contamination


communities, daily, contamination = load_data()


def accuracy(df: pd.DataFrame) -> float:
    """分类准确率 = 分对的量 / 总分出量"""
    total = df["total_kg"].sum()
    return df["correct_kg"].sum() / total if total > 0 else np.nan


def participation(df: pd.DataFrame) -> float:
    """参与率 = 参与户次 / 应参与户次(总户数×天数)"""
    denom = df["total_households"].sum()
    return df["participated_households"].sum() / denom if denom > 0 else np.nan


# ---------------- 侧边栏筛选 ----------------
st.sidebar.header("🔎 数据筛选")
min_d, max_d = daily["date"].min().date(), daily["date"].max().date()
date_range = st.sidebar.date_input("统计周期", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if len(date_range) != 2:
    st.sidebar.warning("请选择完整的起止日期")
    st.stop()

all_streets = sorted(daily["street"].unique())
sel_streets = st.sidebar.multiselect("街道", all_streets, default=all_streets)
comm_pool = communities[communities["street"].isin(sel_streets)]["community"].tolist()
sel_comms = st.sidebar.multiselect("小区(默认全部)", comm_pool, default=comm_pool)

st.sidebar.divider()
red_threshold = st.sidebar.slider("混投预警阈值:分类准确率低于", 0.60, 0.95, 0.80, 0.01, format="%.2f")
st.sidebar.caption("近 3 个月准确率连续下滑的小区也会列入预警")

mask = (
    daily["date"].dt.date.between(date_range[0], date_range[1])
    & daily["street"].isin(sel_streets)
    & daily["community"].isin(sel_comms)
)
df = daily.loc[mask].copy()
if df.empty:
    st.warning("当前筛选条件下没有数据,请调整筛选条件。")
    st.stop()

# ---------------- 标题 + KPI ----------------
st.title("🗂️ 城市垃圾分类成效分析看板")
st.caption(f"统计周期:{date_range[0]} ~ {date_range[1]} ｜ 覆盖 {df['street'].nunique()} 个街道 / {df['community'].nunique()} 个小区 ｜ 数据为模拟数据")

total_t = df["total_kg"].sum() / 1000
acc = accuracy(df)
part = participation(df)
mixed_t = df["mixed_kg"].sum() / 1000

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("累计分出垃圾总量", f"{total_t:,.1f} 吨")
k2.metric("整体分类准确率", f"{acc:.1%}", help="分对的垃圾量 ÷ 总分出量")
k3.metric("居民参与率", f"{part:.1%}", help="每天到定时定点站点投递的户数 ÷ 总户数")
k4.metric("混投垃圾量", f"{mixed_t:,.1f} 吨", delta=f"占比 {1 - acc:.1%}", delta_color="inverse")
k5.metric("四类分出结构", f"厨余 {df['kitchen_kg'].sum() / df['total_kg'].sum():.0%}", help="厨余垃圾占总分出量比例")

st.divider()

# ---------------- 街道排名 + 混投预警 ----------------
col_rank, col_alert = st.columns([3, 2])

with col_rank:
    st.subheader("🏆 各街道分类成效排名")
    street_stat = (
        df.groupby("street")
        .apply(lambda g: pd.Series({
            "分类准确率": accuracy(g),
            "居民参与率": participation(g),
            "日均分出量(吨)": g["total_kg"].sum() / 1000 / g["date"].nunique(),
        }), include_groups=False)
        .reset_index()
        .sort_values("分类准确率", ascending=True)
    )
    fig_rank = px.bar(
        street_stat, x="分类准确率", y="street", orientation="h",
        color="分类准确率", color_continuous_scale="RdYlGn",
        range_color=[max(0.5, street_stat["分类准确率"].min() - 0.05), 1],
        text=street_stat["分类准确率"].map("{:.1%}".format),
        hover_data={"居民参与率": ":.1%", "日均分出量(吨)": ":.2f"},
        labels={"street": ""},
    )
    fig_rank.add_vline(x=red_threshold, line_dash="dash", line_color="red",
                       annotation_text=f"预警线 {red_threshold:.0%}", annotation_position="top")
    fig_rank.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT, height=380,
                           coloraxis_showscale=False, xaxis_tickformat=".0%",
                           margin=dict(l=10, r=10, t=30, b=10))
    fig_rank.update_traces(textposition="outside")
    st.plotly_chart(fig_rank, width="stretch")

# ---- 混投高发小区判定 ----
comm_stat = (
    df.groupby(["street", "community"])
    .apply(lambda g: pd.Series({
        "分类准确率": accuracy(g),
        "居民参与率": participation(g),
        "混投量(吨)": g["mixed_kg"].sum() / 1000,
    }), include_groups=False)
    .reset_index()
)
monthly_comm = (
    df.assign(month=df["date"].dt.to_period("M").astype(str))
    .groupby(["community", "month"])
    .apply(accuracy, include_groups=False)
    .rename("acc").reset_index()
)

def declining_3m(community: str) -> bool:
    s = monthly_comm.loc[monthly_comm["community"] == community].sort_values("month")["acc"]
    s = s.tail(3)
    return len(s) == 3 and (s.diff().dropna() < 0).all()

comm_stat["连续3月下滑"] = comm_stat["community"].map(declining_3m)
red_flag = comm_stat[(comm_stat["分类准确率"] < red_threshold) | comm_stat["连续3月下滑"]]
red_flag = red_flag.sort_values("分类准确率")

with col_alert:
    st.subheader("🚨 混投高发小区预警")
    if red_flag.empty:
        st.success("当前筛选范围内无混投高发小区 🎉")
    else:
        show = red_flag.copy()
        show["分类准确率"] = show["分类准确率"].map("{:.1%}".format)
        show["居民参与率"] = show["居民参与率"].map("{:.1%}".format)
        show["混投量(吨)"] = show["混投量(吨)"].map("{:.2f}".format)
        show["预警原因"] = np.where(
            (red_flag["分类准确率"] < red_threshold) & red_flag["连续3月下滑"], "准确率低+连续下滑",
            np.where(red_flag["连续3月下滑"], "连续3月下滑", "准确率低于阈值"))
        show = show.rename(columns={"street": "街道", "community": "小区"})[
            ["街道", "小区", "分类准确率", "居民参与率", "混投量(吨)", "预警原因"]]
        st.dataframe(
            show.style.map(lambda _: "background-color:#ffe3e3;color:#b30000;font-weight:600"),
            width="stretch", hide_index=True, height=340)
        st.caption(f"共 {len(red_flag)} 个小区触发预警(准确率 < {red_threshold:.0%} 或连续 3 个月下滑)")

st.divider()

# ---------------- 月度趋势 ----------------
st.subheader("📈 分类准确率月度趋势")
trend_mode = st.radio("趋势维度", ["全市整体", "按街道", "按小区"], horizontal=True, label_visibility="collapsed")

monthly = (
    df.assign(month=df["date"].dt.to_period("M").astype(str))
    .groupby("month")
    .apply(lambda g: pd.Series({"分类准确率": accuracy(g), "居民参与率": participation(g)}), include_groups=False)
    .reset_index()
)
fig_trend = go.Figure()
if trend_mode == "全市整体":
    fig_trend.add_trace(go.Scatter(x=monthly["month"], y=monthly["分类准确率"], mode="lines+markers",
                                   name="分类准确率", line=dict(color="#1f77b4", width=3)))
    overall_first, overall_last = monthly["分类准确率"].iloc[0], monthly["分类准确率"].iloc[-1]
    direction = "📈 上升" if overall_last > overall_first else "📉 下降"
    st.caption(f"期内整体趋势:{direction}(首月 {overall_first:.1%} → 末月 {overall_last:.1%})")
else:
    dim = "street" if trend_mode == "按街道" else "community"
    tmp = (df.assign(month=df["date"].dt.to_period("M").astype(str))
           .groupby([dim, "month"]).apply(accuracy, include_groups=False).rename("acc").reset_index())
    for name, g in tmp.groupby(dim):
        flag = name in set(red_flag["community"])
        fig_trend.add_trace(go.Scatter(
            x=g["month"], y=g["acc"], mode="lines+markers", name=name,
            line=dict(width=3 if flag else 1.5, dash="solid"),
            opacity=1 if flag or dim == "street" else 0.55))
fig_trend.add_hline(y=red_threshold, line_dash="dash", line_color="red",
                    annotation_text=f"预警线 {red_threshold:.0%}")
fig_trend.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT, height=420,
                        yaxis_tickformat=".0%", yaxis_title="分类准确率", xaxis_title="",
                        hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_trend, width="stretch")

# ---------------- 每日四类垃圾量 + 参与率 ----------------
col_daily, col_part = st.columns([3, 2])

with col_daily:
    st.subheader("🗑️ 每日四类垃圾分出量")
    cat_map = {"kitchen_kg": "厨余垃圾", "recyclable_kg": "可回收物",
               "hazardous_kg": "有害垃圾", "other_kg": "其他垃圾"}
    daily_cat = df.groupby("date")[list(cat_map)].sum().reset_index()
    daily_cat["date"] = daily_cat["date"].dt.strftime("%Y-%m-%d")
    fig_daily = go.Figure()
    for col_name, label in cat_map.items():
        fig_daily.add_trace(go.Bar(x=daily_cat["date"], y=daily_cat[col_name],
                                   name=label, marker_color=CAT_COLORS[label]))
    fig_daily.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT, height=400, barmode="stack",
                            yaxis_title="分出量(kg)", xaxis_title="", xaxis_nticks=12,
                            margin=dict(l=10, r=10, t=20, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_daily, width="stretch")

with col_part:
    st.subheader("🏘️ 定时定点站点居民参与率")
    part_by_comm = (
        df.groupby("community")
        .apply(lambda g: pd.Series({"参与率": participation(g), "街道": g["street"].iloc[0]}), include_groups=False)
        .reset_index().sort_values("参与率", ascending=True)
    )
    fig_part = px.bar(part_by_comm, x="参与率", y="community", orientation="h",
                      color="参与率", color_continuous_scale="Blues",
                      text=part_by_comm["参与率"].map("{:.0%}".format),
                      labels={"community": ""})
    fig_part.add_vline(x=part, line_dash="dash", line_color="#ff7f0e",
                       annotation_text=f"平均 {part:.0%}", annotation_position="top")
    fig_part.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT, height=400,
                           coloraxis_showscale=False, xaxis_tickformat=".0%",
                           margin=dict(l=10, r=10, t=20, b=10))
    fig_part.update_traces(textposition="outside")
    st.plotly_chart(fig_part, width="stretch")

st.divider()

# ---------------- 混投成分分析 ----------------
st.subheader("🔬 混投成分分析:混进去的到底是什么?")
cont_mask = contamination["community"].isin(df["community"].unique())
cont = contamination.loc[cont_mask].copy()
months_in_range = set(df["date"].dt.to_period("M").astype(str).unique())
cont = cont[cont["month"].isin(months_in_range)]

c1, c2 = st.columns([3, 2])
with c1:
    focus_comms = red_flag["community"].tolist() if not red_flag.empty else \
        comm_stat.nsmallest(8, "分类准确率")["community"].tolist()
    cont_focus = cont[cont["community"].isin(focus_comms)]
    cont_stack = (cont_focus.groupby(["community", "contamination_type"])["weight_kg"].sum()
                  .reset_index())
    order = (cont_stack.groupby("community")["weight_kg"].sum().sort_values().index.tolist())
    fig_cont = px.bar(cont_stack, x="weight_kg", y="community", color="contamination_type",
                      orientation="h", category_orders={"community": order},
                      labels={"weight_kg": "混投量(kg)", "community": "",
                              "contamination_type": "混投类型"},
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_cont.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT, height=420,
                           margin=dict(l=10, r=10, t=20, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, title=""))
    st.plotly_chart(fig_cont, width="stretch")
    st.caption("展示对象:预警小区(无预警时取准确率最低的 8 个小区)")

with c2:
    cont_pie = cont.groupby("contamination_type")["weight_kg"].sum().reset_index()
    fig_pie = px.pie(cont_pie, values="weight_kg", names="contamination_type",
                     hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={"contamination_type": "混投类型", "weight_kg": "混投量(kg)"})
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT, height=420,
                          showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_pie, width="stretch")
    top_type = cont_pie.loc[cont_pie["weight_kg"].idxmax()]
    st.caption(f"最主要混投类型:**{top_type['contamination_type']}**,占全部混投的 {top_type['weight_kg'] / cont_pie['weight_kg'].sum():.1%}")

# ---------------- 原始数据 ----------------
with st.expander("📄 查看明细数据"):
    tab1, tab2, tab3 = st.tabs(["小区月度准确率", "每日记录(抽样)", "混投成分明细"])
    with tab1:
        pivot = monthly_comm.pivot(index="community", columns="month", values="acc")

        def _acc_bg(v):
            """准确率低→红、高→绿(替代 background_gradient,免装 matplotlib)"""
            if pd.isna(v):
                return ""
            t = min(max((v - 0.60) / 0.35, 0), 1)          # 0.60→0, 0.95→1
            if t < 0.5:                                     # 红→黄
                r, g, b = 244, int(67 + (235 - 67) * t * 2), int(54 + (59 - 54) * t * 2)
            else:                                           # 黄→绿
                u = (t - 0.5) * 2
                r, g, b = int(255 + (46 - 255) * u), int(235 + (125 - 235) * u), int(59 + (50 - 59) * u)
            return f"background-color: rgb({r},{g},{b})"

        st.dataframe(pivot.style.format("{:.1%}").map(_acc_bg), width="stretch")
    with tab2:
        st.dataframe(df.sample(min(500, len(df)), random_state=1).sort_values("date"),
                     width="stretch", hide_index=True)
    with tab3:
        st.dataframe(cont.sort_values(["month", "community"]), width="stretch", hide_index=True)
