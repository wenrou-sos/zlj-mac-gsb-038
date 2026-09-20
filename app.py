"""城市垃圾分类成效分析看板

运行: streamlit run app.py
数据: 先执行 python generate_data.py 生成 data/*.csv
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="城市垃圾分类成效分析看板", page_icon="♻️", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
CATS = ["厨余垃圾kg", "可回收物kg", "有害垃圾kg", "其他垃圾kg"]
CAT_SHORT = ["厨余垃圾", "可回收物", "有害垃圾", "其他垃圾"]
CAT_COLORS = {"厨余垃圾": "#2ca02c", "可回收物": "#1f77b4", "有害垃圾": "#d62728", "其他垃圾": "#7f7f7f"}
TEMPLATE = "plotly_white"


@st.cache_data
def load_data():
    daily = pd.read_csv(DATA_DIR / "daily_classification.csv", parse_dates=["日期"])
    misthrow = pd.read_csv(DATA_DIR / "misthrow_detail.csv", parse_dates=["日期"])
    return daily, misthrow


daily, misthrow = load_data()

# ---------------- 侧边栏筛选 ----------------
st.sidebar.header("筛选条件")
min_d, max_d = daily["日期"].min().date(), daily["日期"].max().date()
date_range = st.sidebar.date_input("日期范围", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_d, end_d = (date_range if len(date_range) == 2 else (date_range[0], date_range[0]))

streets_all = sorted(daily["街道"].unique())
streets = st.sidebar.multiselect("街道", streets_all, default=streets_all)
thr = st.sidebar.slider("混投预警阈值（混投率 ≥ 即标红）", 0.05, 0.30, 0.12, 0.01, format="%0.0f%%")

mask = (daily["日期"].dt.date >= start_d) & (daily["日期"].dt.date <= end_d) & daily["街道"].isin(streets)
f = daily.loc[mask].copy()
mf = misthrow.loc[
    (misthrow["日期"].dt.date >= start_d) & (misthrow["日期"].dt.date <= end_d) & misthrow["街道"].isin(streets)
].copy()

if f.empty:
    st.warning("当前筛选条件下没有数据，请调整日期范围或街道。")
    st.stop()

n_days = max((pd.Timestamp(end_d) - pd.Timestamp(start_d)).days + 1, 1)

# ---------------- 标题 & KPI ----------------
st.title("♻️ 城市垃圾分类成效分析看板")
st.caption(f"数据区间：{start_d} ~ {end_d}（{n_days} 天）｜覆盖 {f['街道'].nunique()} 个街道、{f['社区'].nunique()} 个社区｜数据为模拟演示数据")

total_t = f[CATS].sum().sum() / 1000
acc_all = f["正确投放次数"].sum() / (f["正确投放次数"].sum() + f["混投次数"].sum())
part_all = f["参与户数"].sum() / f["总户数"].sum()
mt_total = int(f["混投次数"].sum())

f["月份"] = f["日期"].dt.strftime("%Y-%m")
m_all = f.groupby("月份").agg(正确=("正确投放次数", "sum"), 混投=("混投次数", "sum"))
m_all["acc"] = m_all["正确"] / (m_all["正确"] + m_all["混投"])
acc_delta_pp = (m_all["acc"].iloc[-1] - m_all["acc"].iloc[0]) * 100 if len(m_all) > 1 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("累计分出垃圾总量", f"{total_t:,.1f} 吨")
k2.metric("总体分类准确率", f"{acc_all:.1%}", delta=f"{acc_delta_pp:+.1f} pp（末月 vs 首月）")
k3.metric("定时定点投放参与率", f"{part_all:.1%}")
k4.metric("混投事件总数", f"{mt_total:,} 起")

tab1, tab2, tab3, tab4 = st.tabs(["🏆 街道排名与分出量", "📈 月度趋势", "🏘️ 居民参与率", "🚨 混投分析"])

# ================= Tab 1 街道排名 =================
with tab1:
    st.subheader("各街道分类成效排名（按分类准确率）")
    g = f.groupby("街道", as_index=False).agg(正确=("正确投放次数", "sum"), 混投=("混投次数", "sum"))
    g["分类准确率"] = g["正确"] / (g["正确"] + g["混投"])
    g = g.sort_values("分类准确率")
    fig = px.bar(
        g, y="街道", x="分类准确率", orientation="h",
        color="分类准确率", color_continuous_scale="RdYlGn",
        text=g["分类准确率"].map("{:.1%}".format), range_x=[0, 1.05],
    )
    fig.add_vline(x=acc_all, line_dash="dash", line_color="#333",
                  annotation_text=f"全市平均 {acc_all:.1%}", annotation_position="top right")
    fig.update_traces(textposition="outside")
    fig.update_layout(template=TEMPLATE, coloraxis_showscale=False, height=380,
                      xaxis_title="分类准确率（正确投放次数 / 总投放次数）", yaxis_title="")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    st.subheader("各街道四类垃圾分出量结构")
    w = f.groupby("街道")[CATS].sum().rename(columns=dict(zip(CATS, CAT_SHORT))) / 1000
    fig2 = px.bar(
        w.reset_index(), x="街道", y=CAT_SHORT, barmode="stack",
        color_discrete_map=CAT_COLORS, labels={"value": "分出量（吨）", "variable": "品类"},
    )
    fig2.update_layout(template=TEMPLATE, height=420, legend_title_text="品类")
    st.plotly_chart(fig2, width="stretch")

# ================= Tab 2 月度趋势 =================
with tab2:
    st.subheader("分类准确率月度趋势")
    m = f.groupby(["月份", "街道"], as_index=False).agg(正确=("正确投放次数", "sum"), 混投=("混投次数", "sum"))
    m["分类准确率"] = m["正确"] / (m["正确"] + m["混投"])
    fig3 = px.line(m, x="月份", y="分类准确率", color="街道", markers=True)
    mo = f.groupby("月份").agg(正确=("正确投放次数", "sum"), 混投=("混投次数", "sum"))
    mo["acc"] = mo["正确"] / (mo["正确"] + mo["混投"])
    fig3.add_scatter(x=mo.index, y=mo["acc"], mode="lines+markers", name="全市平均",
                     line=dict(color="black", width=3, dash="dash"))
    fig3.update_yaxes(tickformat=".1%")
    fig3.update_layout(template=TEMPLATE, height=450, yaxis_title="分类准确率", xaxis_title="")
    st.plotly_chart(fig3, width="stretch")

    if len(mo) > 1:
        pivot = m.pivot(index="月份", columns="街道", values="分类准确率")
        delta = (pivot.iloc[-1] - pivot.iloc[0]).sort_values(ascending=False)
        rising, falling = delta[delta > 0], delta[delta <= 0]
        c1, c2 = st.columns(2)
        with c1:
            st.success("📈 **准确率上升的街道**\n\n" + "\n\n".join(
                f"- {s}：{d * 100:+.1f} pp" for s, d in rising.items()))
        with c2:
            if len(falling):
                st.error("📉 **准确率下降的街道（需重点关注）**\n\n" + "\n\n".join(
                    f"- {s}：{d * 100:+.1f} pp" for s, d in falling.items()))
            else:
                st.info("所有街道准确率均在上升 🎉")

# ================= Tab 3 参与率 =================
with tab3:
    st.subheader("定时定点投放站点 · 居民参与率（日均投递户数 / 总户数）")
    p = f.groupby(["街道", "社区"], as_index=False).agg(参与=("参与户数", "sum"), 基数=("总户数", "sum"))
    p["参与率"] = p["参与"] / p["基数"]
    p = p.sort_values("参与率")
    fig4 = px.bar(
        p, y="社区", x="参与率", orientation="h", color="街道",
        text=p["参与率"].map("{:.1%}".format), range_x=[0, 1.05],
    )
    fig4.add_vline(x=part_all, line_dash="dash", line_color="#333",
                   annotation_text=f"全市平均 {part_all:.1%}", annotation_position="top right")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(template=TEMPLATE, height=640, yaxis_title="", xaxis_title="参与率")
    fig4.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig4, width="stretch")

    st.subheader("参与率月度趋势")
    mp = f.groupby(["月份", "街道"], as_index=False).agg(参与=("参与户数", "sum"), 基数=("总户数", "sum"))
    mp["参与率"] = mp["参与"] / mp["基数"]
    fig5 = px.line(mp, x="月份", y="参与率", color="街道", markers=True)
    fig5.update_yaxes(tickformat=".1%")
    fig5.update_layout(template=TEMPLATE, height=400, yaxis_title="参与率", xaxis_title="")
    st.plotly_chart(fig5, width="stretch")

# ================= Tab 4 混投分析 =================
with tab4:
    c = f.groupby(["街道", "社区"], as_index=False).agg(正确=("正确投放次数", "sum"), 混投=("混投次数", "sum"))
    c["分类准确率"] = c["正确"] / (c["正确"] + c["混投"])
    c["混投率"] = 1 - c["分类准确率"]
    c["日均混投次数"] = c["混投"] / n_days
    top_type = (mf.groupby(["社区", "混投类型"], as_index=False)["次数"].sum()
                .sort_values("次数", ascending=False)
                .drop_duplicates("社区")[["社区", "混投类型"]]
                .rename(columns={"混投类型": "主要混投品类"}))
    c = c.merge(top_type, on="社区", how="left")
    bad = c[c["混投率"] >= thr].sort_values("混投率", ascending=False).reset_index(drop=True)

    st.subheader(f"🚨 混投高发小区（混投率 ≥ {thr:.0%}，共 {len(bad)} 个，已标红）")
    if bad.empty:
        st.success("当前阈值下没有混投高发小区 🎉")
    else:
        show = bad[["街道", "社区", "分类准确率", "混投率", "日均混投次数", "主要混投品类"]]
        styled = (show.style
                  .format({"分类准确率": "{:.1%}", "混投率": "{:.1%}", "日均混投次数": "{:.1f}"})
                  .map(lambda _: "background-color: rgba(255, 70, 70, 0.18)"))
        st.dataframe(styled, width="stretch", hide_index=True)

        worst = c.sort_values("混投率", ascending=False).head(10).iloc[::-1]
        fig6 = go.Figure(go.Bar(
            y=worst["社区"], x=worst["混投率"], orientation="h",
            marker_color=["#d62728" if r >= thr else "#ff9896" for r in worst["混投率"]],
            text=worst["混投率"].map("{:.1%}".format), textposition="outside",
        ))
        fig6.update_layout(template=TEMPLATE, height=420, title="混投率最高的 10 个社区（深红 = 超预警阈值）",
                           xaxis_title="混投率", yaxis_title="", xaxis=dict(tickformat=".0%", range=[0, worst["混投率"].max() * 1.2]))
        st.plotly_chart(fig6, width="stretch")

    st.subheader("混投品类构成分析")
    overall_top = mf.groupby("混投类型")["次数"].sum().sort_values(ascending=False)
    if not overall_top.empty:
        st.warning(f"全市混投最主要的品类是 **「{overall_top.index[0]}」**，占全部混投事件的 "
                   f"{overall_top.iloc[0] / overall_top.sum():.1%}。")

    col_l, col_r = st.columns(2)
    with col_l:
        focus_list = bad["社区"].tolist() if not bad.empty else c.sort_values("混投率", ascending=False)["社区"].tolist()
        sel = st.selectbox("选择社区查看混投构成", focus_list)
        d = mf[mf["社区"] == sel].groupby("混投类型")["次数"].sum()
        fig7 = px.pie(values=d.values, names=d.index, hole=0.45, title=f"「{sel}」混投类型构成")
        fig7.update_layout(template=TEMPLATE, height=430)
        st.plotly_chart(fig7, width="stretch")
    with col_r:
        focus8 = (bad["社区"].head(8) if not bad.empty else
                  c.sort_values("混投率", ascending=False)["社区"].head(8))
        d8 = mf[mf["社区"].isin(focus8)].groupby(["社区", "混投类型"], as_index=False)["次数"].sum()
        d8p = d8.pivot(index="社区", columns="混投类型", values="次数").fillna(0)
        share = (d8p.div(d8p.sum(axis=1), axis=0).reset_index()
                 .melt(id_vars="社区", var_name="混投类型", value_name="占比"))
        fig8 = px.bar(share, x="社区", y="占比", color="混投类型", barmode="stack",
                      title="混投高发小区 · 混投类型占比对比")
        fig8.update_layout(template=TEMPLATE, height=430, yaxis_title="占比")
        fig8.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig8, width="stretch")

# ---------------- 数据下载 ----------------
st.sidebar.divider()
st.sidebar.download_button("⬇️ 下载筛选后的明细数据 (CSV)",
                           f.drop(columns=["月份"]).to_csv(index=False).encode("utf-8-sig"),
                           file_name="classification_filtered.csv", mime="text/csv")
