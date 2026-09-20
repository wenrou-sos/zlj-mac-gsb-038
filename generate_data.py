# -*- coding: utf-8 -*-
"""
城市垃圾分类模拟数据生成器
生成三张 CSV:
  data/communities.csv        小区/站点基础信息
  data/daily_records.csv      每日四类垃圾分出量 + 分类正确/混投量 + 参与户数
  data/contamination.csv      混投成分分析(按小区×月)
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# ---------------- 街道 / 小区结构 ----------------
STREETS = {
    "朝阳街道": ["阳光花园", "翠湖小区", "金桥公寓", "文华苑"],
    "滨江街道": ["江畔人家", "望江楼小区", "滨水雅苑", "春风里", "江枫社区"],
    "青山街道": ["山水人家", "绿谷新村", "青竹苑", "松涛小区"],
    "文华街道": ["书香门第", "学府花园", "文昌里", "墨香苑"],
    "高新街道": ["科创家园", "智慧城小区", "未来之光", "星辰苑", "云谷社区"],
    "老城街道": ["永宁里", "和平新村", "鼓楼小区", "长安巷社区"],
}

# 问题小区设定:准确率持续下滑 / 长期低准确率(混投高发)
DECLINING = {"鼓楼小区", "永宁里", "江枫社区"}      # 逐月下滑
CHRONIC_LOW = {"长安巷社区", "和平新村"}             # 长期低位徘徊

START, END = "2026-01-01", "2026-09-19"
dates = pd.date_range(START, END, freq="D")
months = pd.period_range(START, END, freq="M")
month_idx = {m: i for i, m in enumerate(months)}     # 月份序号,用于趋势项

# ---------------- 小区基础信息 ----------------
communities = []
for street, comms in STREETS.items():
    for c in comms:
        households = int(rng.integers(400, 1600))
        communities.append({
            "community": c,
            "street": street,
            "total_households": households,
            "station_count": max(2, round(households / 220)),  # 定时定点投放站点数
        })
communities = pd.DataFrame(communities)

# 每个小区的隐含参数
params = {}
for _, row in communities.iterrows():
    c = row["community"]
    if c in DECLINING:
        base_acc = rng.uniform(0.84, 0.88)
        acc_trend = -rng.uniform(0.010, 0.016)   # 每月下滑
    elif c in CHRONIC_LOW:
        base_acc = rng.uniform(0.66, 0.71)
        acc_trend = rng.uniform(-0.002, 0.002)   # 基本无改善
    else:
        base_acc = rng.uniform(0.72, 0.88)
        acc_trend = rng.uniform(0.004, 0.012)    # 逐月提升
    params[c] = dict(
        base_acc=base_acc,
        acc_trend=acc_trend,
        base_part=rng.uniform(0.55, 0.85),
        part_trend=rng.uniform(0.001, 0.006),
    )

# ---------------- 每日投放记录 ----------------
rows = []
for _, row in communities.iterrows():
    c, street = row["community"], row["street"]
    hh = row["total_households"]
    p = params[c]
    for d in dates:
        m_i = month_idx[d.to_period("M")]
        weekend = d.dayofweek >= 5

        acc = (p["base_acc"] + p["acc_trend"] * m_i
               + (0.012 if weekend else 0)
               + rng.normal(0, 0.018))
        acc = float(np.clip(acc, 0.45, 0.985))

        part = (p["base_part"] + p["part_trend"] * m_i
                + (0.05 if weekend else 0)
                + rng.normal(0, 0.03))
        part = float(np.clip(part, 0.25, 0.98))

        participated = int(round(hh * part))
        wf = 1.15 if weekend else 1.0            # 周末垃圾量上浮
        kitchen = participated * 0.32 * wf * rng.lognormal(0, 0.06)
        recyclable = participated * 0.14 * wf * rng.lognormal(0, 0.08)
        hazardous = participated * 0.008 * rng.lognormal(0, 0.15)
        other = participated * 0.45 * wf * rng.lognormal(0, 0.06)
        total = kitchen + recyclable + hazardous + other

        rows.append({
            "date": d, "street": street, "community": c,
            "kitchen_kg": round(kitchen, 1),
            "recyclable_kg": round(recyclable, 1),
            "hazardous_kg": round(hazardous, 1),
            "other_kg": round(other, 1),
            "total_kg": round(total, 1),
            "correct_kg": round(total * acc, 1),
            "mixed_kg": round(total * (1 - acc), 1),
            "total_households": hh,
            "participated_households": participated,
        })
daily = pd.DataFrame(rows)

# ---------------- 混投成分分析(小区 × 月) ----------------
CONTAM_TYPES = [
    "厨余袋混入塑料瓶/塑料袋",
    "厨余袋混入纸巾渣土等其他垃圾",
    "厨余袋混入玻璃纸盒等可回收物",
    "可回收物桶混入厨余残渣污染",
    "其他垃圾桶混入废电池灯管等有害垃圾",
    "其他垃圾桶混入纸箱金属等可回收物",
]
# 各小区混投"偏好":有的主要是厨余袋混塑料,有的是其他垃圾桶混可回收
propensity = {}
for c in communities["community"]:
    alpha = rng.uniform(0.6, 1.6, size=len(CONTAM_TYPES))
    alpha[0] *= rng.uniform(1.5, 3.0)   # 厨余袋混塑料普遍偏高
    alpha[1] *= rng.uniform(1.2, 2.2)
    propensity[c] = alpha

contam_rows = []
daily["month"] = daily["date"].dt.to_period("M")
for (street, comm, month), g in daily.groupby(["street", "community", "month"]):
    mixed_total = g["mixed_kg"].sum()
    w = rng.dirichlet(propensity[comm])
    for t, wi in zip(CONTAM_TYPES, w):
        contam_rows.append({
            "month": str(month), "street": street, "community": comm,
            "contamination_type": t, "weight_kg": round(mixed_total * wi, 1),
        })
contamination = pd.DataFrame(contam_rows)
daily = daily.drop(columns=["month"])

communities.to_csv(OUT / "communities.csv", index=False, encoding="utf-8-sig")
daily.to_csv(OUT / "daily_records.csv", index=False, encoding="utf-8-sig")
contamination.to_csv(OUT / "contamination.csv", index=False, encoding="utf-8-sig")

print(f"communities.csv   {len(communities):>5} 行")
print(f"daily_records.csv {len(daily):>5} 行  ({daily['date'].min().date()} ~ {daily['date'].max().date()})")
print(f"contamination.csv {len(contamination):>5} 行")
print(f"整体分类准确率: {daily['correct_kg'].sum() / daily['total_kg'].sum():.1%}")
