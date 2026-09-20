"""生成模拟的城市垃圾分类数据（CSV）.

产出两个文件:
- data/daily_classification.csv : 各街道/社区每日四类垃圾分出量、投放正确/混投次数、参与户数
- data/misthrow_detail.csv     : 各社区每日混投明细（混投类型 × 次数）
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260920)

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

START, END = "2026-01-01", "2026-08-31"

STREETS = {
    "高新街道": ["科创园小区", "智慧城", "未来海岸", "云谷公寓"],
    "文华街道": ["书香门第", "文华苑", "学府家园", "梧桐里"],
    "朝阳街道": ["阳光花园", "朝阳新村", "翠湖天地", "金桂苑"],
    "滨江街道": ["江畔人家", "滨江豪庭", "望江公寓", "清风雅苑"],
    "青山街道": ["青山绿庭", "竹韵山庄", "松涛苑", "梅岭新村"],
    "老城街道": ["和平里", "鼓楼新村", "南锣社区", "文庙社区"],
}

# 街道画像: acc=基准分类准确率, slope=准确率月度变化, part=基准参与率, pslope=参与率月度变化
STREET_PROFILE = {
    "高新街道": dict(acc=0.90, slope=+0.008, part=0.85, pslope=+0.004),
    "文华街道": dict(acc=0.88, slope=+0.010, part=0.80, pslope=+0.005),
    "朝阳街道": dict(acc=0.86, slope=+0.006, part=0.78, pslope=+0.003),
    "滨江街道": dict(acc=0.84, slope=+0.012, part=0.74, pslope=+0.006),
    "青山街道": dict(acc=0.82, slope=-0.004, part=0.70, pslope=-0.002),
    "老城街道": dict(acc=0.78, slope=-0.006, part=0.66, pslope=-0.003),
}

# 混投重点小区（在街道基准上额外下调）——这些会成为"标红"对象
COMMUNITY_ADJ = {
    "南锣社区": dict(acc=-0.09, part=-0.10),
    "梅岭新村": dict(acc=-0.07, part=-0.06),
    "鼓楼新村": dict(acc=-0.05, part=-0.04),
    "松涛苑":   dict(acc=-0.05, part=-0.02),
    "文庙社区": dict(acc=-0.04, part=-0.03),
}

MISTHROW_TYPES = [
    "厨余袋混入塑料瓶等可回收物",
    "厨余袋混入纸巾/塑料袋等其他垃圾",
    "其他垃圾中混入厨余垃圾",
    "可回收物中混入厨余残渣",
    "有害垃圾混入其他垃圾",
    "可回收物误投其他垃圾桶",
]
# 普通小区 vs 混投重点小区 的混投类型分布倾向
ALPHA_BASE = [3.0, 2.5, 2.0, 1.5, 0.8, 1.2]
ALPHA_BAD = [5.0, 4.0, 2.0, 1.2, 0.6, 1.0]  # 厨余袋混投问题更突出

dates = pd.date_range(START, END)
n_days = len(dates)
month_idx = dates.month.values - 1          # 0..7
weekend = (dates.dayofweek >= 5).astype(float)

daily_frames, detail_frames = [], []

for street, communities in STREETS.items():
    prof = STREET_PROFILE[street]
    for comm in communities:
        households = int(rng.integers(350, 1400))
        adj = COMMUNITY_ADJ.get(comm, {})
        acc0 = prof["acc"] + rng.normal(0, 0.015) + adj.get("acc", 0)
        part0 = np.clip(prof["part"] + rng.normal(0, 0.04) + adj.get("part", 0), 0.30, 0.97)
        slope = prof["slope"] + rng.normal(0, 0.001)

        # 参与率: 基准 + 周末上浮 + 月度趋势 + 噪声
        p_part = np.clip(
            part0 + 0.06 * weekend + prof["pslope"] * month_idx + rng.normal(0, 0.025, n_days),
            0.05, 0.99,
        )
        participated = rng.binomial(households, p_part)

        # 分类准确率: 基准 + 月度趋势 + 噪声
        acc = np.clip(acc0 + slope * month_idx + rng.normal(0, 0.02, n_days), 0.40, 0.995)
        misthrow = rng.binomial(participated, 1 - acc)
        correct = participated - misthrow

        # 四类垃圾分出量（kg），以参与户数为驱动
        kitchen = participated * 0.55 * rng.lognormal(0, 0.08, n_days)
        recyclable = participated * 0.18 * rng.lognormal(0, 0.10, n_days)
        hazardous = participated * 0.008 * rng.lognormal(0, 0.15, n_days)
        other = participated * 0.42 * rng.lognormal(0, 0.08, n_days)

        daily_frames.append(pd.DataFrame({
            "日期": dates,
            "街道": street,
            "社区": comm,
            "总户数": households,
            "参与户数": participated,
            "厨余垃圾kg": kitchen.round(1),
            "可回收物kg": recyclable.round(1),
            "有害垃圾kg": hazardous.round(1),
            "其他垃圾kg": other.round(1),
            "正确投放次数": correct,
            "混投次数": misthrow,
        }))

        # 混投明细: 把每日混投次数按社区画像分配到各混投类型
        alpha = ALPHA_BAD if comm in COMMUNITY_ADJ else ALPHA_BASE
        weights = rng.dirichlet(alpha)
        counts = np.array([rng.multinomial(m, weights) for m in misthrow])
        d = pd.DataFrame(counts, columns=MISTHROW_TYPES)
        d.insert(0, "日期", dates)
        d["街道"], d["社区"] = street, comm
        d = d.melt(id_vars=["日期", "街道", "社区"], var_name="混投类型", value_name="次数")
        detail_frames.append(d[d["次数"] > 0])

daily = pd.concat(daily_frames, ignore_index=True)
detail = pd.concat(detail_frames, ignore_index=True)

daily.to_csv(OUT / "daily_classification.csv", index=False, encoding="utf-8-sig")
detail.to_csv(OUT / "misthrow_detail.csv", index=False, encoding="utf-8-sig")

# 快速校验输出
acc_all = daily["正确投放次数"].sum() / (daily["正确投放次数"].sum() + daily["混投次数"].sum())
print(f"日期范围: {START} ~ {END}  ({n_days} 天)")
print(f"日数据: {len(daily)} 行 | 混投明细: {len(detail)} 行")
print(f"全市总体分类准确率: {acc_all:.1%}")
print(f"全市总体参与率: {daily['参与户数'].sum() / daily['总户数'].sum():.1%}")
g = daily.groupby(["街道", "社区"])[["正确投放次数", "混投次数"]].sum()
worst = (g["混投次数"] / (g["正确投放次数"] + g["混投次数"])).sort_values(ascending=False).head(5)
print("\n混投率最高的 5 个社区:")
print((worst * 100).round(1).astype(str) + "%")
