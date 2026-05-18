"""
23_group_level_analysis.py
==========================
组级（Group-Level / Second-Level）fMRI分析。

做什么：
  1. 读取所有被试的 gain_zmap 和 loss_zmap
  2. 用 Nilearn 的 SecondLevelModel 做 one-sample t-test
     （问：跨被试，哪些脑区一致地被 gain/loss 金额调制？）
  3. 用 cluster-level 校正控制多重比较
  4. 可视化并保存结果

概念解释：
  - First-level: 每个被试内部，每个体素做一个回归（时间序列 ~ 任务事件）
  - Second-level: 跨被试，每个体素做一个统计检验
    （这个体素的效应在群体中是否显著不为零？）
  - 本质上就是对每个体素的z值做一个 one-sample t-test

运行方式：conda activate narps && python 23_group_level_analysis.py
前提：22_first_level_all_subjects.py 已跑完，first_level_results/ 中有z-map
"""

import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.reporting import get_clusters_table
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

results_dir = 'first_level_results'
group_dir = 'group_level_results'
os.makedirs(group_dir, exist_ok=True)

# ============================================================
# 1. 收集所有被试的 z-map
# ============================================================

print("=" * 60)
print("Group-Level Analysis")
print("=" * 60)

# 找到所有有gain和loss z-map的被试
gain_maps = sorted([
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith('_gain_zmap.nii.gz')
])
loss_maps = sorted([
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith('_loss_zmap.nii.gz')
])

# 确保gain和loss的被试一一对应
gain_subs = [os.path.basename(f).split('_')[0] for f in gain_maps]
loss_subs = [os.path.basename(f).split('_')[0] for f in loss_maps]
assert gain_subs == loss_subs, "Gain和Loss的被试列表不匹配！"

n_subjects = len(gain_maps)
print(f"\n找到 {n_subjects} 个被试的z-map:")
for s in gain_subs:
    print(f"  {s}")

if n_subjects < 3:
    print("\n⚠️  被试数少于3，统计检验不可靠。建议至少有5个以上被试。")
    print("继续运行，但请注意结果仅供参考。")

# ============================================================
# 2. 组级分析：One-Sample T-Test
# ============================================================
# SecondLevelModel 的 one-sample t-test 做的事情：
#   对每个体素，收集所有被试在该体素的z值，
#   检验这些z值的均值是否显著不为零。
#   如果是，说明这个脑区在群体水平上一致地被gain（或loss）调制。

# 设计矩阵：one-sample t-test 只需要一列全1的截距
# intercept = 1 表示我们检验均值是否不为零
design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})

print(f"\n设计矩阵（{n_subjects} × 1）：检验群体均值是否显著不为零")

# ---- GAIN 效应 ----
print("\n" + "─" * 40)
print("分析 GAIN 效应...")

# 创建second-level模型
second_level_gain = SecondLevelModel(smoothing_fwhm=None)
# smoothing_fwhm=None：不在组级再做平滑（first-level已经平滑过了）

# 拟合：输入是所有被试的z-map列表
second_level_gain.fit(gain_maps, design_matrix=design_matrix)

# 计算对比：intercept 的效应
z_map_gain_group = second_level_gain.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)
print("  Gain 组级z-map 计算完成")

# ---- LOSS 效应 ----
print("\n" + "─" * 40)
print("分析 LOSS 效应...")

second_level_loss = SecondLevelModel(smoothing_fwhm=None)
second_level_loss.fit(loss_maps, design_matrix=design_matrix)

z_map_loss_group = second_level_loss.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)
print("  Loss 组级z-map 计算完成")

# ============================================================
# 3. 多重比较校正（Cluster-Level Correction）
# ============================================================
# 为什么需要校正？
#   大脑约有 ~200,000 个体素，如果每个体素都用 p<0.05 做检验，
#   会有 ~10,000 个假阳性。
#
# Cluster-level correction 的逻辑：
#   Step 1: 用一个初始阈值（cluster_threshold，比如 p<0.001 即 z>3.1）
#           筛选出"可能显著"的体素
#   Step 2: 把相邻的显著体素连成"簇"（cluster）
#   Step 3: 检验每个簇的大小是否大到不可能是随机产生的
#   这利用了一个事实：真信号在空间上是连续的（相邻体素一起亮），
#   而噪声产生的假阳性通常是孤立的散点。

print("\n" + "─" * 40)
print("多重比较校正（Cluster-Level）...")

# cluster_threshold: 形成簇的初始z值阈值
# 对于小样本（n=9），我们用稍微宽松一点的阈值
# 标准做法是 z>3.1（对应 p<0.001），但小样本可能太严格
# 这里用 z>2.3（对应 p~0.01），这在小样本探索性分析中是合理的
cluster_threshold = 2.3

# 对 gain z-map 做 cluster 校正
# threshold_stats_img 返回：
#   thresholded_map: 校正后的z-map（不显著的体素被设为0）
#   threshold_value: 实际使用的阈值
thresholded_gain, threshold_gain = threshold_stats_img(
    z_map_gain_group,
    alpha=0.05,                    # 簇级别的显著性水平
    height_control='fpr',          # fpr = 用固定阈值形成簇
    cluster_threshold=10,          # 最小簇大小（体素数），过滤掉太小的簇
    # 注意：alpha=0.05 + height_control='fpr' 的组合
    # 意味着先用 p<0.05 未校正阈值形成簇，然后用最小簇大小过滤
    # 对于更严格的分析，可以用 height_control='fdr' 或 'bonferroni'
)

thresholded_loss, threshold_loss = threshold_stats_img(
    z_map_loss_group,
    alpha=0.05,
    height_control='fpr',
    cluster_threshold=10,
)

print(f"  Gain 阈值: z = {threshold_gain:.2f}")
print(f"  Loss 阈值: z = {threshold_loss:.2f}")

# ============================================================
# 4. 提取显著簇的信息
# ============================================================

print("\n" + "─" * 40)
print("显著簇（Clusters）报告")

# get_clusters_table 会列出每个显著簇的：
#   - 峰值坐标 (MNI x, y, z)
#   - 簇大小（体素数）
#   - 峰值z值
#   - 对应的脑区名称（基于MNI坐标自动标注）

print("\n===== GAIN 效应（正值 = gain越大，激活越强）=====")
try:
    gain_table = get_clusters_table(
        z_map_gain_group,
        stat_threshold=cluster_threshold,
        min_distance=8              # 同一簇内峰值之间的最小距离（mm）
    )
    if len(gain_table) > 0:
        print(gain_table.to_string())
        gain_table.to_csv(os.path.join(group_dir, 'gain_clusters.csv'), index=False)
    else:
        print("  没有找到显著簇")
except Exception as e:
    print(f"  簇提取出错: {e}")

print("\n===== LOSS 效应（正值 = loss越大，激活越强）=====")
try:
    loss_table = get_clusters_table(
        z_map_loss_group,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(loss_table) > 0:
        print(loss_table.to_string())
        loss_table.to_csv(os.path.join(group_dir, 'loss_clusters.csv'), index=False)
    else:
        print("  没有找到显著簇")
except Exception as e:
    print(f"  簇提取出错: {e}")

# ============================================================
# 5. 可视化
# ============================================================

print("\n" + "─" * 40)
print("生成可视化...")

# --- 图1：未校正的组级z-map（展示整体pattern）---
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    z_map_gain_group,
    threshold=cluster_threshold,
    title=f'Group GAIN effect (n={n_subjects}, z>{cluster_threshold})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[0],
)

plotting.plot_stat_map(
    z_map_loss_group,
    threshold=cluster_threshold,
    title=f'Group LOSS effect (n={n_subjects}, z>{cluster_threshold})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[1],
)

plt.tight_layout()
fig1.savefig(os.path.join(group_dir, 'group_gain_loss_zmaps.png'), dpi=150)
print("  保存: group_gain_loss_zmaps.png")

# --- 图2：校正后的z-map ---
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    thresholded_gain,
    threshold=0.1,      # 已经校正过，只显示幸存的体素
    title=f'Group GAIN (cluster-corrected, n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes2[0],
)

plotting.plot_stat_map(
    thresholded_loss,
    threshold=0.1,
    title=f'Group LOSS (cluster-corrected, n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes2[1],
)

plt.tight_layout()
fig2.savefig(os.path.join(group_dir, 'group_gain_loss_corrected.png'), dpi=150)
print("  保存: group_gain_loss_corrected.png")

# --- 图3：玻璃脑（glass brain）视图 ---
# 玻璃脑把3D大脑"压扁"到三个平面，能看到激活的整体空间分布
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

plotting.plot_glass_brain(
    z_map_gain_group,
    threshold=cluster_threshold,
    title=f'Gain (z>{cluster_threshold})',
    axes=axes3[0],
    display_mode='lyrz',    # 左、右、上、前 四个视角
    colorbar=True,
)

plotting.plot_glass_brain(
    z_map_loss_group,
    threshold=cluster_threshold,
    title=f'Loss (z>{cluster_threshold})',
    axes=axes3[1],
    display_mode='lyrz',
    colorbar=True,
)

plt.tight_layout()
fig3.savefig(os.path.join(group_dir, 'group_glass_brain.png'), dpi=150)
print("  保存: group_glass_brain.png")

# ============================================================
# 6. 保存组级z-map（供后续分析使用）
# ============================================================

z_map_gain_group.to_filename(os.path.join(group_dir, 'group_gain_zmap.nii.gz'))
z_map_loss_group.to_filename(os.path.join(group_dir, 'group_loss_zmap.nii.gz'))
thresholded_gain.to_filename(os.path.join(group_dir, 'group_gain_corrected.nii.gz'))
thresholded_loss.to_filename(os.path.join(group_dir, 'group_loss_corrected.nii.gz'))
print("\n组级z-map已保存")

# ============================================================
# 总结
# ============================================================

print(f"\n{'=' * 60}")
print("Group-Level Analysis 完成！")
print(f"{'=' * 60}")
print(f"被试数: {n_subjects}")
print(f"簇形成阈值: z > {cluster_threshold}")
print(f"结果保存在: {group_dir}/")
print(f"\n下一步:")
print(f"  1. 查看 group_gain_loss_zmaps.png — gain和loss的组级效应")
print(f"  2. 查看 group_gain_loss_corrected.png — 校正后的显著区域")
print(f"  3. 查看 gain_clusters.csv / loss_clusters.csv — 显著簇的详细信息")
print(f"  4. 运行 24_fatigue_neural_contrast.py — 前半vs后半的神经疲劳对比")