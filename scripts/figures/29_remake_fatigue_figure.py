"""
29_remake_fatigue_figure.py
===========================
重新绘制 Part D 的脑图，改进可视化效果。
"""

import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

fatigue_dir = 'fatigue_neural_results'
output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

# 读取组级差异z-map
z_gain = image.load_img(os.path.join(fatigue_dir, 'group_gain_fatigue_zmap.nii.gz'))
z_loss = image.load_img(os.path.join(fatigue_dir, 'group_loss_fatigue_zmap.nii.gz'))

# ============================================================
# 图1：改进的 gain vs loss time-on-task对比
# ============================================================

# 选择更好的切面：包含vmPFC(-12,36,-13)的位置
# z=-13 正好穿过vmPFC簇，z=6 穿过insula，z=36 穿过dACC
cut_coords_z = [-14, -4, 6, 14, 22, 32, 42, 54]

fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    z_gain,
    threshold=2.0,  # 稍低的阈值，显示更多效应
    title='GAIN: Time-on-Task Effect (Late − Early, n=41)',
    display_mode='z',
    cut_coords=cut_coords_z,
    axes=axes1[0],
    colorbar=True,
    annotate=True,
)

plotting.plot_stat_map(
    z_loss,
    threshold=2.0,
    title='LOSS: Time-on-Task Effect (Late − Early, n=41)',
    display_mode='z',
    cut_coords=cut_coords_z,
    axes=axes1[1],
    colorbar=True,
    annotate=True,
)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig3_time_on_task_neural.png'), dpi=200)
print("保存: fig3_time_on_task_neural.png")

# ============================================================
# 图2：vmPFC聚焦图 — 用sagittal和coronal切面展示vmPFC簇
# ============================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

# Sagittal (x=-12，穿过vmPFC簇的峰值)
plotting.plot_stat_map(
    z_loss,
    threshold=2.0,
    display_mode='x',
    cut_coords=[-12],
    axes=axes2[0],
    title='Sagittal (x=-12)',
    colorbar=True,
    annotate=True,
)

# Coronal (y=36，穿过vmPFC簇)
plotting.plot_stat_map(
    z_loss,
    threshold=2.0,
    display_mode='y',
    cut_coords=[36],
    axes=axes2[1],
    title='Coronal (y=36)',
    colorbar=True,
    annotate=True,
)

# Axial (z=-13，穿过vmPFC簇)
plotting.plot_stat_map(
    z_loss,
    threshold=2.0,
    display_mode='z',
    cut_coords=[-13],
    axes=axes2[2],
    title='Axial (z=-13)',
    colorbar=True,
    annotate=True,
)

plt.suptitle('LOSS Time-on-Task Effect: vmPFC Detail (Late − Early, n=41)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'fig3b_vmpfc_detail.png'), dpi=200, bbox_inches='tight')
print("保存: fig3b_vmpfc_detail.png")

# ============================================================
# 图3：三行对比 — Part C 整体效应 vs Part D time-on-task变化
# ============================================================

# 读取Part C的整体loss效应
group_loss_overall = os.path.join('group_level_results', 'group_loss_zmap.nii.gz')
group_gain_overall = os.path.join('group_level_results', 'group_gain_zmap.nii.gz')

if os.path.exists(group_loss_overall) and os.path.exists(group_gain_overall):
    z_loss_overall = image.load_img(group_loss_overall)
    z_gain_overall = image.load_img(group_gain_overall)

    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 12))

    plotting.plot_stat_map(
        z_loss_overall,
        threshold=2.3,
        title='A. Overall LOSS Effect (all runs, n=41)',
        display_mode='z',
        cut_coords=cut_coords_z,
        axes=axes3[0],
        colorbar=True,
    )

    plotting.plot_stat_map(
        z_loss,
        threshold=2.0,
        title='B. LOSS Time-on-Task Change (Late − Early, n=41)',
        display_mode='z',
        cut_coords=cut_coords_z,
        axes=axes3[1],
        colorbar=True,
    )

    plotting.plot_stat_map(
        z_gain,
        threshold=2.0,
        title='C. GAIN Time-on-Task Change (Late − Early, n=41)',
        display_mode='z',
        cut_coords=cut_coords_z,
        axes=axes3[2],
        colorbar=True,
    )

    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'fig3c_comprehensive.png'), dpi=200)
    print("保存: fig3c_comprehensive.png")

# ============================================================
# 也重画Part C的gain vs loss（用统一格式）
# ============================================================

if os.path.exists(group_loss_overall) and os.path.exists(group_gain_overall):
    fig4, axes4 = plt.subplots(2, 1, figsize=(14, 8))

    plotting.plot_stat_map(
        z_gain_overall,
        threshold=2.3,
        title='GAIN Parametric Effect (n=41, z>2.3)',
        display_mode='z',
        cut_coords=cut_coords_z,
        axes=axes4[0],
        colorbar=True,
    )

    plotting.plot_stat_map(
        z_loss_overall,
        threshold=2.3,
        title='LOSS Parametric Effect (n=41, z>2.3)',
        display_mode='z',
        cut_coords=cut_coords_z,
        axes=axes4[1],
        colorbar=True,
    )

    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'fig2_gain_loss_overall.png'), dpi=200)
    print("保存: fig2_gain_loss_overall.png")

print("\n所有论文图片保存在 paper_figures/ 目录")
print("可用于上传到Overleaf")