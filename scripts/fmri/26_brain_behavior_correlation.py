"""
26_brain_behavior_correlation.py
================================
脑-行为关联分析：个体差异层面的证据。

核心问题：
  行为上λ变化大的人，大脑里loss敏感区域的激活变化也大吗？

做什么：
  1. 从行为数据中提取每个被试的 Δλ = late_λ - early_λ
  2. 从fMRI差异图中提取每个被试在关键脑区（如vmPFC）的 Δactivation
  3. 计算跨被试的 Δλ 与 Δactivation 的相关性
  4. 还做一个全脑的体素级关联（哪些体素的变化和λ变化最相关？）

为什么这很重要？
  Part D 展示了两个平行的时间效应：行为上λ↑，神经上loss区域激活↑。
  但这只是"同时发生"——可能是巧合。
  如果在个体层面，λ变化大的人脑变化也大，
  就建立了更强的联系：行为变化和神经变化不只是同时发生，而是共变。

运行方式：conda activate narps && python 26_brain_behavior_correlation.py
前提：
  - runwise_parameters_fixed.csv（行为参数，已有）
  - fatigue_neural_results/ 中的单被试差异图（24脚本的输出）
"""

import pandas as pd
import numpy as np
from scipy import stats
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import get_clusters_table
from nilearn import plotting, maskers, datasets, image
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

fatigue_dir = 'fatigue_neural_results'
output_dir = 'brain_behavior_results'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. 提取每个被试的行为 Δλ
# ============================================================

print("=" * 60)
print("脑-行为关联分析")
print("=" * 60)

# 读取run-wise MLE参数
runwise = pd.read_csv('runwise_parameters_fixed.csv')
print(f"\n行为数据: {runwise.shape[0]} 行")
print(f"列名: {runwise.columns.tolist()}")

# 计算每个被试的 early λ（Run 1-2 均值）和 late λ（Run 3-4 均值）
# runwise_parameters_fixed.csv 中应该有 subject, run, lambda 列
# 先确认列名
print(f"\n前几行:")
print(runwise.head())

# 识别列名（可能是 lambda, lam, loss_aversion 等）
lambda_col = None
for candidate in ['lambda', 'lam', 'loss_aversion', 'λ']:
    if candidate in runwise.columns:
        lambda_col = candidate
        break

if lambda_col is None:
    # 尝试不区分大小写
    for col in runwise.columns:
        if 'lam' in col.lower() or 'loss' in col.lower():
            lambda_col = col
            break

if lambda_col is None:
    print("\n⚠️  找不到lambda列！请检查 runwise_parameters_fixed.csv 的列名。")
    print(f"可用列: {runwise.columns.tolist()}")
    print("脚本将尝试使用第一个看起来像参数的列...")
    # 作为fallback，显示所有列让用户判断
    import sys

    sys.exit(1)

print(f"\n使用 '{lambda_col}' 列作为 loss aversion 参数")

# 识别 subject 和 run 列
subj_col = None
run_col = None
for candidate in ['subject', 'sub', 'subject_id', 'subj']:
    if candidate in runwise.columns:
        subj_col = candidate
        break
for candidate in ['run', 'run_number', 'run_id']:
    if candidate in runwise.columns:
        run_col = candidate
        break

if subj_col is None or run_col is None:
    print(f"⚠️  找不到subject或run列。可用列: {runwise.columns.tolist()}")
    import sys

    sys.exit(1)

print(f"Subject列: '{subj_col}', Run列: '{run_col}'")

# 计算 Δλ = mean(Run 3-4) - mean(Run 1-2)
delta_lambda = {}
for subj in runwise[subj_col].unique():
    subj_data = runwise[runwise[subj_col] == subj]
    runs = subj_data[run_col].values

    early_runs = subj_data[subj_data[run_col].isin([1, 2])][lambda_col].values
    late_runs = subj_data[subj_data[run_col].isin([3, 4])][lambda_col].values

    if len(early_runs) >= 1 and len(late_runs) >= 1:
        delta_lambda[subj] = np.mean(late_runs) - np.mean(early_runs)

print(f"\n计算了 {len(delta_lambda)} 个被试的 Δλ")
print(f"Δλ 均值: {np.mean(list(delta_lambda.values())):.3f}")
print(f"Δλ 标准差: {np.std(list(delta_lambda.values())):.3f}")

# ============================================================
# 2. 匹配有fMRI差异图的被试
# ============================================================

# 找到同时有行为数据和fMRI差异图的被试
available_subs = []
fmri_maps = []
lambda_values = []

for subj_id, dlam in delta_lambda.items():
    # 构造被试编号字符串（确保格式为 sub-XXX）
    if isinstance(subj_id, (int, float)):
        subj_str = f'sub-{int(subj_id):03d}'
    elif isinstance(subj_id, str) and subj_id.startswith('sub-'):
        subj_str = subj_id
    else:
        subj_str = f'sub-{int(subj_id):03d}'

    # 检查是否有对应的fMRI差异图
    loss_diff_path = os.path.join(fatigue_dir, f'{subj_str}_loss_late_minus_early.nii.gz')
    if os.path.exists(loss_diff_path):
        available_subs.append(subj_str)
        fmri_maps.append(loss_diff_path)
        lambda_values.append(dlam)

n_matched = len(available_subs)
print(f"\n同时有行为和fMRI差异数据的被试: {n_matched}")
for s, dl in zip(available_subs, lambda_values):
    print(f"  {s}: Δλ = {dl:+.3f}")

if n_matched < 5:
    print("\n⚠️  匹配的被试太少（<5），相关分析不可靠。")
    print("请先确保 22 和 24 脚本已对所有被试跑完。")

# ============================================================
# 3. ROI分析：vmPFC中的脑-行为关联
# ============================================================

print(f"\n{'─' * 40}")
print("ROI分析：vmPFC")

# 定义vmPFC的球形ROI
# vmPFC的典型MNI坐标：约 (0, 40, -10) 到 (0, 30, -20)
# 我们用一个以 (0, 34, -16) 为中心、半径10mm的球
# 这个坐标接近Part D中最大簇的峰值
from nilearn.maskers import NiftiSpheresMasker

vmpfc_coords = [(0, 34, -16)]  # vmPFC中心坐标
vmpfc_masker = NiftiSpheresMasker(
    seeds=vmpfc_coords,
    radius=10,  # 10mm半径的球
    standardize=False,
)

# 提取每个被试在vmPFC中的平均Δactivation
vmpfc_values = []
for fmap in fmri_maps:
    vals = vmpfc_masker.fit_transform(fmap)
    vmpfc_values.append(vals.flat[0])

vmpfc_values = np.array(vmpfc_values)
lambda_arr = np.array(lambda_values)

print(f"vmPFC Δactivation 均值: {vmpfc_values.mean():.3f}")
print(f"vmPFC Δactivation 标准差: {vmpfc_values.std():.3f}")

# 相关检验
r_vmpfc, p_vmpfc = stats.pearsonr(lambda_arr, vmpfc_values)
r_spearman, p_spearman = stats.spearmanr(lambda_arr, vmpfc_values)

print(f"\nPearson r = {r_vmpfc:.3f}, p = {p_vmpfc:.4f}")
print(f"Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")

# 散点图
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.scatter(lambda_arr, vmpfc_values, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

# 回归线
if n_matched >= 3:
    slope, intercept = np.polyfit(lambda_arr, vmpfc_values, 1)
    x_line = np.linspace(lambda_arr.min() - 0.1, lambda_arr.max() + 0.1, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, alpha=0.7)

ax.set_xlabel('Δλ (Late - Early)', fontsize=13)
ax.set_ylabel('vmPFC Δactivation (Late - Early)', fontsize=13)
ax.set_title(f'Brain-Behavior Correlation (n={n_matched})\n'
             f'Pearson r={r_vmpfc:.3f}, p={p_vmpfc:.4f}', fontsize=14)
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

# 标注被试
for i, subj in enumerate(available_subs):
    ax.annotate(subj.replace('sub-', ''),
                (lambda_arr[i], vmpfc_values[i]),
                fontsize=8, alpha=0.6,
                xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'vmpfc_brain_behavior_scatter.png'), dpi=150)
print("\n保存: vmpfc_brain_behavior_scatter.png")

# ============================================================
# 4. 多ROI分析：几个关键区域
# ============================================================

print(f"\n{'─' * 40}")
print("多ROI分析")

# 定义几个理论驱动的ROI
rois = {
    'vmPFC': (0, 34, -16),  # 价值计算
    'L_insula': (-34, 18, -4),  # 损失厌恶/风险
    'R_insula': (34, 18, -4),  # 损失厌恶/风险
    'dACC': (0, 24, 32),  # 冲突监控
    'L_amygdala': (-22, -4, -18),  # 负面情绪
    'R_amygdala': (22, -4, -18),  # 负面情绪
    'ventral_striatum': (0, 10, -6),  # 奖赏
    'PCC': (0, -52, 16),  # 自我参照/默认网络
}

roi_results = []

for roi_name, coords in rois.items():
    masker = NiftiSpheresMasker(seeds=[coords], radius=8, standardize=False)

    roi_vals = []
    for fmap in fmri_maps:
        vals = masker.fit_transform(fmap)
        roi_vals.append(vals.flat[0])

    roi_vals = np.array(roi_vals)
    r, p = stats.pearsonr(lambda_arr, roi_vals)
    rho, p_sp = stats.spearmanr(lambda_arr, roi_vals)

    roi_results.append({
        'ROI': roi_name,
        'MNI_coords': coords,
        'mean_delta_activation': roi_vals.mean(),
        'pearson_r': r,
        'pearson_p': p,
        'spearman_rho': rho,
        'spearman_p': p_sp,
    })

    sig = '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.1 else ''
    print(f"  {roi_name:20s}  r={r:+.3f}  p={p:.4f} {sig}")

roi_df = pd.DataFrame(roi_results)
roi_df.to_csv(os.path.join(output_dir, 'roi_brain_behavior_correlations.csv'), index=False)
print(f"\n保存: roi_brain_behavior_correlations.csv")

# ============================================================
# 5. 全脑体素级关联（探索性）
# ============================================================
# 用 SecondLevelModel 把 Δλ 作为协变量
# 这会告诉我们：哪些体素的fatigue变化和行为变化最相关？

print(f"\n{'─' * 40}")
print("全脑体素级关联（探索性分析）...")

# 设计矩阵：截距 + Δλ 作为协变量
# Δλ 需要去均值（这样截距捕获群体平均效应，Δλ列捕获个体差异）
lambda_demeaned = lambda_arr - lambda_arr.mean()

design_matrix_bb = pd.DataFrame({
    'intercept': np.ones(n_matched),
    'delta_lambda': lambda_demeaned,
})

second_level_bb = SecondLevelModel(smoothing_fwhm=None)
second_level_bb.fit(fmri_maps, design_matrix=design_matrix_bb)

# 检验 delta_lambda 的效应
z_map_bb = second_level_bb.compute_contrast(
    second_level_contrast='delta_lambda',
    output_type='z_score'
)

# 保存和可视化
z_map_bb.to_filename(os.path.join(output_dir, 'brain_behavior_wholebrain_zmap.nii.gz'))

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 4))
plotting.plot_stat_map(
    z_map_bb,
    threshold=2.3,
    title=f'Brain regions where LOSS fatigue change correlates with Δλ (n={n_matched})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=ax2,
)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'brain_behavior_wholebrain.png'), dpi=150)
print("保存: brain_behavior_wholebrain.png")

# 提取显著簇
try:
    bb_table = get_clusters_table(z_map_bb, stat_threshold=2.3, min_distance=8)
    if len(bb_table) > 0:
        print(f"\n显著簇（Δλ与脑激活变化的关联）:")
        print(bb_table.head(15).to_string())
        bb_table.to_csv(os.path.join(output_dir, 'brain_behavior_clusters.csv'), index=False)
    else:
        print("\n没有找到显著簇（可能需要更多被试）")
except Exception as e:
    print(f"簇提取出错: {e}")

# ============================================================
# 总结
# ============================================================

print(f"\n{'=' * 60}")
print("脑-行为关联分析完成！")
print(f"{'=' * 60}")
print(f"匹配被试数: {n_matched}")
print(f"\n核心结果:")
print(f"  vmPFC: Pearson r = {r_vmpfc:.3f}, p = {p_vmpfc:.4f}")
print(f"\n解读:")
if r_vmpfc > 0 and p_vmpfc < 0.05:
    print("  ✓ 正相关且显著：行为上λ变化大的人，vmPFC对loss的神经敏感度变化也大")
    print("    → 行为和神经的time-on-task效应在个体层面是联系的")
elif r_vmpfc > 0:
    print("  → 正相关但不显著：方向符合预期，但统计力不足")
    print("    → 可能需要更多被试才能达到显著水平")
else:
    print("  → 未发现预期的正相关")
    print("    → 可能vmPFC不是联接行为和神经变化的关键区域")
    print("    → 查看多ROI结果，可能其他区域（如insula）更相关")

print(f"\n结果保存在: {output_dir}/")
print(f"\n论文中可以这样写:")
print(f"  'To test whether behavioral and neural time-on-task effects")
print(f"   are linked at the individual level, we correlated each")
print(f"   participant's change in loss aversion (Δλ) with the change")
print(f"   in neural sensitivity to loss magnitude in vmPFC.'")