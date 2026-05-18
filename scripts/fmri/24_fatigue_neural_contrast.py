"""
24_fatigue_neural_contrast.py
=============================
Part D: 疲劳的神经签名 — 前半（Run 1-2）vs 后半（Run 3-4）的神经对比。

做什么：
  1. 对每个被试，分别用前两个run和后两个run拟合GLM
  2. 计算每个被试的"后半 - 前半"差异z-map（gain和loss各一个）
  3. 组级分析：这些差异在群体中是否一致？
  4. 如果loss的差异显著，说明疲劳不仅改变了行为（λ↑），
     也改变了大脑对loss的处理方式——这就是Part D的核心发现

概念解释：
  - 我们不是比较"前半的脑激活 vs 后半的脑激活"（那会混入很多东西）
  - 而是比较"前半中loss金额的调制效应 vs 后半中loss金额的调制效应"
  - 也就是说：大脑对loss金额的敏感程度，有没有随时间改变？
  - 这和行为上的发现（λ从早期到晚期增加）是同一个问题的神经版本

运行方式：conda activate narps && python 24_fatigue_neural_contrast.py
预计耗时：约 15-20 分钟（每个被试跑2个半程GLM）
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn.image import math_img
from nilearn import plotting
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'fatigue_neural_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

# 已确认有fMRIPrep数据的被试
import os
subjects = sorted([d for d in os.listdir(os.path.join('data', 'derivatives', 'fmriprep'))
                   if d.startswith('sub-')])

# ============================================================
# 辅助函数（和22脚本相同）
# ============================================================

def prepare_events_for_run(events_file):
    """读取事件文件，构建 gamble + gain_mod + loss_mod 三行格式。"""
    raw_events = pd.read_csv(events_file, sep='\t')
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()

    rows = []
    for _, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']

        rows.append({'onset': onset, 'duration': duration,
                     'trial_type': 'gamble', 'modulation': 1.0})
        rows.append({'onset': onset, 'duration': duration,
                     'trial_type': 'gain_mod', 'modulation': gain_val})
        rows.append({'onset': onset, 'duration': duration,
                     'trial_type': 'loss_mod', 'modulation': loss_val})

    events_df = pd.DataFrame(rows)

    # 去均值
    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    loss_mean = events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean
    events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'] -= loss_mean

    return events_df


def prepare_confounds_for_run(confounds_file):
    """读取confounds，提取6个运动参数。"""
    confounds = pd.read_csv(confounds_file, sep='\t')
    available = confounds.columns.tolist()
    if 'X' in available:
        motion_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    else:
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    return confounds[motion_cols].fillna(0)


def fit_half_glm(subject, runs, label):
    """
    对指定被试的指定run列表拟合GLM，返回gain和loss的z-map。

    参数:
        subject: 被试ID，如 'sub-001'
        runs: run编号列表，如 [1, 2] 或 [3, 4]
        label: 用于打印的标签，如 'early' 或 'late'
    返回:
        (z_map_gain, z_map_loss): 两个NIfTI图像
    """
    fmriprep_dir = os.path.join(fmriprep_base, subject, 'func')

    fmri_imgs = []
    events_list = []
    confounds_list = []

    for run in runs:
        run_str = f'{run:02d}'

        bold_file = os.path.join(
            fmriprep_dir,
            f'{subject}_task-MGT_run-{run_str}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
        )
        events_file = os.path.join(
            data_dir, subject, 'func',
            f'{subject}_task-MGT_run-{run_str}_events.tsv'
        )
        confounds_file = os.path.join(
            fmriprep_dir,
            f'{subject}_task-MGT_run-{run_str}_bold_confounds.tsv'
        )

        fmri_imgs.append(bold_file)
        events_list.append(prepare_events_for_run(events_file))
        confounds_list.append(prepare_confounds_for_run(confounds_file))

    glm = FirstLevelModel(
        t_r=TR,
        hrf_model='spm',
        drift_model='cosine',
        high_pass=0.01,
        smoothing_fwhm=6,
        minimize_memory=True,
    )

    glm.fit(fmri_imgs, events_list, confounds_list)

    z_gain = glm.compute_contrast('gain_mod', output_type='z_score')
    z_loss = glm.compute_contrast('loss_mod', output_type='z_score')

    return z_gain, z_loss


# ============================================================
# 1. 对每个被试分别跑前半和后半的GLM
# ============================================================

print("=" * 60)
print("Part D: 疲劳的神经签名")
print("Run 1-2 (early) vs Run 3-4 (late)")
print("=" * 60)

# 存储每个被试的差异图（late - early）
diff_gain_maps = []   # 每个元素是一个被试的 (late_gain - early_gain) z-map
diff_loss_maps = []   # 每个元素是一个被试的 (late_loss - early_loss) z-map

successful = []
failed = []
total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"处理 {subj} ...")
    subj_start = time.time()

    try:
        # 前半：Run 1-2
        print(f"  拟合 early GLM (Run 1-2)...")
        z_gain_early, z_loss_early = fit_half_glm(subj, [1, 2], 'early')

        # 后半：Run 3-4
        print(f"  拟合 late GLM (Run 3-4)...")
        z_gain_late, z_loss_late = fit_half_glm(subj, [3, 4], 'late')

        # 计算差异：late - early
        # 如果 late > early（正值），说明后半对该调制更敏感
        # 如果 late < early（负值），说明后半对该调制不那么敏感了
        diff_gain = math_img('img1 - img2', img1=z_gain_late, img2=z_gain_early)
        diff_loss = math_img('img1 - img2', img1=z_loss_late, img2=z_loss_early)

        # 保存单被试差异图
        diff_gain.to_filename(os.path.join(output_dir, f'{subj}_gain_late_minus_early.nii.gz'))
        diff_loss.to_filename(os.path.join(output_dir, f'{subj}_loss_late_minus_early.nii.gz'))

        diff_gain_maps.append(diff_gain)
        diff_loss_maps.append(diff_loss)

        elapsed = time.time() - subj_start
        print(f"  ✓ 完成（耗时 {elapsed:.0f} 秒）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ 失败: {e}（耗时 {elapsed:.0f} 秒）")
        failed.append((subj, str(e)))

print(f"\n第一阶段完成：{len(successful)} 成功, {len(failed)} 失败")
if failed:
    for s, r in failed:
        print(f"  {s}: {r}")

# ============================================================
# 2. 组级分析：差异图的 One-Sample T-Test
# ============================================================
# 问题：跨被试，"late - early"的差异是否一致地不为零？
# 如果某个脑区的loss差异显著为正，说明该区域在后半对loss更敏感
# → 疲劳增强了该区域的loss处理，和行为上λ↑的发现对应

print(f"\n{'=' * 60}")
print("组级分析：Late - Early 差异")
print(f"{'=' * 60}")

n_subjects = len(diff_gain_maps)
design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})

# ---- GAIN 差异 ----
print("\n分析 GAIN 差异 (late - early)...")
second_level_gain = SecondLevelModel(smoothing_fwhm=None)
second_level_gain.fit(diff_gain_maps, design_matrix=design_matrix)
z_group_gain_diff = second_level_gain.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

# ---- LOSS 差异（这是核心！）----
print("分析 LOSS 差异 (late - early)...")
second_level_loss = SecondLevelModel(smoothing_fwhm=None)
second_level_loss.fit(diff_loss_maps, design_matrix=design_matrix)
z_group_loss_diff = second_level_loss.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

# ============================================================
# 3. 显著簇报告
# ============================================================

cluster_threshold = 2.3   # 和Part C保持一致

print(f"\n{'─' * 40}")
print("显著簇报告")

print("\n===== GAIN 差异 (late - early) =====")
print("正值 = 后半对gain更敏感; 负值 = 后半对gain不敏感了")
try:
    gain_diff_table = get_clusters_table(
        z_group_gain_diff,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(gain_diff_table) > 0:
        # 只显示前20个簇
        print(gain_diff_table.head(20).to_string())
        gain_diff_table.to_csv(os.path.join(output_dir, 'gain_fatigue_clusters.csv'), index=False)
    else:
        print("  没有找到显著簇")
except Exception as e:
    print(f"  簇提取出错: {e}")

print("\n===== LOSS 差异 (late - early) =====")
print("正值 = 后半对loss更敏感; 负值 = 后半对loss不敏感了")
try:
    loss_diff_table = get_clusters_table(
        z_group_loss_diff,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(loss_diff_table) > 0:
        print(loss_diff_table.head(20).to_string())
        loss_diff_table.to_csv(os.path.join(output_dir, 'loss_fatigue_clusters.csv'), index=False)
    else:
        print("  没有找到显著簇")
except Exception as e:
    print(f"  簇提取出错: {e}")

# ============================================================
# 4. 可视化
# ============================================================

print(f"\n{'─' * 40}")
print("生成可视化...")

# --- 图1：Gain和Loss的疲劳差异 ---
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    z_group_gain_diff,
    threshold=cluster_threshold,
    title=f'GAIN fatigue effect: Late - Early (n={n_subjects}, z>{cluster_threshold})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[0],
)

plotting.plot_stat_map(
    z_group_loss_diff,
    threshold=cluster_threshold,
    title=f'LOSS fatigue effect: Late - Early (n={n_subjects}, z>{cluster_threshold})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[1],
)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fatigue_neural_contrast.png'), dpi=150)
print("  保存: fatigue_neural_contrast.png")

# --- 图2：只看Loss的疲劳效应（核心结果），用玻璃脑 ---
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))

plotting.plot_glass_brain(
    z_group_loss_diff,
    threshold=cluster_threshold,
    title=f'Loss sensitivity change with fatigue (Late - Early, n={n_subjects})',
    display_mode='lyrz',
    colorbar=True,
    axes=ax2,
)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'loss_fatigue_glass_brain.png'), dpi=150)
print("  保存: loss_fatigue_glass_brain.png")

# --- 图3：对比Part C的整体效应 vs Part D的疲劳变化 ---
# 如果 group_level_results 中有整体loss z-map，做一个并排对比
group_loss_path = os.path.join('group_level_results', 'group_loss_zmap.nii.gz')
if os.path.exists(group_loss_path):
    from nilearn.image import load_img
    group_loss = load_img(group_loss_path)

    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))

    plotting.plot_stat_map(
        group_loss,
        threshold=cluster_threshold,
        title=f'Part C: Overall LOSS effect (all runs)',
        display_mode='z',
        cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
        axes=axes3[0],
    )

    plotting.plot_stat_map(
        z_group_loss_diff,
        threshold=cluster_threshold,
        title=f'Part D: LOSS fatigue change (Late - Early)',
        display_mode='z',
        cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
        axes=axes3[1],
    )

    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'loss_overall_vs_fatigue.png'), dpi=150)
    print("  保存: loss_overall_vs_fatigue.png")

# ============================================================
# 5. 保存组级差异z-map
# ============================================================

z_group_gain_diff.to_filename(os.path.join(output_dir, 'group_gain_fatigue_zmap.nii.gz'))
z_group_loss_diff.to_filename(os.path.join(output_dir, 'group_loss_fatigue_zmap.nii.gz'))

# ============================================================
# 总结
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Part D 完成！总耗时: {total_elapsed/60:.1f} 分钟")
print(f"{'=' * 60}")
print(f"被试数: {n_subjects}")
print(f"结果保存在: {output_dir}/")
print(f"\n解读指南:")
print(f"  fatigue_neural_contrast.png:")
print(f"    上图 = gain的疲劳变化（预期较弱）")
print(f"    下图 = loss的疲劳变化（核心结果）")
print(f"    红色 = 后半比前半更敏感（增强）")
print(f"    蓝色 = 后半比前半不敏感（减弱）")
print(f"\n  和行为结果的对应:")
print(f"    行为：λ↑（后半更怕loss）→ 预期loss区域红色（更敏感）")
print(f"    行为：α→1（后半感知更线性）→ gain区域可能变化不大")
print(f"\n  loss_overall_vs_fatigue.png:")
print(f"    对比整体loss效应 vs 疲劳变化")
print(f"    如果同一区域两图都亮：该区域既编码loss，又随疲劳改变")