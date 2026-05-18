"""
27_trial_level_fatigue.py
=========================
Trial-level 疲劳分析：在GLM中加入 trial_number 和 loss × trial_number 交互项。

Part D 用的是前半 vs 后半的粗略对比。这个脚本更精细：
  - 直接把 trial_number 作为连续变量放进GLM
  - 加入 loss × trial_number 的交互项
  - 如果 loss_x_trial 的系数显著为正：loss敏感度随时间线性增强
  - 这比2段对比的信息量大得多

Design matrix 每个trial 5行事件：
  1. gamble       — modulation=1（基本事件）
  2. gain_mod     — modulation=gain（demeaned）
  3. loss_mod     — modulation=loss（demeaned）
  4. trial_mod    — modulation=trial_number（demeaned，整体时间效应）
  5. loss_x_trial — modulation=loss×trial_number（demeaned，核心交互项）

运行方式：conda activate narps && python 27_trial_level_fatigue.py
预计耗时：约 3-4 小时（41个被试）
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
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
output_dir = 'trial_level_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

# 自动检测有fMRIPrep数据的被试
subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])
print(f"找到 {len(subjects)} 个被试")


# ============================================================
# 辅助函数
# ============================================================

def prepare_events_trial_level(events_file, run_number):
    """
    构建包含 trial_number 和 loss×trial 交互项的事件文件。

    参数:
        events_file: 原始事件TSV路径
        run_number: 1-4，用于计算全局trial编号
    返回:
        events_df: 包含5种trial_type的DataFrame
    """
    raw_events = pd.read_csv(events_file, sep='\t')
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()
    raw_events = raw_events.reset_index(drop=True)

    # 全局trial编号：Run 1 的 trial 是 1-64，Run 2 是 65-128，以此类推
    # 这样trial_number反映的是整个实验过程中的时间位置
    base_trial = (run_number - 1) * 64  # 每个run最多64个trial

    rows = []
    for i, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']
        trial_num = base_trial + i + 1  # 全局trial编号，从1开始

        # 1. 基本gamble事件
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gamble', 'modulation': 1.0
        })

        # 2. gain参数调制
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gain_mod', 'modulation': gain_val
        })

        # 3. loss参数调制
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'loss_mod', 'modulation': loss_val
        })

        # 4. trial编号调制（捕获整体时间趋势）
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'trial_mod', 'modulation': float(trial_num)
        })

        # 5. loss × trial 交互项（核心！）
        # 这个值 = loss金额 × trial编号
        # 如果系数为正：trial越晚，大脑对loss越敏感
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'loss_x_trial', 'modulation': loss_val * trial_num
        })

    events_df = pd.DataFrame(rows)

    # 对每个调制类型分别去均值
    for tt in ['gain_mod', 'loss_mod', 'trial_mod', 'loss_x_trial']:
        mean_val = events_df.loc[events_df['trial_type'] == tt, 'modulation'].mean()
        events_df.loc[events_df['trial_type'] == tt, 'modulation'] -= mean_val

    return events_df


def prepare_confounds(confounds_file):
    """读取confounds，提取6个运动参数。"""
    confounds = pd.read_csv(confounds_file, sep='\t')
    available = confounds.columns.tolist()
    if 'X' in available:
        motion_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    else:
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    return confounds[motion_cols].fillna(0)


# ============================================================
# 主循环：对每个被试跑trial-level GLM
# ============================================================

print("=" * 60)
print("Trial-Level 疲劳分析")
print("GLM 包含 loss × trial_number 交互项")
print("=" * 60)

# 存储每个被试的z-map
loss_x_trial_maps = []  # 核心：loss×trial交互
loss_maps = []  # loss主效应（对比用）
trial_maps = []  # trial主效应（整体时间趋势）

successful = []
failed = []
total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"处理 {subj} ...")
    subj_start = time.time()

    # 检查输出是否已存在
    out_file = os.path.join(output_dir, f'{subj}_loss_x_trial_zmap.nii.gz')
    if os.path.exists(out_file):
        from nilearn.image import load_img

        loss_x_trial_maps.append(load_img(out_file))
        loss_maps.append(load_img(os.path.join(output_dir, f'{subj}_loss_zmap.nii.gz')))
        trial_maps.append(load_img(os.path.join(output_dir, f'{subj}_trial_zmap.nii.gz')))
        print(f"  ✓ 已存在，跳过")
        successful.append(subj)
        continue

    fmriprep_dir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fmriprep_dir):
        print(f"  ⚠️  fMRIPrep目录不存在，跳过")
        failed.append((subj, "fMRIPrep目录不存在"))
        continue

    try:
        fmri_imgs = []
        events_list = []
        confounds_list = []

        for run in range(1, 5):
            run_str = f'{run:02d}'

            bold_file = os.path.join(
                fmriprep_dir,
                f'{subj}_task-MGT_run-{run_str}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
            )
            events_file = os.path.join(
                data_dir, subj, 'func',
                f'{subj}_task-MGT_run-{run_str}_events.tsv'
            )
            confounds_file = os.path.join(
                fmriprep_dir,
                f'{subj}_task-MGT_run-{run_str}_bold_confounds.tsv'
            )

            for f, label in [(bold_file, 'BOLD'), (events_file, 'events'), (confounds_file, 'confounds')]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"{label}: {f}")

            fmri_imgs.append(bold_file)
            events_list.append(prepare_events_trial_level(events_file, run))
            confounds_list.append(prepare_confounds(confounds_file))

        # 拟合GLM
        glm = FirstLevelModel(
            t_r=TR,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=0.01,
            smoothing_fwhm=6,
            minimize_memory=True,
        )

        glm.fit(fmri_imgs, events_list, confounds_list)

        # 确认design matrix包含预期的列
        dm_cols = glm.design_matrices_[0].columns.tolist()
        print(f"  Design matrix列: {[c for c in dm_cols if not c.startswith('drift') and c != 'constant']}")

        # 计算对比
        z_loss = glm.compute_contrast('loss_mod', output_type='z_score')
        z_trial = glm.compute_contrast('trial_mod', output_type='z_score')
        z_loss_x_trial = glm.compute_contrast('loss_x_trial', output_type='z_score')

        # 保存
        z_loss.to_filename(os.path.join(output_dir, f'{subj}_loss_zmap.nii.gz'))
        z_trial.to_filename(os.path.join(output_dir, f'{subj}_trial_zmap.nii.gz'))
        z_loss_x_trial.to_filename(os.path.join(output_dir, f'{subj}_loss_x_trial_zmap.nii.gz'))

        loss_maps.append(z_loss)
        trial_maps.append(z_trial)
        loss_x_trial_maps.append(z_loss_x_trial)

        elapsed = time.time() - subj_start
        print(f"  ✓ 完成（{elapsed:.0f} 秒）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ 失败: {e}（{elapsed:.0f} 秒）")
        failed.append((subj, str(e)))

print(f"\n第一阶段完成：{len(successful)} 成功, {len(failed)} 失败")

# ============================================================
# 组级分析
# ============================================================

n_subjects = len(loss_x_trial_maps)
print(f"\n{'=' * 60}")
print(f"组级分析（n={n_subjects}）")
print(f"{'=' * 60}")

design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})
cluster_threshold = 2.3

# ---- loss × trial 交互效应（核心结果）----
print("\n分析 loss × trial 交互效应...")
print("正值 = trial越晚，该区域对loss越敏感")

sl_interaction = SecondLevelModel(smoothing_fwhm=None)
sl_interaction.fit(loss_x_trial_maps, design_matrix=design_matrix)
z_group_interaction = sl_interaction.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

# ---- trial 主效应 ----
print("分析 trial 主效应...")
print("正值 = trial越晚，激活越强（整体时间趋势）")

sl_trial = SecondLevelModel(smoothing_fwhm=None)
sl_trial.fit(trial_maps, design_matrix=design_matrix)
z_group_trial = sl_trial.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

# ============================================================
# 显著簇报告
# ============================================================

print(f"\n{'─' * 40}")
print("显著簇报告")

print("\n===== LOSS × TRIAL 交互（核心结果）=====")
print("正值 = 大脑对loss的敏感度随trial线性增强")
try:
    interaction_table = get_clusters_table(
        z_group_interaction,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(interaction_table) > 0:
        print(interaction_table.head(20).to_string())
        interaction_table.to_csv(os.path.join(output_dir, 'loss_x_trial_clusters.csv'), index=False)
    else:
        print("  没有找到显著簇")
except Exception as e:
    print(f"  簇提取出错: {e}")

print("\n===== TRIAL 主效应 =====")
print("正值 = 整体激活随时间增强（不特定于loss）")
try:
    trial_table = get_clusters_table(
        z_group_trial,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(trial_table) > 0:
        print(trial_table.head(15).to_string())
        trial_table.to_csv(os.path.join(output_dir, 'trial_effect_clusters.csv'), index=False)
    else:
        print("  没有找到显著簇")
except Exception as e:
    print(f"  簇提取出错: {e}")

# ============================================================
# 可视化
# ============================================================

print(f"\n{'─' * 40}")
print("生成可视化...")

# --- 图1：loss × trial 交互 vs loss 主效应对比 ---
# 重新跑一下loss主效应的组级分析（和Part C一样，但用这个GLM的结果）
sl_loss = SecondLevelModel(smoothing_fwhm=None)
sl_loss.fit(loss_maps, design_matrix=design_matrix)
z_group_loss = sl_loss.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))

plotting.plot_stat_map(
    z_group_loss,
    threshold=cluster_threshold,
    title=f'Loss main effect (n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[0],
)

plotting.plot_stat_map(
    z_group_trial,
    threshold=cluster_threshold,
    title=f'Trial number effect — overall time trend (n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[1],
)

plotting.plot_stat_map(
    z_group_interaction,
    threshold=cluster_threshold,
    title=f'Loss × Trial interaction — loss sensitivity increasing over time? (n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[2],
)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'trial_level_three_effects.png'), dpi=150)
print("  保存: trial_level_three_effects.png")

# --- 图2：交互效应的玻璃脑 ---
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
plotting.plot_glass_brain(
    z_group_interaction,
    threshold=cluster_threshold,
    title=f'Loss × Trial interaction (n={n_subjects})',
    display_mode='lyrz',
    colorbar=True,
    axes=ax2,
)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'loss_x_trial_glass_brain.png'), dpi=150)
print("  保存: loss_x_trial_glass_brain.png")

# ============================================================
# 保存
# ============================================================

z_group_interaction.to_filename(os.path.join(output_dir, 'group_loss_x_trial_zmap.nii.gz'))
z_group_trial.to_filename(os.path.join(output_dir, 'group_trial_zmap.nii.gz'))
z_group_loss.to_filename(os.path.join(output_dir, 'group_loss_zmap.nii.gz'))

# ============================================================
# 总结
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Trial-Level 分析完成！总耗时: {total_elapsed / 60:.1f} 分钟")
print(f"{'=' * 60}")
print(f"被试数: {n_subjects}")
print(f"\n解读指南:")
print(f"  图中三行分别是:")
print(f"    1. Loss主效应 — 哪些区域编码loss金额（和Part C类似）")
print(f"    2. Trial主效应 — 哪些区域的激活随时间整体改变（不特定于loss）")
print(f"    3. Loss×Trial交互 — 哪些区域对loss的敏感度随时间线性增强")
print(f"\n  如果第3行有显著区域:")
print(f"    → loss敏感度的增强是渐进的、线性的")
print(f"    → 更支持'渐进疲劳'或'渐进策略调整'的解释")
print(f"  如果第3行没有显著区域:")
print(f"    → 变化可能是突变的（比如在某个时间点之后跳变）")
print(f"    → 更支持'策略切换'的解释")