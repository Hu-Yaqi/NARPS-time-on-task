"""
28_binned_fatigue_trajectory.py
===============================
分段GLM：把256个trial分成8个时间窗口（每个run内分前半后半），
对每个窗口单独估计loss的参数效应，在关键ROI中提取值，
画出行为+神经的渐进变化双轴图。

8个bin的分配：
  Run 1 前半 → bin 1,  Run 1 后半 → bin 2
  Run 2 前半 → bin 3,  Run 2 后半 → bin 4
  Run 3 前半 → bin 5,  Run 3 后半 → bin 6
  Run 4 前半 → bin 7,  Run 4 后半 → bin 8

运行方式：conda activate narps && python 28_binned_fatigue_trajectory.py
预计耗时：约 3-4 小时
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiSpheresMasker
from scipy import stats
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
output_dir = 'binned_fatigue_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0
N_BINS = 8

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])
print(f"找到 {len(subjects)} 个被试")

# ROI定义（从脚本27和Part D的关键簇）
rois = {
    'dmPFC': (8, 44, 54),  # 脚本27 loss×trial交互最大簇
    'vmPFC': (-12, 36, -13),  # Part D early vs late最大簇
    'mPFC': (6, 30, 59),  # 脚本27第二大交互簇
}


# ============================================================
# 辅助函数
# ============================================================

def prepare_events_binned(events_file, run_number):
    """构建分bin的事件文件。所有8个loss_bin都出现在每个run中。"""
    raw_events = pd.read_csv(events_file, sep='\t')
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()
    raw_events = raw_events.reset_index(drop=True)

    n_trials = len(raw_events)
    half = n_trials // 2

    bin_first = (run_number - 1) * 2 + 1
    bin_second = bin_first + 1

    rows = []
    for i, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']

        bin_num = bin_first if i < half else bin_second

        # 基本gamble事件
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gamble', 'modulation': 1.0
        })

        # gain调制
        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gain_mod', 'modulation': gain_val
        })

        # 所有8个loss_bin都添加，但只有当前bin有真实值，其余为0
        for b in range(1, 9):
            if b == bin_num:
                rows.append({
                    'onset': onset, 'duration': duration,
                    'trial_type': f'loss_bin{b}',
                    'modulation': loss_val
                })
            else:
                rows.append({
                    'onset': onset, 'duration': duration,
                    'trial_type': f'loss_bin{b}',
                    'modulation': 0.0
                })

    events_df = pd.DataFrame(rows)

    # gain_mod 整体去均值
    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean

    # 每个loss_bin分别去均值（只对非零值去均值）
    for b in range(1, 9):
        tt = f'loss_bin{b}'
        mask = (events_df['trial_type'] == tt) & (events_df['modulation'] != 0)
        if mask.sum() > 0:
            mean_val = events_df.loc[mask, 'modulation'].mean()
            events_df.loc[mask, 'modulation'] -= mean_val

    return events_df


def prepare_confounds(confounds_file):
    confounds = pd.read_csv(confounds_file, sep='\t')
    available = confounds.columns.tolist()
    if 'X' in available:
        motion_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    else:
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    return confounds[motion_cols].fillna(0)


# ============================================================
# 主循环
# ============================================================

print("=" * 60)
print(f"分段GLM：{N_BINS}个时间窗口")
print("=" * 60)

roi_trajectories = {name: [] for name in rois}
maskers = {name: NiftiSpheresMasker(seeds=[coords], radius=8, standardize=False)
           for name, coords in rois.items()}

successful = []
failed = []
total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"处理 {subj} ...")
    subj_start = time.time()

    fmriprep_dir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fmriprep_dir):
        print(f"  ⚠️  跳过（无fMRIPrep）")
        failed.append((subj, "无fMRIPrep"))
        continue

    try:
        fmri_imgs = []
        events_list = []
        confounds_list = []

        for run in range(1, 5):
            run_str = f'{run:02d}'

            bold_file = os.path.join(fmriprep_dir,
                                     f'{subj}_task-MGT_run-{run_str}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
            events_file = os.path.join(data_dir, subj, 'func',
                                       f'{subj}_task-MGT_run-{run_str}_events.tsv')
            confounds_file = os.path.join(fmriprep_dir,
                                          f'{subj}_task-MGT_run-{run_str}_bold_confounds.tsv')

            for f, label in [(bold_file, 'BOLD'), (events_file, 'events'), (confounds_file, 'confounds')]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"{label}: {f}")

            fmri_imgs.append(bold_file)
            events_list.append(prepare_events_binned(events_file, run))
            confounds_list.append(prepare_confounds(confounds_file))

        glm = FirstLevelModel(
            t_r=TR, hrf_model='spm', drift_model='cosine',
            high_pass=0.01, smoothing_fwhm=6, minimize_memory=True,
        )
        glm.fit(fmri_imgs, events_list, confounds_list)

        print(
            f"  Design matrix columns: {[c for c in glm.design_matrices_[0].columns if 'loss' in c or 'gain' in c or 'gamble' in c]}")

        # 对每个bin提取ROI中的loss效应
        bin_values = {name: [] for name in rois}

        for b in range(1, N_BINS + 1):
            contrast_name = f'loss_bin{b}'
            z_map = glm.compute_contrast(contrast_name, output_type='z_score')

            for roi_name, masker in maskers.items():
                val = masker.fit_transform(z_map)
                bin_values[roi_name].append(val.flat[0])

        for roi_name in rois:
            roi_trajectories[roi_name].append(bin_values[roi_name])

        elapsed = time.time() - subj_start
        print(f"  ✓ 完成（{elapsed:.0f} 秒）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ 失败: {e}（{elapsed:.0f} 秒）")
        failed.append((subj, str(e)))

n_subjects = len(successful)
print(f"\n第一阶段完成：{n_subjects} 成功, {len(failed)} 失败")

# 转成numpy数组
for roi_name in rois:
    roi_trajectories[roi_name] = np.array(roi_trajectories[roi_name])

# ============================================================
# 行为数据：每个bin的拒绝率
# ============================================================

print(f"\n{'─' * 40}")
print("计算每个bin的行为指标...")

behavior = pd.read_csv('all_subjects_behavior.csv')
behavior_fmri = behavior[behavior['subject'].isin(successful)].copy()

bin_behavior = []

for subj in successful:
    subj_data = behavior_fmri[behavior_fmri['subject'] == subj]

    for run in range(1, 5):
        run_data = subj_data[subj_data['run'] == run].reset_index(drop=True)
        n = len(run_data)
        half = n // 2

        bin_first = (run - 1) * 2 + 1
        bin_second = bin_first + 1

        # 前半
        first_half = run_data.iloc[:half]
        if len(first_half) > 0:
            bin_behavior.append({
                'subject': subj, 'bin': bin_first,
                'reject_rate': (first_half['accepted'] == 0).mean(),
                'n_trials': len(first_half),
            })

        # 后半
        second_half = run_data.iloc[half:]
        if len(second_half) > 0:
            bin_behavior.append({
                'subject': subj, 'bin': bin_second,
                'reject_rate': (second_half['accepted'] == 0).mean(),
                'n_trials': len(second_half),
            })

bin_df = pd.DataFrame(bin_behavior)

print(f"bin_df shape: {bin_df.shape}")
print(f"bin_df columns: {bin_df.columns.tolist()}")
print(bin_df.head())

# 群体平均
bin_means = bin_df.groupby('bin').agg(
    reject_mean=('reject_rate', 'mean'),
    reject_sem=('reject_rate', lambda x: x.std() / np.sqrt(len(x))),
).reset_index()

print("\n每个bin的平均拒绝率:")
for _, row in bin_means.iterrows():
    print(f"  Bin {int(row['bin'])}: {row['reject_mean']:.3f} ± {row['reject_sem']:.3f}")

# ============================================================
# 可视化：双轴图
# ============================================================

print(f"\n{'─' * 40}")
print("生成可视化...")

x = np.arange(1, N_BINS + 1)
bin_labels = ['R1\nfirst', 'R1\nsecond', 'R2\nfirst', 'R2\nsecond',
              'R3\nfirst', 'R3\nsecond', 'R4\nfirst', 'R4\nsecond']

for roi_name in rois:
    data = roi_trajectories[roi_name]  # shape: (n_subjects, N_BINS)

    # 群体平均和SEM
    roi_mean = np.nanmean(data, axis=0)
    roi_sem = np.nanstd(data, axis=0) / np.sqrt(n_subjects)

    # ---- 双轴图 ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：行为拒绝率
    color1 = '#E74C3C'  # 红色
    ax1.set_xlabel('Time Window', fontsize=13)
    ax1.set_ylabel('Rejection Rate (behavioral)', fontsize=13, color=color1)
    ax1.errorbar(x, bin_means['reject_mean'].values, yerr=bin_means['reject_sem'].values,
                 color=color1, marker='o', markersize=8, linewidth=2.5, capsize=4,
                 label='Rejection rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=10)

    # 右轴：神经loss效应
    ax2 = ax1.twinx()
    color2 = '#2E86C1'  # 蓝色
    ax2.set_ylabel(f'{roi_name} loss sensitivity (z-score)', fontsize=13, color=color2)
    ax2.errorbar(x, roi_mean, yerr=roi_sem,
                 color=color2, marker='s', markersize=8, linewidth=2.5, capsize=4,
                 label=f'{roi_name} BOLD')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加run分隔线
    for boundary in [2.5, 4.5, 6.5]:
        ax1.axvline(boundary, color='gray', linestyle='--', alpha=0.3)

    # run标签
    for i, label in enumerate(['Run 1', 'Run 2', 'Run 3', 'Run 4']):
        ax1.text(i * 2 + 1.5, ax1.get_ylim()[1], label,
                 ha='center', va='bottom', fontsize=11, color='gray', fontweight='bold')

    # 标题
    ax1.set_title(f'Behavioral and Neural Loss Sensitivity Across Time\n'
                  f'{roi_name} (n={n_subjects})', fontsize=14, fontweight='bold')

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    # 添加线性趋势检验
    # 行为
    slope_b, intercept_b, r_b, p_b, _ = stats.linregress(x, bin_means['reject_mean'].values)
    # 神经
    slope_n, intercept_n, r_n, p_n, _ = stats.linregress(x, roi_mean)

    textstr = (f'Behavioral trend: r={r_b:.3f}, p={p_b:.4f}\n'
               f'Neural trend: r={r_n:.3f}, p={p_n:.4f}')
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'trajectory_{roi_name.replace("/", "_")}.png'), dpi=150)
    print(f"  保存: trajectory_{roi_name.replace('/', '_')}.png")
    plt.close()

# ---- 综合图：三个ROI并排 ----
fig_all, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, roi_name in enumerate(rois):
    data = roi_trajectories[roi_name]
    roi_mean = np.nanmean(data, axis=0)
    roi_sem = np.nanstd(data, axis=0) / np.sqrt(n_subjects)

    ax = axes[idx]

    # 行为（左轴）
    color1 = '#E74C3C'
    ax.errorbar(x, bin_means['reject_mean'].values, yerr=bin_means['reject_sem'].values,
                color=color1, marker='o', markersize=6, linewidth=2, capsize=3, alpha=0.8)
    ax.set_ylabel('Rejection Rate', fontsize=11, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{i}' for i in range(1, 9)], fontsize=9)
    ax.set_xlabel('Time Bin', fontsize=11)

    # 神经（右轴）
    ax2 = ax.twinx()
    color2 = '#2E86C1'
    ax2.errorbar(x, roi_mean, yerr=roi_sem,
                 color=color2, marker='s', markersize=6, linewidth=2, capsize=3, alpha=0.8)
    ax2.set_ylabel('Loss BOLD (z)', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # run分隔线
    for boundary in [2.5, 4.5, 6.5]:
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3)

    # 趋势检验
    slope_n, _, r_n, p_n, _ = stats.linregress(x, roi_mean)
    ax.set_title(f'{roi_name}\nr={r_n:.3f}, p={p_n:.4f}', fontsize=12, fontweight='bold')

plt.suptitle(f'Loss Sensitivity Trajectory: Behavior (red) vs Brain (blue), n={n_subjects}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_all.savefig(os.path.join(output_dir, 'trajectory_all_rois.png'), dpi=150, bbox_inches='tight')
print(f"  保存: trajectory_all_rois.png")

# ============================================================
# 保存数据
# ============================================================

# 保存ROI轨迹数据
trajectory_data = {'bin': x}
for roi_name in rois:
    data = roi_trajectories[roi_name]
    trajectory_data[f'{roi_name}_mean'] = np.nanmean(data, axis=0)
    trajectory_data[f'{roi_name}_sem'] = np.nanstd(data, axis=0) / np.sqrt(n_subjects)
trajectory_data['reject_rate_mean'] = bin_means['reject_mean'].values
trajectory_data['reject_rate_sem'] = bin_means['reject_sem'].values

traj_df = pd.DataFrame(trajectory_data)
traj_df.to_csv(os.path.join(output_dir, 'trajectory_data.csv'), index=False)
print(f"\n保存: trajectory_data.csv")

# ============================================================
# 总结
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"分段GLM分析完成！总耗时: {total_elapsed / 60:.1f} 分钟")
print(f"{'=' * 60}")
print(f"被试数: {n_subjects}")
print(f"\n线性趋势检验:")
for roi_name in rois:
    data = roi_trajectories[roi_name]
    roi_mean = np.nanmean(data, axis=0)
    _, _, r, p, _ = stats.linregress(x, roi_mean)
    print(f"  {roi_name}: r={r:.3f}, p={p:.4f}")

slope_b, _, r_b, p_b, _ = stats.linregress(x, bin_means['reject_mean'].values)
print(f"  Behavior (reject rate): r={r_b:.3f}, p={p_b:.4f}")