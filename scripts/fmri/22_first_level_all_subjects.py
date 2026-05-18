"""
22_first_level_all_subjects.py
==============================
对所有已下载的被试（sub-001 到 sub-009）循环跑 First-Level GLM。
每个被试生成 gain_zmap 和 loss_zmap，保存到 first_level_results/ 目录。

逻辑和 21_first_level_glm.py 完全一致，只是套了一个被试循环。

运行方式：conda activate narps && python 22_first_level_all_subjects.py
预计耗时：每个被试约 5-10 分钟，9个被试总共约 45-90 分钟。
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

data_dir = 'data'  # 数据根目录
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'first_level_results'  # 保存z-map的目录
os.makedirs(output_dir, exist_ok=True)
TR = 1.0  # 重复时间（秒）

# 已下载fMRI数据的被试列表
import os
subjects = sorted([d for d in os.listdir(os.path.join('data', 'derivatives', 'fmriprep'))
                   if d.startswith('sub-')])

# ============================================================
# 辅助函数：为单个被试准备事件文件
# ============================================================

def prepare_events_for_run(events_file):
    """
    读取原始事件文件，构建三行事件格式（gamble + gain_mod + loss_mod）。
    和 21_first_level_glm.py 中的逻辑完全相同。

    参数:
        events_file: 原始事件TSV文件路径
    返回:
        events_df: 处理后的事件DataFrame，包含 onset, duration, trial_type, modulation
    """
    raw_events = pd.read_csv(events_file, sep='\t')
    # 去掉没有反应的trial（NoResp）
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()

    rows = []
    for _, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']

        # 基本gamble事件：modulation=1，表示"这里有一个trial"
        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'gamble',
            'modulation': 1.0
        })

        # gain参数调制：gain金额作为调制值
        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'gain_mod',
            'modulation': gain_val
        })

        # loss参数调制：loss金额作为调制值
        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'loss_mod',
            'modulation': loss_val
        })

    events_df = pd.DataFrame(rows)

    # 对 gain_mod 和 loss_mod 的调制值去均值（demeaning）
    # 这样 gamble 列捕获的是"平均"激活，gain_mod/loss_mod 捕获的是"偏离均值"的效应
    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    loss_mean = events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean
    events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'] -= loss_mean

    return events_df


def prepare_confounds_for_run(confounds_file):
    """
    读取confounds文件，提取6个运动参数。

    参数:
        confounds_file: fMRIPrep输出的confounds TSV文件路径
    返回:
        confounds_df: 只包含6个运动参数的DataFrame
    """
    confounds = pd.read_csv(confounds_file, sep='\t')
    available = confounds.columns.tolist()

    # fMRIPrep不同版本的列名可能不同
    if 'X' in available:
        motion_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    else:
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    # fillna(0)：第一个时间点的运动参数可能是NaN
    return confounds[motion_cols].fillna(0)


# ============================================================
# 主循环：逐个被试跑GLM
# ============================================================

print("=" * 60)
print("First-Level GLM — 批量处理所有被试")
print("=" * 60)

# 记录成功/失败的被试
successful = []
failed = []

total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"处理 {subj} ...")
    subj_start = time.time()

    # 检查这个被试的fMRIPrep数据是否存在
    fmriprep_dir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fmriprep_dir):
        print(f"  ⚠️  {subj} 的fMRIPrep目录不存在，跳过")
        failed.append((subj, "fMRIPrep目录不存在"))
        continue

    # 检查输出是否已经存在（避免重复计算）
    gain_out = os.path.join(output_dir, f'{subj}_gain_zmap.nii.gz')
    loss_out = os.path.join(output_dir, f'{subj}_loss_zmap.nii.gz')
    if os.path.exists(gain_out) and os.path.exists(loss_out):
        print(f"  ✓ {subj} 的z-map已存在，跳过（如需重跑请先删除）")
        successful.append(subj)
        continue

    try:
        # ---- 准备4个run的数据 ----
        fmri_imgs = []
        events_list = []
        confounds_list = []

        for run in range(1, 5):
            run_str = f'{run:02d}'

            # BOLD图像路径
            bold_file = os.path.join(
                fmriprep_dir,
                f'{subj}_task-MGT_run-{run_str}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
            )

            # 事件文件路径（在原始数据目录中，不是fMRIPrep目录）
            events_file = os.path.join(
                data_dir, subj, 'func',
                f'{subj}_task-MGT_run-{run_str}_events.tsv'
            )

            # confounds文件路径
            confounds_file = os.path.join(
                fmriprep_dir,
                f'{subj}_task-MGT_run-{run_str}_bold_confounds.tsv'
            )

            # 检查文件是否存在
            for f, label in [(bold_file, 'BOLD'), (events_file, 'events'), (confounds_file, 'confounds')]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"{label} 文件不存在: {f}")

            fmri_imgs.append(bold_file)
            events_list.append(prepare_events_for_run(events_file))
            confounds_list.append(prepare_confounds_for_run(confounds_file))

        print(f"  数据准备完成（4个run）")

        # ---- 拟合GLM ----
        glm = FirstLevelModel(
            t_r=TR,
            hrf_model='spm',  # SPM标准血流动力学响应函数
            drift_model='cosine',  # 余弦低频漂移模型
            high_pass=0.01,  # 高通滤波截止频率（Hz）
            smoothing_fwhm=6,  # 6mm高斯平滑核
            minimize_memory=True,  # 节省内存（对16GB很重要）
        )

        glm.fit(fmri_imgs, events_list, confounds_list)
        print(f"  GLM拟合完成")

        # ---- 计算对比，生成z-map ----
        z_map_gain = glm.compute_contrast('gain_mod', stat_type='t', output_type='z_score')
        z_map_loss = glm.compute_contrast('loss_mod', stat_type='t', output_type='z_score')

        # ---- 保存 ----
        z_map_gain.to_filename(gain_out)
        z_map_loss.to_filename(loss_out)

        elapsed = time.time() - subj_start
        print(f"  ✓ 完成！保存到 {output_dir}/  （耗时 {elapsed:.0f} 秒）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ 失败: {e}  （耗时 {elapsed:.0f} 秒）")
        failed.append((subj, str(e)))

# ============================================================
# 总结
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"全部完成！总耗时: {total_elapsed / 60:.1f} 分钟")
print(f"成功: {len(successful)} 个被试 — {successful}")
if failed:
    print(f"失败: {len(failed)} 个被试:")
    for subj, reason in failed:
        print(f"  {subj}: {reason}")

# 列出所有已有的z-map文件
print(f"\n{output_dir}/ 目录中的文件:")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")