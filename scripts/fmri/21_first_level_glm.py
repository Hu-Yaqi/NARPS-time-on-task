import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

subject = 'sub-001'
data_dir = 'data'
fmriprep_dir = os.path.join(data_dir, 'derivatives', 'fmriprep', subject, 'func')
TR = 1.0

# ============================================================
# 准备每个run的数据
# ============================================================

fmri_imgs = []
events_list = []
confounds_list = []

for run in range(1, 5):
    run_str = f'{run:02d}'

    # BOLD图像
    bold_file = os.path.join(
        fmriprep_dir,
        f'{subject}_task-MGT_run-{run_str}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
    )
    fmri_imgs.append(bold_file)

    # 事件文件
    events_file = os.path.join(
        data_dir, subject, 'func',
        f'{subject}_task-MGT_run-{run_str}_events.tsv'
    )
    raw_events = pd.read_csv(events_file, sep='\t')
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()

    # 关键改动：创建三行事件来替代一行
    # 对每个trial，我们创建三个"事件"：
    #   1. "gamble" — 基本事件（这里有一个trial发生了）
    #   2. "gain_mod" — gain的参数调制（gain金额越高，这个regressor越强）
    #   3. "loss_mod" — loss的参数调制（loss金额越高，这个regressor越强）
    #
    # 这样GLM的design matrix里就会有三列：
    #   gamble: 所有trial的平均激活（不管金额多少）
    #   gain_mod: gain金额的效应（gain越大，信号越强吗？）
    #   loss_mod: loss金额的效应（loss越大，信号越强吗？）

    rows = []
    for _, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']

        # 基本gamble事件
        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'gamble',
            'modulation': 1.0  # 固定为1，只表示"这里有一个trial"
        })

        # gain 参数调制
        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'gain_mod',
            'modulation': gain_val  # gain金额作为调制值
        })

        # loss 参数调制
        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'loss_mod',
            'modulation': loss_val  # loss金额作为调制值
        })

    events_df = pd.DataFrame(rows)

    # 对 gain_mod 和 loss_mod 的调制值去均值
    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    loss_mean = events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean
    events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'] -= loss_mean

    events_list.append(events_df)

    # Confounds
    confounds_file = os.path.join(
        fmriprep_dir,
        f'{subject}_task-MGT_run-{run_str}_bold_confounds.tsv'
    )
    confounds = pd.read_csv(confounds_file, sep='\t')
    available = confounds.columns.tolist()

    if 'X' in available:
        motion_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    else:
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    confounds_selected = confounds[motion_cols].fillna(0)
    confounds_list.append(confounds_selected)

print(f"准备好了 {len(fmri_imgs)} 个run的数据")

# ============================================================
# 创建并运行 First-Level GLM
# ============================================================

print("\n开始拟合GLM...")

glm = FirstLevelModel(
    t_r=TR,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=0.01,
    smoothing_fwhm=6,
    minimize_memory=True,
)

glm.fit(fmri_imgs, events_list, confounds_list)
print("GLM拟合完成！")

# 确认design matrix列名
design_matrices = glm.design_matrices_
print("\nDesign matrix列名（Run 1）：")
for col in design_matrices[0].columns:
    print(f"  {col}")

# ============================================================
# 定义对比并生成统计图
# ============================================================

# 用gain_mod和loss_mod来定义对比
z_map_gain = glm.compute_contrast('gain_mod', stat_type='t', output_type='z_score')
z_map_loss = glm.compute_contrast('loss_mod', stat_type='t', output_type='z_score')

print("\n统计图已生成！")

# ============================================================
# 可视化
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    z_map_gain,
    threshold=2.5,
    title='Sub-001: Brain regions tracking GAIN (positive = more gain, more activity)',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes[0],
)

plotting.plot_stat_map(
    z_map_loss,
    threshold=2.5,
    title='Sub-001: Brain regions tracking LOSS (positive = more loss, more activity)',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes[1],
)

plt.tight_layout()
plt.savefig('fmri_glm_sub001.png', dpi=150)
print("\n脑图已保存为 fmri_glm_sub001.png")

# ============================================================
# 保存z-map以便后续group分析使用
# ============================================================

# 把单被试的z-map保存下来，后面做group分析时要用
output_dir = 'first_level_results'
os.makedirs(output_dir, exist_ok=True)

z_map_gain.to_filename(os.path.join(output_dir, f'{subject}_gain_zmap.nii.gz'))
z_map_loss.to_filename(os.path.join(output_dir, f'{subject}_loss_zmap.nii.gz'))

print(f"z-maps保存到 {output_dir}/")
print("\n完成！")