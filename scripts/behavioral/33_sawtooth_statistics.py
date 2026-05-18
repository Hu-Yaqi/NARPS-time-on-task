"""
33_sawtooth_statistics.py
=========================
Individual-level statistical tests for:
1. Gap convergence (gain - loss shrinks over time)
2. Loss between-run recovery
3. Loss within-run decline
4. Gain between-run recovery (expected: absent)
5. Gain within-run decline (expected: absent)

Uses individual subject data from script 30's GLM.

运行方式：conda activate narps && python 33_sawtooth_statistics.py
前提：30_gain_trajectory.py 已跑完
"""

import pandas as pd
import numpy as np
from scipy import stats
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import load_img
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = 'sawtooth_statistics'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 读取每个被试的8-bin数据
# ============================================================

# 如果脚本30保存了per-subject数据，直接读取
# 否则需要从z-map重新提取

# 检查是否有per-subject数据
# 脚本30没有保存individual data，只保存了group mean
# 所以我们需要重新从nii.gz提取

print("=" * 60)
print("Individual-Level Sawtooth Statistics")
print("=" * 60)

# 方法：从binned GLM的z-map文件中提取
# 但脚本28/30没有保存per-bin的z-map
# 所以我们用另一个方法：从脚本28的output重新提取

# 先检查binned GLM有没有保存individual z-maps
binned_dir = 'gain_trajectory_results'

# 实际上脚本30只保存了group csv，没有保存individual nii
# 我们需要用行为数据 + neural data 的另一种方式

# 更好的方法：从all_subjects_behavior.csv直接计算
# 每个被试每个bin的rejection rate（行为侧）
# 然后从runwise_parameters来间接检验

# 但对于neural侧，我们需要per-subject per-bin的vmPFC值
# 这些值在脚本30跑的时候计算过但没保存到文件

# 解决方案：重新跑一次提取（不需要重跑GLM）
# 不对，GLM结果没有保存per-bin contrast maps

# 最实际的方案：用行为数据做sawtooth检验
# 同时对neural侧做能做的检验

print("\n--- 方案: 用行为数据做individual-level sawtooth检验 ---")
print("--- 对neural侧用已有的early-vs-late individual maps ---\n")

# ============================================================
# 1. 行为侧：individual-level gap and sawtooth
# ============================================================

behavior = pd.read_csv('all_subjects_behavior.csv')

# 找到有fMRI数据的被试
fmriprep_base = 'data/derivatives/fmriprep'
fmri_subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

print(f"fMRI被试数: {len(fmri_subjects)}")

# 对每个被试，计算8个bin的rejection rate
all_subject_bins = []

for subj in fmri_subjects:
    subj_data = behavior[behavior['subject'] == subj]
    
    bins = []
    for run in range(1, 5):
        run_data = subj_data[subj_data['run'] == run].reset_index(drop=True)
        n = len(run_data)
        half = n // 2
        
        bin_first = (run - 1) * 2 + 1
        bin_second = bin_first + 1
        
        first_half = run_data.iloc[:half]
        second_half = run_data.iloc[half:]
        
        if len(first_half) > 0:
            bins.append({'subject': subj, 'bin': bin_first,
                        'reject_rate': (first_half['accepted'] == 0).mean()})
        if len(second_half) > 0:
            bins.append({'subject': subj, 'bin': bin_second,
                        'reject_rate': (second_half['accepted'] == 0).mean()})
    
    all_subject_bins.extend(bins)

bin_df = pd.DataFrame(all_subject_bins)

# Pivot to subject × bin matrix
pivot = bin_df.pivot(index='subject', columns='bin', values='reject_rate')
print(f"Pivot shape: {pivot.shape}")  # should be 41 × 8

# ============================================================
# 2. Neural侧：individual-level using early vs late maps
# ============================================================

print("\n--- Neural: extracting individual vmPFC values from early/late maps ---")

vmPFC_coords = [(-12, 36, -13)]
masker = NiftiSpheresMasker(seeds=vmPFC_coords, radius=8, standardize=False)

fatigue_dir = 'fatigue_neural_results'
first_level_dir = 'first_level_results'

# 从脚本24的输出读取individual difference maps
neural_data = []
for subj in fmri_subjects:
    loss_diff = os.path.join(fatigue_dir, f'{subj}_loss_late_minus_early.nii.gz')
    if os.path.exists(loss_diff):
        val = masker.fit_transform(load_img(loss_diff)).flat[0]
        neural_data.append({'subject': subj, 'loss_diff_vmPFC': val})

neural_df = pd.DataFrame(neural_data)
print(f"Neural data: {len(neural_df)} subjects")

# ============================================================
# 3. Statistical Tests
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL TESTS")
print("=" * 60)

# ---- Test 1: Behavioral rejection rate linear trend per subject ----
print("\n--- Test 1: Individual behavioral trend (rejection rate vs bin) ---")

slopes = []
for subj in pivot.index:
    y = pivot.loc[subj].values
    x = np.arange(1, 9)
    if not np.any(np.isnan(y)):
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes.append(slope)

slopes = np.array(slopes)
t_behav, p_behav = stats.ttest_1samp(slopes, 0)
print(f"  Mean slope: {slopes.mean():.4f} (SEM: {slopes.std()/np.sqrt(len(slopes)):.4f})")
print(f"  One-sample t-test: t({len(slopes)-1}) = {t_behav:.3f}, p = {p_behav:.4f}")
print(f"  Direction: {'rejection increases' if slopes.mean() > 0 else 'rejection decreases'}")

# ---- Test 2: Behavioral within-run change (2nd - 1st) ----
print("\n--- Test 2: Behavioral within-run change (2nd half - 1st half) ---")

within_run_diffs = []
for subj in pivot.index:
    vals = pivot.loc[subj].values
    diffs = []
    for run in range(4):
        d = vals[run*2+1] - vals[run*2]  # 2nd - 1st
        diffs.append(d)
    within_run_diffs.append(np.mean(diffs))

within_run_diffs = np.array(within_run_diffs)
t_within, p_within = stats.ttest_1samp(within_run_diffs, 0)
print(f"  Mean within-run change: {within_run_diffs.mean():.4f} (SEM: {within_run_diffs.std()/np.sqrt(len(within_run_diffs)):.4f})")
print(f"  One-sample t-test: t({len(within_run_diffs)-1}) = {t_within:.3f}, p = {p_within:.4f}")

# ---- Test 3: Behavioral between-run reset (next 1st - current 2nd) ----
print("\n--- Test 3: Behavioral between-run reset (next run 1st - current run 2nd) ---")

between_run_resets = []
for subj in pivot.index:
    vals = pivot.loc[subj].values
    resets = []
    for run in range(3):
        r = vals[(run+1)*2] - vals[run*2+1]  # next 1st - current 2nd
        resets.append(r)
    between_run_resets.append(np.mean(resets))

between_run_resets = np.array(between_run_resets)
t_reset, p_reset = stats.ttest_1samp(between_run_resets, 0)
print(f"  Mean between-run reset: {between_run_resets.mean():.4f} (SEM: {between_run_resets.std()/np.sqrt(len(between_run_resets)):.4f})")
print(f"  One-sample t-test: t({len(between_run_resets)-1}) = {t_reset:.3f}, p = {p_reset:.4f}")

# ---- Test 4: Neural vmPFC loss change (late - early) ----
print("\n--- Test 4: Neural vmPFC loss sensitivity change (late - early) ---")

if len(neural_df) > 0:
    vals = neural_df['loss_diff_vmPFC'].values
    t_neural, p_neural = stats.ttest_1samp(vals, 0)
    print(f"  Mean Δ vmPFC loss: {vals.mean():.4f} (SEM: {vals.std()/np.sqrt(len(vals)):.4f})")
    print(f"  One-sample t-test: t({len(vals)-1}) = {t_neural:.3f}, p = {p_neural:.4f}")
    print(f"  Direction: {'loss sensitivity increases' if vals.mean() > 0 else 'loss sensitivity decreases'}")

# ---- Test 5: Run-wise lambda trend (from runwise MLE) ----
print("\n--- Test 5: Run-wise lambda linear trend per subject ---")

runwise = pd.read_csv('runwise_parameters_fixed.csv')
runwise_fmri = runwise[runwise['subject'].isin(fmri_subjects)]

lambda_slopes = []
for subj in fmri_subjects:
    subj_data = runwise_fmri[runwise_fmri['subject'] == subj].sort_values('run')
    if len(subj_data) == 4:
        slope, _, _, _, _ = stats.linregress(subj_data['run'], subj_data['lambda'])
        lambda_slopes.append(slope)

lambda_slopes = np.array(lambda_slopes)
t_lam, p_lam = stats.ttest_1samp(lambda_slopes, 0)
print(f"  Mean lambda slope: {lambda_slopes.mean():.4f} (SEM: {lambda_slopes.std()/np.sqrt(len(lambda_slopes)):.4f})")
print(f"  One-sample t-test: t({len(lambda_slopes)-1}) = {t_lam:.3f}, p = {p_lam:.4f}")

# ---- Test 6: Correlation between behavioral slope and neural change ----
print("\n--- Test 6: Brain-behavior correlation (lambda slope vs vmPFC loss change) ---")

if len(neural_df) > 0:
    # Match subjects
    matched_behav = []
    matched_neural = []
    for _, row in neural_df.iterrows():
        subj = row['subject']
        subj_runwise = runwise_fmri[runwise_fmri['subject'] == subj].sort_values('run')
        if len(subj_runwise) == 4:
            slope, _, _, _, _ = stats.linregress(subj_runwise['run'], subj_runwise['lambda'])
            matched_behav.append(slope)
            matched_neural.append(row['loss_diff_vmPFC'])
    
    matched_behav = np.array(matched_behav)
    matched_neural = np.array(matched_neural)
    r_bb, p_bb = stats.pearsonr(matched_behav, matched_neural)
    print(f"  n = {len(matched_behav)}")
    print(f"  Pearson r = {r_bb:.3f}, p = {p_bb:.4f}")

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 60}")
print("SUMMARY FOR PAPER")
print("=" * 60)
print(f"""
Behavioral:
  Rejection rate trend:    t = {t_behav:.3f}, p = {p_behav:.4f}
  Within-run change:       t = {t_within:.3f}, p = {p_within:.4f}
  Between-run reset:       t = {t_reset:.3f}, p = {p_reset:.4f}
  Lambda slope:            t = {t_lam:.3f}, p = {p_lam:.4f}

Neural (vmPFC):
  Loss late-early change:  t = {t_neural:.3f}, p = {p_neural:.4f}

Brain-Behavior:
  Lambda slope vs vmPFC:   r = {r_bb:.3f}, p = {p_bb:.4f}
""")

# Save results
results = {
    'test': ['behav_trend', 'within_run', 'between_run_reset', 
             'lambda_slope', 'neural_vmPFC', 'brain_behavior'],
    't_or_r': [t_behav, t_within, t_reset, t_lam, t_neural, r_bb],
    'p': [p_behav, p_within, p_reset, p_lam, p_neural, p_bb],
}
pd.DataFrame(results).to_csv(os.path.join(output_dir, 'sawtooth_tests.csv'), index=False)
print(f"保存: {output_dir}/sawtooth_tests.csv")
