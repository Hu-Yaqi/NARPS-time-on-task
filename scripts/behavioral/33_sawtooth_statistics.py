"""
33_sawtooth_statistics.py
========================
Individual-level statistical tests for behavioral sawtooth patterns:
rejection rate trend, within-run change, between-run reset,
lambda slope, neural vmPFC change, and brain-behavior correlation.

Outputs:
  - sawtooth_statistics/sawtooth_tests.csv
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
# ============================================================



print("=" * 60)
print("Individual-Level Sawtooth Statistics")
print("=" * 60)


binned_dir = 'gain_trajectory_results'






print("\n--- Using behavioral data for individual-level sawtooth tests ---")
print("--- Neural side uses existing early/late difference maps ---\n")

# ============================================================
# ============================================================

behavior = pd.read_csv('all_subjects_behavior.csv')

fmriprep_base = 'data/derivatives/fmriprep'
fmri_subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

print(f"fMRIN subjects: {len(fmri_subjects)}")

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
# ============================================================

print("\n--- Neural: extracting individual vmPFC values from early/late maps ---")

vmPFC_coords = [(-12, 36, -13)]
masker = NiftiSpheresMasker(seeds=vmPFC_coords, radius=8, standardize=False)

fatigue_dir = 'fatigue_neural_results'
first_level_dir = 'first_level_results'

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
print(f"Saved: {output_dir}/sawtooth_tests.csv")
