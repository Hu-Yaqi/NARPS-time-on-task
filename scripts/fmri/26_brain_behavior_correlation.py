"""
26_brain_behavior_correlation.py
================================
Brain-behavior correlation: correlates individual delta-lambda
(late minus early run-wise MLE) with delta-activation in vmPFC
and seven additional ROIs. Also runs whole-brain voxel-level
association with delta-lambda as a covariate.

Outputs:
  - brain_behavior_results/vmpfc_brain_behavior_scatter.png
  - brain_behavior_results/roi_brain_behavior_correlations.csv
  - brain_behavior_results/brain_behavior_wholebrain_zmap.nii.gz
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
# ============================================================

fatigue_dir = 'fatigue_neural_results'
output_dir = 'brain_behavior_results'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# ============================================================

print("=" * 60)
print("Brain-behavior correlation")
print("=" * 60)

runwise = pd.read_csv('runwise_parameters_fixed.csv')
print(f"\nBehavioral data: {runwise.shape[0]} rows")
print(f"Columns: {runwise.columns.tolist()}")

print(f"\nFirst rows:")
print(runwise.head())

lambda_col = None
for candidate in ['lambda', 'lam', 'loss_aversion', 'λ']:
    if candidate in runwise.columns:
        lambda_col = candidate
        break

if lambda_col is None:
    for col in runwise.columns:
        if 'lam' in col.lower() or 'loss' in col.lower():
            lambda_col = col
            break

if lambda_col is None:
    print("\n⚠️  Cannot find lambda column! Check runwise_parameters_fixed.csv columns.")
    print(f"Available columns: {runwise.columns.tolist()}")
    print("Will try the first parameter-like column...")
    import sys

    sys.exit(1)

print(f"\nUsing column as loss aversion parameter")

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
    print(f"⚠️  Cannot find subject or run column.Available columns: {runwise.columns.tolist()}")
    import sys

    sys.exit(1)

print(f"Subject column: '{subj_col}', Run column: '{run_col}'")

delta_lambda = {}
for subj in runwise[subj_col].unique():
    subj_data = runwise[runwise[subj_col] == subj]
    runs = subj_data[run_col].values

    early_runs = subj_data[subj_data[run_col].isin([1, 2])][lambda_col].values
    late_runs = subj_data[subj_data[run_col].isin([3, 4])][lambda_col].values

    if len(early_runs) >= 1 and len(late_runs) >= 1:
        delta_lambda[subj] = np.mean(late_runs) - np.mean(early_runs)

print(f"\nComputed delta-lambda for {len(delta_lambda)} subjects")
print(f"Δλ mean: {np.mean(list(delta_lambda.values())):.3f}")
print(f"Δλ SD: {np.std(list(delta_lambda.values())):.3f}")

# ============================================================
# ============================================================

available_subs = []
fmri_maps = []
lambda_values = []

for subj_id, dlam in delta_lambda.items():
    if isinstance(subj_id, (int, float)):
        subj_str = f'sub-{int(subj_id):03d}'
    elif isinstance(subj_id, str) and subj_id.startswith('sub-'):
        subj_str = subj_id
    else:
        subj_str = f'sub-{int(subj_id):03d}'

    loss_diff_path = os.path.join(fatigue_dir, f'{subj_str}_loss_late_minus_early.nii.gz')
    if os.path.exists(loss_diff_path):
        available_subs.append(subj_str)
        fmri_maps.append(loss_diff_path)
        lambda_values.append(dlam)

n_matched = len(available_subs)
print(f"\nSubjects with both behavioral and fMRI data: {n_matched}")
for s, dl in zip(available_subs, lambda_values):
    print(f"  {s}: Δλ = {dl:+.3f}")

if n_matched < 5:
    print("\n⚠️  Too few matched subjects (<5); correlation unreliable.")
    print("Ensure scripts 22 and 24 have run for all subjects.")

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("ROI analysis: vmPFC")

from nilearn.maskers import NiftiSpheresMasker

vmpfc_coords = [(0, 34, -16)]
vmpfc_masker = NiftiSpheresMasker(
    seeds=vmpfc_coords,
    radius=10,
    standardize=False,
)

vmpfc_values = []
for fmap in fmri_maps:
    vals = vmpfc_masker.fit_transform(fmap)
    vmpfc_values.append(vals.flat[0])

vmpfc_values = np.array(vmpfc_values)
lambda_arr = np.array(lambda_values)

print(f"vmPFC Δactivation mean: {vmpfc_values.mean():.3f}")
print(f"vmPFC Δactivation SD: {vmpfc_values.std():.3f}")

r_vmpfc, p_vmpfc = stats.pearsonr(lambda_arr, vmpfc_values)
r_spearman, p_spearman = stats.spearmanr(lambda_arr, vmpfc_values)

print(f"\nPearson r = {r_vmpfc:.3f}, p = {p_vmpfc:.4f}")
print(f"Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.scatter(lambda_arr, vmpfc_values, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

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

for i, subj in enumerate(available_subs):
    ax.annotate(subj.replace('sub-', ''),
                (lambda_arr[i], vmpfc_values[i]),
                fontsize=8, alpha=0.6,
                xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'vmpfc_brain_behavior_scatter.png'), dpi=150)
print("\nSaved: vmpfc_brain_behavior_scatter.png")

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Multi-ROI analysis")

rois = {
    'vmPFC': (0, 34, -16),
    'L_insula': (-34, 18, -4),
    'R_insula': (34, 18, -4),
    'dACC': (0, 24, 32),
    'L_amygdala': (-22, -4, -18),
    'R_amygdala': (22, -4, -18),
    'ventral_striatum': (0, 10, -6),
    'PCC': (0, -52, 16),
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
print(f"\nSaved: roi_brain_behavior_correlations.csv")

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Whole-brain voxel-level association (exploratory)...")

lambda_demeaned = lambda_arr - lambda_arr.mean()

design_matrix_bb = pd.DataFrame({
    'intercept': np.ones(n_matched),
    'delta_lambda': lambda_demeaned,
})

second_level_bb = SecondLevelModel(smoothing_fwhm=None)
second_level_bb.fit(fmri_maps, design_matrix=design_matrix_bb)

z_map_bb = second_level_bb.compute_contrast(
    second_level_contrast='delta_lambda',
    output_type='z_score'
)

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
print("Saved: brain_behavior_wholebrain.png")

try:
    bb_table = get_clusters_table(z_map_bb, stat_threshold=2.3, min_distance=8)
    if len(bb_table) > 0:
        print(f"\nClusters: delta-lambda vs activation change:")
        print(bb_table.head(15).to_string())
        bb_table.to_csv(os.path.join(output_dir, 'brain_behavior_clusters.csv'), index=False)
    else:
        print("\nNo significant clusters found (may need more subjects)")
except Exception as e:
    print(f"Cluster extraction error: {e}")

# ============================================================
# ============================================================

print(f"\n{'=' * 60}")
print("Brain-behavior correlationDone！")
print(f"{'=' * 60}")
print(f"Matched subjects: {n_matched}")
print(f"\nCore result:")
print(f"  vmPFC: Pearson r = {r_vmpfc:.3f}, p = {p_vmpfc:.4f}")
print(f"\nInterpretation:")
if r_vmpfc > 0 and p_vmpfc < 0.05:
    print("  ✓ Significant positive correlation: behavioral and neural effects are linked
    print("    → Behavioral and neural time-on-task effects are linked at the individual level")
elif r_vmpfc > 0:
    print("  → Positive but not significant — expected direction, insufficient power
    print("    → May need more subjects to reach significance")
else:
    print("  → Expected positive correlation not found")
    print("    → vmPFC may not be the key linking region")
    print("    → Check multi-ROI results; other regions (e.g., insula) may be more relevant")

print(f"\nResults saved in: {output_dir}/")
print(f"\nSuggested paper text:")
print(f"  'To test whether behavioral and neural time-on-task effects")
print(f"   are linked at the individual level, we correlated each")
print(f"   participant's change in loss aversion (Δλ) with the change")
print(f"   in neural sensitivity to loss magnitude in vmPFC.'")