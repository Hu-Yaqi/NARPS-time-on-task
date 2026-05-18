"""
35_other_regions_trajectory.py
==============================
8-bin trajectory extraction for six additional ROIs (L/R insula,
dACC, L amygdala, R IFG, ventral striatum). Tests whether these
regions maintain stable loss coding while vmPFC degrades.

Key result: all six regions show flat loss trajectories (p > .16),
supporting selective vmPFC integration failure.

Outputs:
  - other_regions_results/individual_bin_all_rois.csv
  - other_regions_results/roi_loss_trends_summary.csv
  - other_regions_results/all_rois_loss_trajectory.pdf
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiSpheresMasker
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'other_regions_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

# ROIs: regions that showed positive loss responses in Part C
# and are part of the "fatigue network"
rois = {
    'L_insula':    (-34, 18, -4),    # Left anterior insula
    'R_insula':    (34, 18, -4),     # Right anterior insula
    'dACC':        (8, 44, 54),      # From script 27 loss x trial cluster
    'L_amygdala':  (-22, -4, -18),   # Left amygdala
    'R_IFG':       (48, 22, 13),     # Right IFG - largest cluster in Part C
    'v_striatum':  (0, 10, -6),      # Ventral striatum
}

maskers = {name: NiftiSpheresMasker(seeds=[coords], radius=8, standardize=False)
           for name, coords in rois.items()}


def prepare_events(events_file, run_number):
    raw = pd.read_csv(events_file, sep='\t')
    raw = raw[raw['participant_response'] != 'NoResp'].reset_index(drop=True)
    n = len(raw)
    half = n // 2
    bin_first = (run_number - 1) * 2 + 1
    bin_second = bin_first + 1

    rows = []
    for i, trial in raw.iterrows():
        onset, duration = trial['onset'], trial['duration']
        gain_val, loss_val = trial['gain'], trial['loss']
        b = bin_first if i < half else bin_second

        rows.append({'onset': onset, 'duration': duration,
                     'trial_type': 'gamble', 'modulation': 1.0})
        for bb in range(1, 9):
            rows.append({'onset': onset, 'duration': duration,
                         'trial_type': f'gain_bin{bb}',
                         'modulation': gain_val if bb == b else 0.0})
            rows.append({'onset': onset, 'duration': duration,
                         'trial_type': f'loss_bin{bb}',
                         'modulation': loss_val if bb == b else 0.0})

    df = pd.DataFrame(rows)
    for prefix in ['gain_bin', 'loss_bin']:
        for bb in range(1, 9):
            tt = f'{prefix}{bb}'
            mask = (df['trial_type'] == tt) & (df['modulation'] != 0)
            if mask.sum() > 0:
                df.loc[mask, 'modulation'] -= df.loc[mask, 'modulation'].mean()
    return df


def prepare_confounds(confounds_file):
    c = pd.read_csv(confounds_file, sep='\t')
    cols = ['X','Y','Z','RotX','RotY','RotZ'] if 'X' in c.columns else \
           ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    return c[cols].fillna(0)


print("=" * 60)
print("Extracting 8-bin trajectories for additional ROIs")
print("=" * 60)

all_rows = []
successful = []
t0 = time.time()

for subj in subjects:
    print(f"Processing {subj} ...", end=' ')
    ts = time.time()
    fdir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fdir):
        print("Skip"); continue

    try:
        imgs, evts, confs = [], [], []
        for run in range(1, 5):
            rs = f'{run:02d}'
            bold = os.path.join(fdir,
                f'{subj}_task-MGT_run-{rs}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
            evt = os.path.join(data_dir, subj, 'func',
                f'{subj}_task-MGT_run-{rs}_events.tsv')
            conf = os.path.join(fdir,
                f'{subj}_task-MGT_run-{rs}_bold_confounds.tsv')
            if not os.path.exists(bold):
                raise FileNotFoundError(bold)
            imgs.append(bold)
            evts.append(prepare_events(evt, run))
            confs.append(prepare_confounds(conf))

        glm = FirstLevelModel(t_r=TR, hrf_model='spm', drift_model='cosine',
                              high_pass=0.01, smoothing_fwhm=6, minimize_memory=True)
        glm.fit(imgs, evts, confs)

        for b in range(1, 9):
            zl = glm.compute_contrast(f'loss_bin{b}', output_type='z_score')
            zg = glm.compute_contrast(f'gain_bin{b}', output_type='z_score')

            row = {'subject': subj, 'bin': b}
            for roi_name, masker in maskers.items():
                row[f'{roi_name}_loss'] = masker.fit_transform(zl).flat[0]
                row[f'{roi_name}_gain'] = masker.fit_transform(zg).flat[0]
            all_rows.append(row)

        print(f"✓ ({time.time()-ts:.0f}s)")
        successful.append(subj)
    except Exception as e:
        print(f"✗ {e}")

elapsed = time.time() - t0
n_subj = len(successful)
print(f"\nDone: {n_subj} subjects, {elapsed/60:.1f} min")

ind_df = pd.DataFrame(all_rows)
ind_df.to_csv(os.path.join(output_dir, 'individual_bin_all_rois.csv'), index=False)

# ============================================================
# Statistics and Visualization
# ============================================================

print(f"\n{'=' * 60}")
print("LOSS SENSITIVITY TRENDS BY REGION")
print("=" * 60)

# Also load vmPFC data from script 34
vmPFC_file = 'sawtooth_statistics/individual_bin_vmPFC.csv'
if os.path.exists(vmPFC_file):
    vmPFC_df = pd.read_csv(vmPFC_file)

x = np.arange(1, 9)

# Collect results for summary table
summary = []

# vmPFC (from script 34)
if os.path.exists(vmPFC_file):
    pivot_vm = vmPFC_df.pivot(index='subject', columns='bin', values='loss_vmPFC')
    slopes_vm = []
    for subj in pivot_vm.index:
        s, _, _, _, _ = stats.linregress(x, pivot_vm.loc[subj].values)
        slopes_vm.append(s)
    slopes_vm = np.array(slopes_vm)
    t_vm, p_vm = stats.ttest_1samp(slopes_vm, 0)
    means_vm = [pivot_vm[b].mean() for b in range(1, 9)]
    summary.append({'ROI': 'vmPFC', 'mean_slope': slopes_vm.mean(),
                    't': t_vm, 'p': p_vm, 'direction': 'toward zero (↑)',
                    'bin1': means_vm[0], 'bin8': means_vm[7]})
    print(f"\n  vmPFC: slope={slopes_vm.mean():.4f}, t={t_vm:.3f}, p={p_vm:.4f}")

# Other ROIs
for roi_name in rois:
    col = f'{roi_name}_loss'
    pivot = ind_df.pivot(index='subject', columns='bin', values=col)
    
    slopes = []
    for subj in pivot.index:
        s, _, _, _, _ = stats.linregress(x, pivot.loc[subj].values)
        slopes.append(s)
    slopes = np.array(slopes)
    t_val, p_val = stats.ttest_1samp(slopes, 0)
    means = [pivot[b].mean() for b in range(1, 9)]
    
    direction = '↑' if slopes.mean() > 0 else '↓'
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    
    summary.append({'ROI': roi_name, 'mean_slope': slopes.mean(),
                    't': t_val, 'p': p_val, 'direction': direction,
                    'bin1': means[0], 'bin8': means[7]})
    print(f"  {roi_name:12s}: slope={slopes.mean():.4f}, t={t_val:.3f}, p={p_val:.4f} {sig}  [{means[0]:.3f} → {means[7]:.3f}]")

# Save summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, 'roi_loss_trends_summary.csv'), index=False)

# ============================================================
# Visualization: all ROIs on one figure
# ============================================================

print(f"\nGenerating figures...")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 200,
})

colors = {
    'vmPFC': '#C0392B',      # Red - value encoding (degrades)
    'L_insula': '#2471A3',   # Blue
    'R_insula': '#1A5276',   # Dark blue
    'dACC': '#27AE60',       # Green
    'L_amygdala': '#8E44AD', # Purple
    'R_IFG': '#E67E22',      # Orange
    'v_striatum': '#16A085', # Teal
}

fig, ax = plt.subplots(figsize=(10, 6))

# vmPFC
if os.path.exists(vmPFC_file):
    pivot_vm = vmPFC_df.pivot(index='subject', columns='bin', values='loss_vmPFC')
    means = [pivot_vm[b].mean() for b in range(1, 9)]
    sems = [pivot_vm[b].std() / np.sqrt(n_subj) for b in range(1, 9)]
    ax.errorbar(x, means, yerr=sems, color=colors['vmPFC'],
                marker='o', linewidth=2.5, capsize=3, label='vmPFC', zorder=5)

# Other ROIs
for roi_name in rois:
    col = f'{roi_name}_loss'
    pivot = ind_df.pivot(index='subject', columns='bin', values=col)
    means = [pivot[b].mean() for b in range(1, 9)]
    sems = [pivot[b].std() / np.sqrt(n_subj) for b in range(1, 9)]
    ax.errorbar(x, means, yerr=sems, color=colors.get(roi_name, 'gray'),
                marker='s', linewidth=1.5, capsize=2, label=roi_name, alpha=0.8)

ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

ax.set_xlabel('Time Bin')
ax.set_ylabel('Loss Sensitivity ($z_{stat}$)')
ax.set_xticks(x)
ax.set_xticklabels(['B1\nR1-1st', 'B2\nR1-2nd', 'B3\nR2-1st', 'B4\nR2-2nd',
                     'B5\nR3-1st', 'B6\nR3-2nd', 'B7\nR4-1st', 'B8\nR4-2nd'], fontsize=8)
ax.set_title(f'Loss Sensitivity Trajectories Across Brain Regions (n={n_subj})', fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'all_rois_loss_trajectory.pdf'), format='pdf',
            bbox_inches='tight', dpi=200)
fig.savefig(os.path.join(output_dir, 'all_rois_loss_trajectory.png'),
            bbox_inches='tight', dpi=200)
print(f"Saved: {output_dir}/all_rois_loss_trajectory.pdf / .png")

print(f"\n{'=' * 60}")
print("KEY QUESTION: Do other regions show INCREASING loss sensitivity")
print("while vmPFC shows DECREASING (toward zero)?")
print("If yes → balance shifts from valuation to salience signaling")
print("=" * 60)
