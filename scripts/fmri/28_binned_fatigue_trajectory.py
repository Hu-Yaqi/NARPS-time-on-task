"""
28_binned_fatigue_trajectory.py
===============================
8-bin loss trajectory analysis. Each run is split at its midpoint,
yielding 8 time bins across the session. Bin-specific loss regressors
are included in a single GLM; mean z-stat is extracted from vmPFC.

Outputs:
  - binned_fatigue_results/trajectory_data.csv
  - binned_fatigue_results/*.png
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
print(f"Found {len(subjects)}  subjects")

rois = {
    'dmPFC': (8, 44, 54),
    'vmPFC': (-12, 36, -13),
    'mPFC': (6, 30, 59),
}


# ============================================================
# ============================================================

def prepare_events_binned(events_file, run_number):
    """Build binned event file. All 8 loss bins appear in every run."""
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

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gamble', 'modulation': 1.0
        })

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gain_mod', 'modulation': gain_val
        })

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

    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean

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
# ============================================================

print("=" * 60)
print(f"Binned GLM: {N_BINS} time bins")
print("=" * 60)

roi_trajectories = {name: [] for name in rois}
maskers = {name: NiftiSpheresMasker(seeds=[coords], radius=8, standardize=False)
           for name, coords in rois.items()}

successful = []
failed = []
total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"Processing {subj} ...")
    subj_start = time.time()

    fmriprep_dir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fmriprep_dir):
        print(f"  ⚠️  Skip（no fMRIPrep）")
        failed.append((subj, "no fMRIPrep"))
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
        print(f"  ✓ Done（{elapsed:.0f} s）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ failed: {e}（{elapsed:.0f} s）")
        failed.append((subj, str(e)))

n_subjects = len(successful)
print(f"\nStage 1 complete: {n_subjects} succeeded, {len(failed)} failed")

for roi_name in rois:
    roi_trajectories[roi_name] = np.array(roi_trajectories[roi_name])

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Computing behavioral metrics per bin...")

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

        first_half = run_data.iloc[:half]
        if len(first_half) > 0:
            bin_behavior.append({
                'subject': subj, 'bin': bin_first,
                'reject_rate': (first_half['accepted'] == 0).mean(),
                'n_trials': len(first_half),
            })

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

bin_means = bin_df.groupby('bin').agg(
    reject_mean=('reject_rate', 'mean'),
    reject_sem=('reject_rate', lambda x: x.std() / np.sqrt(len(x))),
).reset_index()

print("\nMean rejection rate per bin:")
for _, row in bin_means.iterrows():
    print(f"  Bin {int(row['bin'])}: {row['reject_mean']:.3f} ± {row['reject_sem']:.3f}")

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Generating figures...")

x = np.arange(1, N_BINS + 1)
bin_labels = ['R1\nfirst', 'R1\nsecond', 'R2\nfirst', 'R2\nsecond',
              'R3\nfirst', 'R3\nsecond', 'R4\nfirst', 'R4\nsecond']

for roi_name in rois:
    data = roi_trajectories[roi_name]  # shape: (n_subjects, N_BINS)

    roi_mean = np.nanmean(data, axis=0)
    roi_sem = np.nanstd(data, axis=0) / np.sqrt(n_subjects)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = '
    ax1.set_xlabel('Time Window', fontsize=13)
    ax1.set_ylabel('Rejection Rate (behavioral)', fontsize=13, color=color1)
    ax1.errorbar(x, bin_means['reject_mean'].values, yerr=bin_means['reject_sem'].values,
                 color=color1, marker='o', markersize=8, linewidth=2.5, capsize=4,
                 label='Rejection rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=10)

    ax2 = ax1.twinx()
    color2 = '
    ax2.set_ylabel(f'{roi_name} loss sensitivity (z-score)', fontsize=13, color=color2)
    ax2.errorbar(x, roi_mean, yerr=roi_sem,
                 color=color2, marker='s', markersize=8, linewidth=2.5, capsize=4,
                 label=f'{roi_name} BOLD')
    ax2.tick_params(axis='y', labelcolor=color2)

    for boundary in [2.5, 4.5, 6.5]:
        ax1.axvline(boundary, color='gray', linestyle='--', alpha=0.3)

    for i, label in enumerate(['Run 1', 'Run 2', 'Run 3', 'Run 4']):
        ax1.text(i * 2 + 1.5, ax1.get_ylim()[1], label,
                 ha='center', va='bottom', fontsize=11, color='gray', fontweight='bold')

    ax1.set_title(f'Behavioral and Neural Loss Sensitivity Across Time\n'
                  f'{roi_name} (n={n_subjects})', fontsize=14, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    slope_b, intercept_b, r_b, p_b, _ = stats.linregress(x, bin_means['reject_mean'].values)
    slope_n, intercept_n, r_n, p_n, _ = stats.linregress(x, roi_mean)

    textstr = (f'Behavioral trend: r={r_b:.3f}, p={p_b:.4f}\n'
               f'Neural trend: r={r_n:.3f}, p={p_n:.4f}')
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'trajectory_{roi_name.replace("/", "_")}.png'), dpi=150)
    print(f"  Saved: trajectory_{roi_name.replace('/', '_')}.png")
    plt.close()

fig_all, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, roi_name in enumerate(rois):
    data = roi_trajectories[roi_name]
    roi_mean = np.nanmean(data, axis=0)
    roi_sem = np.nanstd(data, axis=0) / np.sqrt(n_subjects)

    ax = axes[idx]

    color1 = '#E74C3C'
    ax.errorbar(x, bin_means['reject_mean'].values, yerr=bin_means['reject_sem'].values,
                color=color1, marker='o', markersize=6, linewidth=2, capsize=3, alpha=0.8)
    ax.set_ylabel('Rejection Rate', fontsize=11, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{i}' for i in range(1, 9)], fontsize=9)
    ax.set_xlabel('Time Bin', fontsize=11)

    ax2 = ax.twinx()
    color2 = '#2E86C1'
    ax2.errorbar(x, roi_mean, yerr=roi_sem,
                 color=color2, marker='s', markersize=6, linewidth=2, capsize=3, alpha=0.8)
    ax2.set_ylabel('Loss BOLD (z)', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    for boundary in [2.5, 4.5, 6.5]:
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3)

    slope_n, _, r_n, p_n, _ = stats.linregress(x, roi_mean)
    ax.set_title(f'{roi_name}\nr={r_n:.3f}, p={p_n:.4f}', fontsize=12, fontweight='bold')

plt.suptitle(f'Loss Sensitivity Trajectory: Behavior (red) vs Brain (blue), n={n_subjects}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_all.savefig(os.path.join(output_dir, 'trajectory_all_rois.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: trajectory_all_rois.png")

# ============================================================
# ============================================================

trajectory_data = {'bin': x}
for roi_name in rois:
    data = roi_trajectories[roi_name]
    trajectory_data[f'{roi_name}_mean'] = np.nanmean(data, axis=0)
    trajectory_data[f'{roi_name}_sem'] = np.nanstd(data, axis=0) / np.sqrt(n_subjects)
trajectory_data['reject_rate_mean'] = bin_means['reject_mean'].values
trajectory_data['reject_rate_sem'] = bin_means['reject_sem'].values

traj_df = pd.DataFrame(trajectory_data)
traj_df.to_csv(os.path.join(output_dir, 'trajectory_data.csv'), index=False)
print(f"\nSaved: trajectory_data.csv")

# ============================================================
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Binned GLM analysis done! Total time: {total_elapsed / 60:.1f} min")
print(f"{'=' * 60}")
print(f"N subjects: {n_subjects}")
print(f"\nLinear trend test:")
for roi_name in rois:
    data = roi_trajectories[roi_name]
    roi_mean = np.nanmean(data, axis=0)
    _, _, r, p, _ = stats.linregress(x, roi_mean)
    print(f"  {roi_name}: r={r:.3f}, p={p:.4f}")

slope_b, _, r_b, p_b, _ = stats.linregress(x, bin_means['reject_mean'].values)
print(f"  Behavior (reject rate): r={r_b:.3f}, p={p_b:.4f}")