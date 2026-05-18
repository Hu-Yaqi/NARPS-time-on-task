"""
24_fatigue_neural_contrast.py
=============================
Early (Runs 1-2) vs late (Runs 3-4) neural contrast. Separate
GLMs are fit per half; the difference maps (late minus early) are
submitted to group-level one-sample t-tests.

Outputs:
  - fatigue_neural_results/{subject}_{gain,loss}_late_minus_early.nii.gz
  - fatigue_neural_results/group_{gain,loss}_fatigue_zmap.nii.gz
  - fatigue_neural_results/*.png
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
# ============================================================

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'fatigue_neural_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

import os
subjects = sorted([d for d in os.listdir(os.path.join('data', 'derivatives', 'fmriprep'))
                   if d.startswith('sub-')])

# ============================================================
# ============================================================

def prepare_events_for_run(events_file):
    """Read events and build gamble + gain_mod + loss_mod format."""
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

    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    loss_mean = events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean
    events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'] -= loss_mean

    return events_df


def prepare_confounds_for_run(confounds_file):
    """Read confounds and extract 6 motion parameters."""
    confounds = pd.read_csv(confounds_file, sep='\t')
    available = confounds.columns.tolist()
    if 'X' in available:
        motion_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    else:
        motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    return confounds[motion_cols].fillna(0)


def fit_half_glm(subject, runs, label):
    """
    Fit GLM for specified runs; return gain and loss z-maps.

    Args:
        subject: subject ID, e.g., 'sub-001'
        runs: list of run numbers (e.g., [1, 2])
        label: label for logging
    Returns:
        (z_map_gain, z_map_loss): tuple of (gain_zmap, loss_zmap) NIfTI images
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
# ============================================================

print("=" * 60)
print("Part D: Neural signatures of time-on-task")
print("Run 1-2 (early) vs Run 3-4 (late)")
print("=" * 60)

diff_gain_maps = []
diff_loss_maps = []

successful = []
failed = []
total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"Processing {subj} ...")
    subj_start = time.time()

    try:
        print(f"  Fitting early GLM (Run 1-2)...")
        z_gain_early, z_loss_early = fit_half_glm(subj, [1, 2], 'early')

        print(f"  Fitting late GLM (Run 3-4)...")
        z_gain_late, z_loss_late = fit_half_glm(subj, [3, 4], 'late')

        diff_gain = math_img('img1 - img2', img1=z_gain_late, img2=z_gain_early)
        diff_loss = math_img('img1 - img2', img1=z_loss_late, img2=z_loss_early)

        diff_gain.to_filename(os.path.join(output_dir, f'{subj}_gain_late_minus_early.nii.gz'))
        diff_loss.to_filename(os.path.join(output_dir, f'{subj}_loss_late_minus_early.nii.gz'))

        diff_gain_maps.append(diff_gain)
        diff_loss_maps.append(diff_loss)

        elapsed = time.time() - subj_start
        print(f"  ✓ Done( {elapsed:.0f} s）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ failed: {e}( {elapsed:.0f} s）")
        failed.append((subj, str(e)))

print(f"\nStage 1 complete: {len(successful)} succeeded, {len(failed)} failed")
if failed:
    for s, r in failed:
        print(f"  {s}: {r}")

# ============================================================
# ============================================================

print(f"\n{'=' * 60}")
print("Group analysis: Late - Early difference")
print(f"{'=' * 60}")

n_subjects = len(diff_gain_maps)
design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})

print("\nAnalyzing difference (late - early)...")
second_level_gain = SecondLevelModel(smoothing_fwhm=None)
second_level_gain.fit(diff_gain_maps, design_matrix=design_matrix)
z_group_gain_diff = second_level_gain.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

print("Analyzing difference (late - early)...")
second_level_loss = SecondLevelModel(smoothing_fwhm=None)
second_level_loss.fit(diff_loss_maps, design_matrix=design_matrix)
z_group_loss_diff = second_level_loss.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

# ============================================================
# ============================================================

cluster_threshold = 2.3

print(f"\n{'─' * 40}")
print("Cluster report")

print("\n===== GAIN difference (late - early) =====")
print("Positive = late > early; Negative = late < early")
try:
    gain_diff_table = get_clusters_table(
        z_group_gain_diff,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(gain_diff_table) > 0:
        print(gain_diff_table.head(20).to_string())
        gain_diff_table.to_csv(os.path.join(output_dir, 'gain_fatigue_clusters.csv'), index=False)
    else:
        print("  No significant clusters found")
except Exception as e:
    print(f"  Cluster extraction error: {e}")

print("\n===== LOSS difference (late - early) =====")
print("Positive = late > early; Negative = late < early")
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
        print("  No significant clusters found")
except Exception as e:
    print(f"  Cluster extraction error: {e}")

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Generating figures...")

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
print("  Saved: fatigue_neural_contrast.png")

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
print("  Saved: loss_fatigue_glass_brain.png")

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
    print("  Saved: loss_overall_vs_fatigue.png")

# ============================================================
# ============================================================

z_group_gain_diff.to_filename(os.path.join(output_dir, 'group_gain_fatigue_zmap.nii.gz'))
z_group_loss_diff.to_filename(os.path.join(output_dir, 'group_loss_fatigue_zmap.nii.gz'))

# ============================================================
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Part D complete! Total time: {total_elapsed/60:.1f} min")
print(f"{'=' * 60}")
print(f"N subjects: {n_subjects}")
print(f"Results saved in: {output_dir}/")
print(f"\nInterpretation guide:")
print(f"  fatigue_neural_contrast.png:")
print(f"    Top = gain change (expected weak)")
print(f"    Bottom = loss time-on-task change (core result)")
print(f"    Red = late > early (increase)")
print(f"    Blue = late < early (decrease)")
print(f"\n  Correspondence with behavior:")
print(f"    Behavioral lambda up -> expect red in loss regions")
print(f"    Behavioral alpha -> 1 -> gain regions may not change")
print(f"\n  loss_overall_vs_fatigue.png:")
print(f"    Comparing overall loss effect vs time-on-task change")
print(f"    Same region in both: encodes loss AND changes with time")