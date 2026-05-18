"""
27_trial_level_fatigue.py
=========================
Trial-level GLM with loss x trial_number interaction. Tests whether
loss sensitivity increases gradually (linear) rather than abruptly.
Design matrix includes gamble, gain_mod, loss_mod, trial_mod, and
loss_x_trial interaction (all demeaned).

Outputs:
  - trial_level_results/{subject}_{loss,trial,loss_x_trial}_zmap.nii.gz
  - trial_level_results/group_loss_x_trial_zmap.nii.gz
  - trial_level_results/*.png
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
# ============================================================

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'trial_level_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])
print(f"Found {len(subjects)}  subjects")


# ============================================================
# ============================================================

def prepare_events_trial_level(events_file, run_number):
    """
    Build event file with trial_number and loss x trial interaction.

    Args:
        events_file: path to raw events TSV
        run_number: 1-4, used to compute global trial number
    Returns:
        events_df: DataFrame with 5 trial_type columns
    """
    raw_events = pd.read_csv(events_file, sep='\t')
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()
    raw_events = raw_events.reset_index(drop=True)

    base_trial = (run_number - 1) * 64

    rows = []
    for i, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']
        trial_num = base_trial + i + 1

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gamble', 'modulation': 1.0
        })

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'gain_mod', 'modulation': gain_val
        })

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'loss_mod', 'modulation': loss_val
        })

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'trial_mod', 'modulation': float(trial_num)
        })

        rows.append({
            'onset': onset, 'duration': duration,
            'trial_type': 'loss_x_trial', 'modulation': loss_val * trial_num
        })

    events_df = pd.DataFrame(rows)

    for tt in ['gain_mod', 'loss_mod', 'trial_mod', 'loss_x_trial']:
        mean_val = events_df.loc[events_df['trial_type'] == tt, 'modulation'].mean()
        events_df.loc[events_df['trial_type'] == tt, 'modulation'] -= mean_val

    return events_df


def prepare_confounds(confounds_file):
    """Read confounds and extract 6 motion parameters."""
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
print("Trial-level time-on-task analysis")
print("GLM with loss x trial_number interaction")
print("=" * 60)

loss_x_trial_maps = []
loss_maps = []
trial_maps = []

successful = []
failed = []
total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"Processing {subj} ...")
    subj_start = time.time()

    out_file = os.path.join(output_dir, f'{subj}_loss_x_trial_zmap.nii.gz')
    if os.path.exists(out_file):
        from nilearn.image import load_img

        loss_x_trial_maps.append(load_img(out_file))
        loss_maps.append(load_img(os.path.join(output_dir, f'{subj}_loss_zmap.nii.gz')))
        trial_maps.append(load_img(os.path.join(output_dir, f'{subj}_trial_zmap.nii.gz')))
        print(f"  ✓ Exists, skipping")
        successful.append(subj)
        continue

    fmriprep_dir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fmriprep_dir):
        print(f"  ⚠️  fMRIPrep directory not found，Skip")
        failed.append((subj, "fMRIPrep directory not found"))
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

        glm = FirstLevelModel(
            t_r=TR,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=0.01,
            smoothing_fwhm=6,
            minimize_memory=True,
        )

        glm.fit(fmri_imgs, events_list, confounds_list)

        dm_cols = glm.design_matrices_[0].columns.tolist()
        print(f"  Design matrix columns: {[c for c in dm_cols if not c.startswith('drift') and c != 'constant']}")

        z_loss = glm.compute_contrast('loss_mod', output_type='z_score')
        z_trial = glm.compute_contrast('trial_mod', output_type='z_score')
        z_loss_x_trial = glm.compute_contrast('loss_x_trial', output_type='z_score')

        z_loss.to_filename(os.path.join(output_dir, f'{subj}_loss_zmap.nii.gz'))
        z_trial.to_filename(os.path.join(output_dir, f'{subj}_trial_zmap.nii.gz'))
        z_loss_x_trial.to_filename(os.path.join(output_dir, f'{subj}_loss_x_trial_zmap.nii.gz'))

        loss_maps.append(z_loss)
        trial_maps.append(z_trial)
        loss_x_trial_maps.append(z_loss_x_trial)

        elapsed = time.time() - subj_start
        print(f"  ✓ Done（{elapsed:.0f} s）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ failed: {e}（{elapsed:.0f} s）")
        failed.append((subj, str(e)))

print(f"\nStage 1 complete: {len(successful)} succeeded, {len(failed)} failed")

# ============================================================
# ============================================================

n_subjects = len(loss_x_trial_maps)
print(f"\n{'=' * 60}")
print(f"Group-level analysis（n={n_subjects}）")
print(f"{'=' * 60}")

design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})
cluster_threshold = 2.3

print("\nAnalyzing loss x trial interaction...")
print("Positive = increasing loss sensitivity over trials")

sl_interaction = SecondLevelModel(smoothing_fwhm=None)
sl_interaction.fit(loss_x_trial_maps, design_matrix=design_matrix)
z_group_interaction = sl_interaction.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

print("Analyzing trial main effect...")
print("Positive = increasing activation over trials (non-specific)")

sl_trial = SecondLevelModel(smoothing_fwhm=None)
sl_trial.fit(trial_maps, design_matrix=design_matrix)
z_group_trial = sl_trial.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Cluster report")

print("\n===== LOSS x TRIAL interaction (core result)=====")
print("Positive = loss sensitivity increases linearly over trials")
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
        print("  No significant clusters found")
except Exception as e:
    print(f"  Cluster extraction error: {e}")

print("\n===== TRIAL main effect =====")
print("Positive = non-specific time effect")
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
        print("  No significant clusters found")
except Exception as e:
    print(f"  Cluster extraction error: {e}")

# ============================================================
# ============================================================

print(f"\n{'─' * 40}")
print("Generating figures...")

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
print("  Saved: trial_level_three_effects.png")

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
print("  Saved: loss_x_trial_glass_brain.png")

# ============================================================
# ============================================================

z_group_interaction.to_filename(os.path.join(output_dir, 'group_loss_x_trial_zmap.nii.gz'))
z_group_trial.to_filename(os.path.join(output_dir, 'group_trial_zmap.nii.gz'))
z_group_loss.to_filename(os.path.join(output_dir, 'group_loss_zmap.nii.gz'))

# ============================================================
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Trial-level analysis done! Total time: {total_elapsed / 60:.1f} min")
print(f"{'=' * 60}")
print(f"N subjects: {n_subjects}")
print(f"\nInterpretation guide:")
print(f"  Three rows show:")
print(f"    1. Loss main effect — regions encoding loss magnitude")
print(f"    2. Trial main effect — non-specific time trend")
print(f"    3. Loss x Trial — regions with linearly increasing loss sensitivity")
print(f"\n  If row 3 shows significant clusters:")
print(f"    → Loss sensitivity increase is gradual and linear")
print(f"    → Supports gradual fatigue / strategy adjustment")
print(f"  If row 3 shows no significant clusters:")
print(f"    → Change may be abrupt
print(f"    → Supports strategy switching")