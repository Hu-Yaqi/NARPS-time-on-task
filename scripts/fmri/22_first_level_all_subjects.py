"""
22_first_level_all_subjects.py
==============================
First-level GLM for all subjects. Each subject's BOLD signal is
modeled with gamble onset, demeaned gain and loss parametric
modulators, motion confounds, and cosine drift regressors.
Produces gain and loss z-maps per subject.

Outputs:
  - first_level_results/{subject}_gain_zmap.nii.gz
  - first_level_results/{subject}_loss_zmap.nii.gz
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================
# ============================================================

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'first_level_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

import os
subjects = sorted([d for d in os.listdir(os.path.join('data', 'derivatives', 'fmriprep'))
                   if d.startswith('sub-')])

# ============================================================
# ============================================================

def prepare_events_for_run(events_file):
    """
    Read events file and construct gamble + gain_mod + loss_mod event format.
    

    Args:
        events_file: path to raw events TSV
    Returns:
        events_df: processed events DataFrame with onset, duration, trial_type, modulation
    """
    raw_events = pd.read_csv(events_file, sep='\t')
    raw_events = raw_events[raw_events['participant_response'] != 'NoResp'].copy()

    rows = []
    for _, trial in raw_events.iterrows():
        onset = trial['onset']
        duration = trial['duration']
        gain_val = trial['gain']
        loss_val = trial['loss']

        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'gamble',
            'modulation': 1.0
        })

        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'gain_mod',
            'modulation': gain_val
        })

        rows.append({
            'onset': onset,
            'duration': duration,
            'trial_type': 'loss_mod',
            'modulation': loss_val
        })

    events_df = pd.DataFrame(rows)

    gain_mean = events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'].mean()
    loss_mean = events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'].mean()
    events_df.loc[events_df['trial_type'] == 'gain_mod', 'modulation'] -= gain_mean
    events_df.loc[events_df['trial_type'] == 'loss_mod', 'modulation'] -= loss_mean

    return events_df


def prepare_confounds_for_run(confounds_file):
    """
    Read confounds TSV and extract 6 motion parameters.

    Args:
        confounds_file: path to fMRIPrep confounds TSV
    Returns:
        confounds_df: DataFrame with 6 motion parameters
    """
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
print("First-Level GLM — batch processing")
print("=" * 60)

successful = []
failed = []

total_start = time.time()

for subj in subjects:
    print(f"\n{'─' * 50}")
    print(f"Processing {subj} ...")
    subj_start = time.time()

    fmriprep_dir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fmriprep_dir):
        print(f"  ⚠️  {subj}  fMRIPrep directory not found, skipping")
        failed.append((subj, "fMRIPrep directory not found"))
        continue

    gain_out = os.path.join(output_dir, f'{subj}_gain_zmap.nii.gz')
    loss_out = os.path.join(output_dir, f'{subj}_loss_zmap.nii.gz')
    if os.path.exists(gain_out) and os.path.exists(loss_out):
        print(f"  ✓ {subj}  z-maps exist, skipping (delete to re-run)")
        successful.append(subj)
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
                    raise FileNotFoundError(f"{label} file not found: {f}")

            fmri_imgs.append(bold_file)
            events_list.append(prepare_events_for_run(events_file))
            confounds_list.append(prepare_confounds_for_run(confounds_file))

        print(f"  Data prepared (4 runs)")

        glm = FirstLevelModel(
            t_r=TR,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=0.01,
            smoothing_fwhm=6,
            minimize_memory=True,
        )

        glm.fit(fmri_imgs, events_list, confounds_list)
        print(f"  GLM fit complete")

        z_map_gain = glm.compute_contrast('gain_mod', stat_type='t', output_type='z_score')
        z_map_loss = glm.compute_contrast('loss_mod', stat_type='t', output_type='z_score')

        z_map_gain.to_filename(gain_out)
        z_map_loss.to_filename(loss_out)

        elapsed = time.time() - subj_start
        print(f"  ✓ Done! Saved to {output_dir}/  ( {elapsed:.0f} s）")
        successful.append(subj)

    except Exception as e:
        elapsed = time.time() - subj_start
        print(f"  ✗ failed: {e}  ( {elapsed:.0f} s）")
        failed.append((subj, str(e)))

# ============================================================
# ============================================================

total_elapsed = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"All done! Total time: {total_elapsed / 60:.1f} min")
print(f"Succeeded: {len(successful)} subjects: {successful}")
if failed:
    print(f"failed: {len(failed)} subjects:")
    for subj, reason in failed:
        print(f"  {subj}: {reason}")

print(f"\n{output_dir}/ files in directory:")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")