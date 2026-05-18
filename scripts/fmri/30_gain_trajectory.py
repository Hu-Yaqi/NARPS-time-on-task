"""
30_gain_trajectory.py
=====================
8-bin gain + loss trajectory analysis. Extends script 28 by adding
8 gain-bin regressors alongside the 8 loss-bin regressors, then
extracts vmPFC values for both to visualize gain vs loss convergence.

Outputs:
  - gain_trajectory_results/trajectory_vmPFC.csv
  - gain_trajectory_results/gain_vs_loss_vmPFC.png
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

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'gain_trajectory_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0
N_BINS = 8

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])
print(f"Found {len(subjects)}  subjects")

rois = {'vmPFC': (-12, 36, -13)}


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
print("Gain + Loss 8-bin GLM")
print("=" * 60)

gain_traj = {k: [] for k in rois}
loss_traj = {k: [] for k in rois}
maskers = {k: NiftiSpheresMasker(seeds=[v], radius=8, standardize=False)
           for k, v in rois.items()}

successful, failed = [], []
t0 = time.time()

for subj in subjects:
    print(f"\n{'─'*50}\nProcessing {subj} ...")
    ts = time.time()
    fdir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fdir):
        failed.append((subj, "no fMRIPrep")); continue

    try:
        imgs, evts, confs = [], [], []
        for run in range(1, 5):
            rs = f'{run:02d}'
            imgs.append(os.path.join(fdir,
                f'{subj}_task-MGT_run-{rs}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'))
            evts.append(prepare_events(
                os.path.join(data_dir, subj, 'func', f'{subj}_task-MGT_run-{rs}_events.tsv'), run))
            confs.append(prepare_confounds(
                os.path.join(fdir, f'{subj}_task-MGT_run-{rs}_bold_confounds.tsv')))
            for f in [imgs[-1]]:
                if not os.path.exists(f): raise FileNotFoundError(f)

        glm = FirstLevelModel(t_r=TR, hrf_model='spm', drift_model='cosine',
                              high_pass=0.01, smoothing_fwhm=6, minimize_memory=True)
        glm.fit(imgs, evts, confs)

        gv = {k: [] for k in rois}
        lv = {k: [] for k in rois}
        for b in range(1, 9):
            zg = glm.compute_contrast(f'gain_bin{b}', output_type='z_score')
            zl = glm.compute_contrast(f'loss_bin{b}', output_type='z_score')
            for roi, m in maskers.items():
                gv[roi].append(m.fit_transform(zg).flat[0])
                lv[roi].append(m.fit_transform(zl).flat[0])

        for roi in rois:
            gain_traj[roi].append(gv[roi])
            loss_traj[roi].append(lv[roi])

        print(f"  ✓ Done（{time.time()-ts:.0f} s）")
        successful.append(subj)
    except Exception as e:
        print(f"  ✗ failed: {e}")
        failed.append((subj, str(e)))

n = len(successful)
for roi in rois:
    gain_traj[roi] = np.array(gain_traj[roi])
    loss_traj[roi] = np.array(loss_traj[roi])

# ============================================================
# ============================================================

x = np.arange(1, 9)
labels = ['R1\nfirst','R1\nsecond','R2\nfirst','R2\nsecond',
          'R3\nfirst','R3\nsecond','R4\nfirst','R4\nsecond']

for roi in rois:
    gm = np.nanmean(gain_traj[roi], axis=0)
    gs = np.nanstd(gain_traj[roi], axis=0) / np.sqrt(n)
    lm = np.nanmean(loss_traj[roi], axis=0)
    ls = np.nanstd(loss_traj[roi], axis=0) / np.sqrt(n)

    _, _, rg, pg, _ = stats.linregress(x, gm)
    _, _, rl, pl, _ = stats.linregress(x, lm)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, lm, yerr=ls, color='#E74C3C', marker='o', markersize=8,
                linewidth=2.5, capsize=4, label=f'Loss (r={rl:.3f}, p={pl:.4f})')
    ax.errorbar(x, gm, yerr=gs, color='#2E86C1', marker='s', markersize=8,
                linewidth=2.5, capsize=4, label=f'Gain (r={rg:.3f}, p={pg:.4f})')

    for b in [2.5, 4.5, 6.5]:
        ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
    for i, lab in enumerate(['Run 1','Run 2','Run 3','Run 4']):
        ax.text(i*2+1.5, ax.get_ylim()[1], lab, ha='center', va='bottom',
                fontsize=11, color='gray', fontweight='bold')

    ax.axhline(0, color='gray', linestyle='-', alpha=0.2)
    ax.set_xlabel('Time Bin', fontsize=13)
    ax.set_ylabel(f'{roi} sensitivity (z-score)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f'Gain vs Loss Sensitivity in {roi} (n={n})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'gain_vs_loss_{roi}.png'), dpi=150)
    print(f"Saved: gain_vs_loss_{roi}.png")
    plt.close()

    pd.DataFrame({'bin': x, 'gain_mean': gm, 'gain_sem': gs,
                   'loss_mean': lm, 'loss_sem': ls}).to_csv(
        os.path.join(output_dir, f'trajectory_{roi}.csv'), index=False)

elapsed = time.time() - t0
print(f"\n{'='*60}\nDone！elapsed: {elapsed/60:.1f} min\nsubjects: {n}")
for roi in rois:
    gm = np.nanmean(gain_traj[roi], axis=0)
    lm = np.nanmean(loss_traj[roi], axis=0)
    _, _, rg, pg, _ = stats.linregress(x, gm)
    _, _, rl, pl, _ = stats.linregress(x, lm)
    print(f"  {roi}: Gain r={rg:.3f} p={pg:.4f} | Loss r={rl:.3f} p={pl:.4f}")