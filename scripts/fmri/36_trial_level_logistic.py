"""
36_trial_level_logistic.py
==========================
Pipeline 2: Trial-Level Logistic Regression

Question: Does the influence of different brain regions on choice 
shift over time? Specifically:
- Does insula/dACC activation become MORE predictive of rejection?
- Does vmPFC activation become LESS predictive of choice?

Method: Extract trial-level BOLD (HRF-adjusted peak) from ROIs,
then logistic regression with ROI × time interactions.

运行方式：conda activate narps && python 36_trial_level_logistic.py
预计耗时：约 30-60 分钟
需要先安装：pip install statsmodels --break-system-packages
"""

import pandas as pd
import numpy as np
from scipy import stats
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import load_img
import os
import time
import warnings
warnings.filterwarnings('ignore')

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'trial_logistic_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0
HRF_DELAY = 5

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

rois = {
    'vmPFC':      (-12, 36, -13),
    'L_insula':   (-34, 18, -4),
    'R_insula':   (34, 18, -4),
    'dACC':       (8, 44, 54),
    'L_amygdala': (-22, -4, -18),
    'R_IFG':      (48, 22, 13),
}

maskers = {name: NiftiSpheresMasker(seeds=[coords], radius=8, standardize=False)
           for name, coords in rois.items()}

# ============================================================
# Step 1: Extract trial-level BOLD
# ============================================================

print("=" * 60)
print("Step 1: Extracting trial-level BOLD signals")
print("=" * 60)

all_trial_data = []
successful = []
t0 = time.time()

for subj in subjects:
    print(f"  {subj} ...", end=' ')
    ts = time.time()
    fdir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fdir):
        print("skip"); continue

    try:
        global_trial = 0
        for run in range(1, 5):
            rs = f'{run:02d}'
            bold_file = os.path.join(fdir,
                f'{subj}_task-MGT_run-{rs}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
            events_file = os.path.join(data_dir, subj, 'func',
                f'{subj}_task-MGT_run-{rs}_events.tsv')
            if not os.path.exists(bold_file):
                raise FileNotFoundError(bold_file)

            bold_img = load_img(bold_file)
            n_vol = bold_img.shape[3]

            roi_ts = {}
            for roi_name, masker in maskers.items():
                roi_ts[roi_name] = masker.fit_transform(bold_img).flatten()

            events = pd.read_csv(events_file, sep='\t')
            events = events[events['participant_response'] != 'NoResp'].reset_index(drop=True)

            for i, trial in events.iterrows():
                global_trial += 1
                peak_vol = int(round(trial['onset'] + HRF_DELAY))
                v0 = max(0, peak_vol - 1)
                v1 = min(n_vol - 1, peak_vol + 1)

                row = {
                    'subject': subj, 'run': run,
                    'global_trial': global_trial,
                    'gain': trial['gain'], 'loss': trial['loss'],
                    'accepted': 1 if trial['participant_response'] in ['strongly_accept', 'weakly_accept'] else 0,
                }
                for roi_name in rois:
                    row[f'{roi_name}_bold'] = np.mean(roi_ts[roi_name][v0:v1+1])
                all_trial_data.append(row)

        print(f"ok ({time.time()-ts:.0f}s)")
        successful.append(subj)
    except Exception as e:
        print(f"fail: {e}")

trial_df = pd.DataFrame(all_trial_data)
trial_df.to_csv(os.path.join(output_dir, 'trial_level_bold_data.csv'), index=False)
print(f"\nSaved: {len(trial_df)} trials from {len(successful)} subjects")

# ============================================================
# Step 2: Z-score and prepare variables
# ============================================================

print("\nStep 2: Preparing variables...")

for roi_name in rois:
    col = f'{roi_name}_bold'
    trial_df[f'{roi_name}_z'] = trial_df.groupby(['subject', 'run'])[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

trial_df['trial_z'] = trial_df.groupby('subject')['global_trial'].transform(
    lambda x: (x - x.mean()) / x.std())
trial_df['reject'] = 1 - trial_df['accepted']
trial_df['late'] = (trial_df['run'] >= 3).astype(int)

# ============================================================
# Step 3: Per-subject logistic regression
# ============================================================

print("\nStep 3: Logistic regressions per subject...")

try:
    from statsmodels.formula.api import logit as logit_model
except ImportError:
    print("Installing statsmodels...")
    import subprocess
    subprocess.run(['pip', 'install', 'statsmodels', '--break-system-packages'], capture_output=True)
    from statsmodels.formula.api import logit as logit_model

results = []
for subj in successful:
    sd = trial_df[trial_df['subject'] == subj].copy()
    try:
        m = logit_model('reject ~ vmPFC_z * trial_z + L_insula_z * trial_z + dACC_z * trial_z + gain + loss',
                        data=sd).fit(disp=0, maxiter=100)
        row = {'subject': subj}
        for p in m.params.index:
            row[f'coef_{p}'] = m.params[p]
            row[f'pval_{p}'] = m.pvalues[p]
        results.append(row)
    except:
        pass

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'logistic_results.csv'), index=False)
print(f"Converged: {len(results_df)} / {len(successful)}")

# ============================================================
# Step 4: Group-level tests
# ============================================================

print(f"\n{'=' * 60}")
print("GROUP-LEVEL RESULTS")
print("=" * 60)

print("\n--- Main Effects (ROI predicting rejection, averaged across time) ---")
for roi in ['vmPFC', 'L_insula', 'dACC']:
    col = f'coef_{roi}_z'
    if col in results_df.columns:
        v = results_df[col].dropna().values
        t, p = stats.ttest_1samp(v, 0)
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
        print(f"  {roi:12s}: mean = {v.mean():.4f}, t({len(v)-1}) = {t:.3f}, p = {p:.4f} {sig}")

print("\n--- ROI × Time Interactions ---")
print("  Positive = ROI becomes MORE predictive of rejection over time")
print("  Negative = ROI becomes LESS predictive over time")
for roi in ['vmPFC', 'L_insula', 'dACC']:
    col = f'coef_{roi}_z:trial_z'
    if col in results_df.columns:
        v = results_df[col].dropna().values
        t, p = stats.ttest_1samp(v, 0)
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
        print(f"  {roi:12s} × time: mean = {v.mean():.4f}, t({len(v)-1}) = {t:.3f}, p = {p:.4f} {sig}")

# ============================================================
# Step 5: Early vs Late separate models
# ============================================================

print("\n--- Early vs Late: ROI predictive coefficients ---")

early_c = {r: [] for r in ['vmPFC', 'L_insula', 'dACC']}
late_c = {r: [] for r in ['vmPFC', 'L_insula', 'dACC']}

for subj in successful:
    sd = trial_df[trial_df['subject'] == subj].copy()
    for runs, store in [([1,2], early_c), ([3,4], late_c)]:
        pd_data = sd[sd['run'].isin(runs)]
        try:
            m = logit_model('reject ~ vmPFC_z + L_insula_z + dACC_z + gain + loss',
                           data=pd_data).fit(disp=0, maxiter=100)
            for roi in ['vmPFC', 'L_insula', 'dACC']:
                if f'{roi}_z' in m.params.index:
                    store[roi].append(m.params[f'{roi}_z'])
        except:
            pass

print(f"\n  {'ROI':12s} {'Early':>10s} {'Late':>10s} {'t':>8s} {'p':>8s} {'sig':>5s}")
for roi in ['vmPFC', 'L_insula', 'dACC']:
    e = np.array(early_c[roi])
    l = np.array(late_c[roi])
    n = min(len(e), len(l))
    if n > 5:
        t, p = stats.ttest_rel(l[:n], e[:n])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
        print(f"  {roi:12s} {e[:n].mean():10.4f} {l[:n].mean():10.4f} {t:8.3f} {p:8.4f} {sig:>5s}")

print(f"\n{'=' * 60}")
print("DONE — all data saved in trial_logistic_results/")
print("=" * 60)
