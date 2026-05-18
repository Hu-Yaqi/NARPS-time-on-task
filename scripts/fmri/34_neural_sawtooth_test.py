"""
34_neural_sawtooth_test.py
==========================
Re-run the 8-bin GLM (gain+loss) but this time SAVE per-subject per-bin
vmPFC values, then do individual-level sawtooth statistics.

Tests:
1. Neural loss between-run reset (next run 1st - current run 2nd)
2. Neural loss within-run change (2nd - 1st)
3. Neural gain between-run reset
4. Neural gain within-run change
5. Gap (gain-loss) linear trend per subject
6. Gap convergence: is the gap at bin 8 smaller than bin 1?

运行方式：conda activate narps && python 34_neural_sawtooth_test.py
预计耗时：约 3-4 小时
"""

import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiSpheresMasker
from scipy import stats
import os
import time
import warnings
warnings.filterwarnings('ignore')

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'sawtooth_statistics'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

vmPFC_masker = NiftiSpheresMasker(seeds=[(-12, 36, -13)], radius=8, standardize=False)


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


# ============================================================
# Extract per-subject per-bin values
# ============================================================

print("=" * 60)
print("Extracting per-subject per-bin vmPFC values")
print("=" * 60)

all_rows = []
successful = []
t0 = time.time()

for subj in subjects:
    print(f"\n处理 {subj} ...", end=' ')
    ts = time.time()
    fdir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fdir):
        print("跳过"); continue

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
            zg = glm.compute_contrast(f'gain_bin{b}', output_type='z_score')
            zl = glm.compute_contrast(f'loss_bin{b}', output_type='z_score')
            gv = vmPFC_masker.fit_transform(zg).flat[0]
            lv = vmPFC_masker.fit_transform(zl).flat[0]
            all_rows.append({'subject': subj, 'bin': b,
                            'gain_vmPFC': gv, 'loss_vmPFC': lv})

        print(f"✓ ({time.time()-ts:.0f}s)")
        successful.append(subj)
    except Exception as e:
        print(f"✗ {e}")

elapsed = time.time() - t0
print(f"\n完成: {len(successful)} subjects, {elapsed/60:.1f} min")

# Save individual data
ind_df = pd.DataFrame(all_rows)
ind_df.to_csv(os.path.join(output_dir, 'individual_bin_vmPFC.csv'), index=False)
print(f"保存: {output_dir}/individual_bin_vmPFC.csv")

# ============================================================
# Statistical Tests
# ============================================================

print(f"\n{'=' * 60}")
print("NEURAL SAWTOOTH STATISTICS (individual level)")
print("=" * 60)

n_subj = len(successful)

for signal, col in [('Loss', 'loss_vmPFC'), ('Gain', 'gain_vmPFC')]:
    print(f"\n--- {signal} ---")
    
    # Pivot to subject × bin
    pivot = ind_df.pivot(index='subject', columns='bin', values=col)
    
    # Linear trend per subject
    slopes = []
    for subj in pivot.index:
        y = pivot.loc[subj].values
        x = np.arange(1, 9)
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes.append(slope)
    slopes = np.array(slopes)
    t_trend, p_trend = stats.ttest_1samp(slopes, 0)
    print(f"  Linear trend: mean slope = {slopes.mean():.4f}, t({n_subj-1}) = {t_trend:.3f}, p = {p_trend:.4f}")
    
    # Within-run change: 2nd - 1st, averaged across 4 runs
    within = []
    for subj in pivot.index:
        vals = pivot.loc[subj].values
        diffs = [vals[r*2+1] - vals[r*2] for r in range(4)]
        within.append(np.mean(diffs))
    within = np.array(within)
    t_within, p_within = stats.ttest_1samp(within, 0)
    print(f"  Within-run (2nd-1st): mean = {within.mean():.4f}, t({n_subj-1}) = {t_within:.3f}, p = {p_within:.4f}")
    
    # Between-run reset: next 1st - current 2nd, averaged across 3 transitions
    between = []
    for subj in pivot.index:
        vals = pivot.loc[subj].values
        resets = [vals[(r+1)*2] - vals[r*2+1] for r in range(3)]
        between.append(np.mean(resets))
    between = np.array(between)
    t_between, p_between = stats.ttest_1samp(between, 0)
    print(f"  Between-run reset: mean = {between.mean():.4f}, t({n_subj-1}) = {t_between:.3f}, p = {p_between:.4f}")

# Gap analysis
print(f"\n--- Gap (Gain - Loss) ---")
ind_df['gap'] = ind_df['gain_vmPFC'] - ind_df['loss_vmPFC']
gap_pivot = ind_df.pivot(index='subject', columns='bin', values='gap')

# Gap linear trend
gap_slopes = []
for subj in gap_pivot.index:
    y = gap_pivot.loc[subj].values
    x = np.arange(1, 9)
    slope, _, _, _, _ = stats.linregress(x, y)
    gap_slopes.append(slope)
gap_slopes = np.array(gap_slopes)
t_gap, p_gap = stats.ttest_1samp(gap_slopes, 0)
print(f"  Gap linear trend: mean slope = {gap_slopes.mean():.4f}, t({n_subj-1}) = {t_gap:.3f}, p = {p_gap:.4f}")

# Gap bin1 vs bin8
gap_b1 = gap_pivot[1].values
gap_b8 = gap_pivot[8].values
t_gap18, p_gap18 = stats.ttest_rel(gap_b8, gap_b1)
print(f"  Gap bin1 vs bin8: bin1 mean = {gap_b1.mean():.3f}, bin8 mean = {gap_b8.mean():.3f}")
print(f"    paired t({n_subj-1}) = {t_gap18:.3f}, p = {p_gap18:.4f}")

# Gap first run vs last run
gap_r1 = (gap_pivot[1].values + gap_pivot[2].values) / 2
gap_r4 = (gap_pivot[7].values + gap_pivot[8].values) / 2
t_gapr, p_gapr = stats.ttest_rel(gap_r4, gap_r1)
d_gap = (gap_r4.mean() - gap_r1.mean()) / np.sqrt((gap_r4.std()**2 + gap_r1.std()**2) / 2)
print(f"  Gap Run1 vs Run4: R1 mean = {gap_r1.mean():.3f}, R4 mean = {gap_r4.mean():.3f}")
print(f"    paired t({n_subj-1}) = {t_gapr:.3f}, p = {p_gapr:.4f}, d = {d_gap:.3f}")

print(f"\n{'=' * 60}")
print("DONE")
print("=" * 60)
