"""
37_ppi_per_run.py
=================
Pipeline 3: Task-Modulated Connectivity, Per-Run

Computes vmPFC-target connectivity separately for each run,
yielding 4 time points that can be directly compared with
the vmPFC sensitivity trajectory.

For each run, we compute two connectivity measures:
  1. Overall: correlation between vmPFC and target across all task volumes
  2. Loss-modulated: correlation during high-loss vs low-loss trials
     The difference (high - low) captures "does connectivity increase
     when loss is large?" = task-modulated connectivity

Then test:
  - Does overall connectivity change across 4 runs? (linear trend)
  - Does loss-modulated connectivity change? (linear trend)

运行方式：conda activate narps && python 37_ppi_per_run.py
预计耗时：约 20-30 分钟（no GLM needed, just time series extraction）
"""

import pandas as pd
import numpy as np
from scipy import stats
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import load_img
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
output_dir = 'ppi_results'
os.makedirs(output_dir, exist_ok=True)
TR = 1.0
HRF_DELAY = 5

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

# Seed
seed_masker = NiftiSpheresMasker(seeds=[(-12, 36, -13)], radius=8,
                                  standardize=True, detrend=True)

# Targets
targets = {
    'L_insula':   (-34, 18, -4),
    'R_insula':   (34, 18, -4),
    'dACC':       (8, 44, 54),
    'L_amygdala': (-22, -4, -18),
    'R_IFG':      (48, 22, 13),
    'v_striatum': (0, 10, -6),
}

target_maskers = {name: NiftiSpheresMasker(seeds=[coords], radius=8,
                                            standardize=True, detrend=True)
                  for name, coords in targets.items()}

print("=" * 60)
print("Per-Run Task-Modulated Connectivity")
print("=" * 60)

all_rows = []
successful = []
t0 = time.time()

for subj in subjects:
    print(f"  {subj} ...", end=' ')
    ts_start = time.time()
    fdir = os.path.join(fmriprep_base, subj, 'func')
    if not os.path.exists(fdir):
        print("skip"); continue

    try:
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

            # Extract time series
            seed_ts = seed_masker.fit_transform(bold_img).flatten()

            # Load events
            events = pd.read_csv(events_file, sep='\t')
            events = events[events['participant_response'] != 'NoResp'].reset_index(drop=True)

            # Identify trial peak volumes
            loss_median = events['loss'].median()

            high_loss_vols = []
            low_loss_vols = []
            all_task_vols = []

            for _, trial in events.iterrows():
                peak = int(round(trial['onset'] + HRF_DELAY))
                # 3-volume window around peak
                vols = [v for v in range(max(0, peak-1), min(n_vol, peak+2))]
                all_task_vols.extend(vols)

                if trial['loss'] >= loss_median:
                    high_loss_vols.extend(vols)
                else:
                    low_loss_vols.extend(vols)

            all_task_vols = sorted(set(all_task_vols))
            high_loss_vols = sorted(set(high_loss_vols))
            low_loss_vols = sorted(set(low_loss_vols))

            for target_name, target_masker in target_maskers.items():
                target_ts = target_masker.fit_transform(bold_img).flatten()

                row = {'subject': subj, 'run': run, 'target': target_name}

                # 1. Overall task-period connectivity
                if len(all_task_vols) > 10:
                    r_task, _ = stats.pearsonr(seed_ts[all_task_vols],
                                               target_ts[all_task_vols])
                    row['conn_task'] = r_task
                    row['conn_task_z'] = np.arctanh(np.clip(r_task, -0.999, 0.999))
                else:
                    row['conn_task'] = np.nan
                    row['conn_task_z'] = np.nan

                # 2. High-loss connectivity
                if len(high_loss_vols) > 10:
                    r_high, _ = stats.pearsonr(seed_ts[high_loss_vols],
                                               target_ts[high_loss_vols])
                    row['conn_high_loss'] = r_high
                else:
                    row['conn_high_loss'] = np.nan

                # 3. Low-loss connectivity
                if len(low_loss_vols) > 10:
                    r_low, _ = stats.pearsonr(seed_ts[low_loss_vols],
                                              target_ts[low_loss_vols])
                    row['conn_low_loss'] = r_low
                else:
                    row['conn_low_loss'] = np.nan

                # 4. Loss-modulated connectivity (PPI-like): high - low
                if not np.isnan(row.get('conn_high_loss', np.nan)) and \
                   not np.isnan(row.get('conn_low_loss', np.nan)):
                    row['conn_loss_mod'] = row['conn_high_loss'] - row['conn_low_loss']
                else:
                    row['conn_loss_mod'] = np.nan

                all_rows.append(row)

        print(f"ok ({time.time()-ts_start:.0f}s)")
        successful.append(subj)
    except Exception as e:
        print(f"fail: {e}")

elapsed = time.time() - t0
n_subj = len(successful)
print(f"\n完成: {n_subj} subjects, {elapsed/60:.1f} min")

conn_df = pd.DataFrame(all_rows)
conn_df.to_csv(os.path.join(output_dir, 'per_run_connectivity.csv'), index=False)
print(f"保存: {output_dir}/per_run_connectivity.csv")

# ============================================================
# Statistical Tests
# ============================================================

print(f"\n{'=' * 60}")
print("RESULTS")
print("=" * 60)

# Test 1: Linear trend across 4 runs (individual level)
print("\n--- Linear Trend in Task-Period Connectivity (4 runs) ---")
print(f"  {'Target':12s} {'R1':>8s} {'R2':>8s} {'R3':>8s} {'R4':>8s} {'t(slope)':>10s} {'p':>8s} {'sig':>5s}")

trend_results = []

for target_name in targets:
    td = conn_df[conn_df['target'] == target_name]

    # Per-subject slopes
    slopes = []
    for subj in successful:
        sd = td[td['subject'] == subj].sort_values('run')
        if len(sd) == 4 and not sd['conn_task_z'].isna().any():
            slope, _, _, _, _ = stats.linregress([1,2,3,4], sd['conn_task_z'].values)
            slopes.append(slope)

    slopes = np.array(slopes)
    if len(slopes) > 5:
        t_val, p_val = stats.ttest_1samp(slopes, 0)
        sig = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else 'n.s.'

        # Run means for display
        run_means = [np.tanh(td[td['run']==r]['conn_task_z'].mean()) for r in range(1,5)]

        trend_results.append({
            'target': target_name, 'measure': 'task_connectivity',
            't': t_val, 'p': p_val, 'mean_slope': slopes.mean(),
            'R1': run_means[0], 'R4': run_means[3],
        })

        print(f"  {target_name:12s} {run_means[0]:8.3f} {run_means[1]:8.3f} {run_means[2]:8.3f} {run_means[3]:8.3f} {t_val:10.3f} {p_val:8.4f} {sig:>5s}")

# Test 2: Loss-modulated connectivity trend
print("\n--- Linear Trend in Loss-Modulated Connectivity ---")
print("  (high-loss connectivity minus low-loss connectivity)")
print(f"  {'Target':12s} {'R1':>8s} {'R2':>8s} {'R3':>8s} {'R4':>8s} {'t(slope)':>10s} {'p':>8s} {'sig':>5s}")

for target_name in targets:
    td = conn_df[conn_df['target'] == target_name]

    slopes = []
    for subj in successful:
        sd = td[td['subject'] == subj].sort_values('run')
        if len(sd) == 4 and not sd['conn_loss_mod'].isna().any():
            slope, _, _, _, _ = stats.linregress([1,2,3,4], sd['conn_loss_mod'].values)
            slopes.append(slope)

    slopes = np.array(slopes)
    if len(slopes) > 5:
        t_val, p_val = stats.ttest_1samp(slopes, 0)
        sig = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else 'n.s.'

        run_means = [td[td['run']==r]['conn_loss_mod'].mean() for r in range(1,5)]

        trend_results.append({
            'target': target_name, 'measure': 'loss_modulated',
            't': t_val, 'p': p_val, 'mean_slope': slopes.mean(),
            'R1': run_means[0], 'R4': run_means[3],
        })

        print(f"  {target_name:12s} {run_means[0]:8.3f} {run_means[1]:8.3f} {run_means[2]:8.3f} {run_means[3]:8.3f} {t_val:10.3f} {p_val:8.4f} {sig:>5s}")

# Test 3: Early vs Late (paired t-test)
print("\n--- Early (Run 1-2) vs Late (Run 3-4) ---")
print(f"  {'Target':12s} {'Early r':>10s} {'Late r':>10s} {'t':>8s} {'p':>8s} {'sig':>5s}")

for target_name in targets:
    td = conn_df[conn_df['target'] == target_name]

    early_z = td[td['run'].isin([1,2])].groupby('subject')['conn_task_z'].mean()
    late_z = td[td['run'].isin([3,4])].groupby('subject')['conn_task_z'].mean()

    common = early_z.index.intersection(late_z.index)
    if len(common) > 5:
        e = early_z.loc[common].values
        l = late_z.loc[common].values
        t_val, p_val = stats.ttest_rel(l, e)
        sig = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else 'n.s.'
        print(f"  {target_name:12s} {np.tanh(e.mean()):10.3f} {np.tanh(l.mean()):10.3f} {t_val:8.3f} {p_val:8.4f} {sig:>5s}")

pd.DataFrame(trend_results).to_csv(
    os.path.join(output_dir, 'connectivity_trends.csv'), index=False)

# ============================================================
# Visualization
# ============================================================

print(f"\n生成可视化...")

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Task connectivity across runs
ax = axes[0]
for target_name in targets:
    td = conn_df[conn_df['target'] == target_name]
    means = [td[td['run']==r]['conn_task'].mean() for r in range(1,5)]
    sems = [td[td['run']==r]['conn_task'].std() / np.sqrt(n_subj) for r in range(1,5)]
    ax.errorbar([1,2,3,4], means, yerr=sems, marker='o', label=target_name,
                linewidth=1.5, capsize=2)

ax.set_xlabel('Run')
ax.set_ylabel('vmPFC-Target Connectivity (r)')
ax.set_title('Task-Period Connectivity Across Runs')
ax.set_xticks([1,2,3,4])
ax.legend(fontsize=8, loc='best')

# Panel B: Loss-modulated connectivity
ax = axes[1]
for target_name in targets:
    td = conn_df[conn_df['target'] == target_name]
    means = [td[td['run']==r]['conn_loss_mod'].mean() for r in range(1,5)]
    sems = [td[td['run']==r]['conn_loss_mod'].std() / np.sqrt(n_subj) for r in range(1,5)]
    ax.errorbar([1,2,3,4], means, yerr=sems, marker='s', label=target_name,
                linewidth=1.5, capsize=2)

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Run')
ax.set_ylabel('Loss-Modulated Connectivity\n(high-loss r - low-loss r)')
ax.set_title('Loss-Modulated Connectivity Across Runs')
ax.set_xticks([1,2,3,4])
ax.legend(fontsize=8, loc='best')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'connectivity_trajectories.pdf'),
            format='pdf', bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'connectivity_trajectories.png'),
            bbox_inches='tight', dpi=200)
print(f"保存: {output_dir}/connectivity_trajectories.pdf / .png")

print(f"\n{'=' * 60}")
print("DONE")
print("=" * 60)
