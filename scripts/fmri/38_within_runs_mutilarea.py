"""
38_within_run_all_regions.py
============================
Analyze within-run change and between-run reset for all regions.
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('other_regions_results/individual_bin_all_rois.csv')
vm = pd.read_csv('sawtooth_statistics/individual_bin_vmPFC.csv')

rois_other = [c.replace('_loss','') for c in df.columns if c.endswith('_loss')]
subjects = df['subject'].unique()
n = len(subjects)

print('WITHIN-RUN CHANGE (2nd - 1st half) by region')
print('='*70)

for roi in rois_other:
    col = f'{roi}_loss'
    pivot = df.pivot(index='subject', columns='bin', values=col)

    run_within = []
    for r in range(4):
        diffs = pivot[r*2+2].values - pivot[r*2+1].values
        run_within.append(diffs.mean())

    all_within = []
    for subj in subjects:
        vals = pivot.loc[subj].values
        w = np.mean([vals[r*2+1]-vals[r*2] for r in range(4)])
        all_within.append(w)
    all_within = np.array(all_within)
    t, p = stats.ttest_1samp(all_within, 0)

    print(f'{roi:12s}: R1={run_within[0]:+.3f} R2={run_within[1]:+.3f} R3={run_within[2]:+.3f} R4={run_within[3]:+.3f} | mean={all_within.mean():+.4f} t={t:.2f} p={p:.3f}')

# vmPFC
pivot_vm = vm.pivot(index='subject', columns='bin', values='loss_vmPFC')
run_within = []
for r in range(4):
    diffs = pivot_vm[r*2+2].values - pivot_vm[r*2+1].values
    run_within.append(diffs.mean())
all_within = []
for subj in vm['subject'].unique():
    vals = pivot_vm.loc[subj].values
    w = np.mean([vals[r*2+1]-vals[r*2] for r in range(4)])
    all_within.append(w)
all_within = np.array(all_within)
t, p = stats.ttest_1samp(all_within, 0)
print(f'{"vmPFC":12s}: R1={run_within[0]:+.3f} R2={run_within[1]:+.3f} R3={run_within[2]:+.3f} R4={run_within[3]:+.3f} | mean={all_within.mean():+.4f} t={t:.2f} p={p:.3f}')

print()
print('BETWEEN-RUN RESET by region')
print('='*70)

for roi in rois_other:
    col = f'{roi}_loss'
    pivot = df.pivot(index='subject', columns='bin', values=col)

    run_between = []
    for r in range(3):
        diffs = pivot[(r+1)*2+1].values - pivot[r*2+2].values
        run_between.append(diffs.mean())

    all_between = []
    for subj in subjects:
        vals = pivot.loc[subj].values
        b = np.mean([vals[(r+1)*2]-vals[r*2+1] for r in range(3)])
        all_between.append(b)
    all_between = np.array(all_between)
    t, p = stats.ttest_1samp(all_between, 0)

    print(f'{roi:12s}: R1-2={run_between[0]:+.3f} R2-3={run_between[1]:+.3f} R3-4={run_between[2]:+.3f} | mean={all_between.mean():+.4f} t={t:.2f} p={p:.3f}')

# vmPFC
pivot_vm = vm.pivot(index='subject', columns='bin', values='loss_vmPFC')
run_between = []
for r in range(3):
    diffs = pivot_vm[(r+1)*2+1].values - pivot_vm[r*2+2].values
    run_between.append(diffs.mean())
all_between = []
for subj in vm['subject'].unique():
    vals = pivot_vm.loc[subj].values
    b = np.mean([vals[(r+1)*2]-vals[r*2+1] for r in range(3)])
    all_between.append(b)
all_between = np.array(all_between)
t, p = stats.ttest_1samp(all_between, 0)
print(f'{"vmPFC":12s}: R1-2={run_between[0]:+.3f} R2-3={run_between[1]:+.3f} R3-4={run_between[2]:+.3f} | mean={all_between.mean():+.4f} t={t:.2f} p={p:.3f}')