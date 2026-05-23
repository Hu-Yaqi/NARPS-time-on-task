"""
45_stimulus_balance_check.py
============================
Verify that gain, loss, EV, and gain/loss ratio are balanced
across runs and across 8 time bins.

The NARPS design explicitly balanced stimuli across runs (Botvinik-Nezer
et al., 2019), but we verify empirically and report the statistics.
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('all_subjects_behavior.csv')

print("=" * 60)
print("Stimulus Balance Check")
print("=" * 60)

df['ev'] = df['gain'] - df['loss']
df['gl_ratio'] = df['gain'] / df['loss']

# ============================================================
# Balance across 4 runs
# ============================================================

print("\n--- Balance across runs ---")
print(f"{'Measure':<15} {'Run1':>8} {'Run2':>8} {'Run3':>8} {'Run4':>8} {'F':>8} {'p':>8}")
print("-" * 70)

for measure in ['gain', 'loss', 'ev', 'gl_ratio']:
    run_means = df.groupby('run')[measure].mean()
    # One-way ANOVA across runs
    groups = [df[df['run'] == r][measure].values for r in range(1, 5)]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"{measure:<15} {run_means[1]:>8.2f} {run_means[2]:>8.2f} "
          f"{run_means[3]:>8.2f} {run_means[4]:>8.2f} {f_stat:>8.3f} {p_val:>8.4f}")

# ============================================================
# Balance across 8 bins
# ============================================================

print("\n--- Balance across 8 bins ---")

# Assign bins
bin_rows = []
for subj in df['subject'].unique():
    s_data = df[df['subject'] == subj]
    for run in range(1, 5):
        run_data = s_data[s_data['run'] == run].reset_index(drop=True)
        n = len(run_data)
        half = n // 2
        bin_first = (run - 1) * 2 + 1
        bin_second = bin_first + 1
        for i in range(n):
            b = bin_first if i < half else bin_second
            bin_rows.append({'subject': subj, 'bin': b,
                            'gain': run_data.iloc[i]['gain'],
                            'loss': run_data.iloc[i]['loss']})

bin_df = pd.DataFrame(bin_rows)
bin_df['ev'] = bin_df['gain'] - bin_df['loss']
bin_df['gl_ratio'] = bin_df['gain'] / bin_df['loss']

print(f"{'Measure':<15} {'Bin1':>7} {'Bin2':>7} {'Bin3':>7} {'Bin4':>7} "
      f"{'Bin5':>7} {'Bin6':>7} {'Bin7':>7} {'Bin8':>7} {'F':>8} {'p':>8}")
print("-" * 95)

for measure in ['gain', 'loss', 'ev', 'gl_ratio']:
    bin_means = bin_df.groupby('bin')[measure].mean()
    groups = [bin_df[bin_df['bin'] == b][measure].values for b in range(1, 9)]
    f_stat, p_val = stats.f_oneway(*groups)
    vals = [f"{bin_means[b]:>7.2f}" for b in range(1, 9)]
    print(f"{measure:<15} {' '.join(vals)} {f_stat:>8.3f} {p_val:>8.4f}")

print("\n--- Summary ---")
print("If all F-tests are non-significant (p > .05),")
print("stimuli are balanced and temporal effects cannot be")
print("attributed to changing stimulus composition.")
print("=" * 60)
