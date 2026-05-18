"""
15_matched_fatigue_analysis.py
==============================
Time-on-task effects by gamble favorability. Trials are grouped into
five EV bins (gain - loss), then Run 1 vs Run 4 acceptance rates are
compared within each bin.

Key finding: unfavorable gambles show the largest fatigue effect (d=0.67);
favorable gambles are unaffected.

Outputs:
  - fatigue_by_ev_bin.png
  - Printed statistical tests per bin and for ambiguous vs extreme gambles
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('all_subjects_behavior.csv')

# ============================================================
# Bin trials by expected value (gain - loss)
# ============================================================

df['ev_diff'] = df['gain'] - df['loss']

print("=== EV difference range ===")
print(f"Min: {df['ev_diff'].min()}, Max: {df['ev_diff'].max()}")
print(f"Unique values: {df['ev_diff'].nunique()}")

df['ev_bin'] = pd.qcut(df['ev_diff'], q=5, labels=[
    'Very unfavorable', 'Unfavorable', 'Neutral',
    'Favorable', 'Very favorable'
])

print("\n=== Bin summary ===")
bin_summary = df.groupby('ev_bin', observed=True).agg(
    n_trials=('ev_diff', 'count'),
    mean_diff=('ev_diff', 'mean'),
    accept_rate=('accepted', 'mean')
)
print(bin_summary.round(2))

# ============================================================
# Per-subject per-run per-bin acceptance rates
# ============================================================

pivot = df.groupby(['subject', 'run', 'ev_bin'], observed=True)['accepted'].mean()
pivot = pivot.reset_index()

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
bins = ['Very unfavorable', 'Unfavorable', 'Neutral', 'Favorable', 'Very favorable']
colors = ['#A32D2D', '#D85A30', '#888780', '#1D9E75', '#0F6E56']

for i, (bin_name, color) in enumerate(zip(bins, colors)):
    ax = axes[i]
    bin_data = pivot[pivot['ev_bin'] == bin_name]

    for sub in df['subject'].unique():
        sub_data = bin_data[bin_data['subject'] == sub].sort_values('run')
        ax.plot(sub_data['run'], sub_data['accepted'],
                color='gray', alpha=0.05, linewidth=0.5)

    means = bin_data.groupby('run')['accepted'].mean()
    sems = bin_data.groupby('run')['accepted'].sem()
    ax.errorbar(means.index, means.values, yerr=sems.values * 1.96,
                color=color, linewidth=2.5, marker='o', markersize=7,
                capsize=4, capthick=2, zorder=10)
    ax.set_xlabel('Run', fontsize=11)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title(bin_name, fontsize=11, color=color)
    ax.set_ylim(-0.05, 1.05)

axes[0].set_ylabel('Accept rate', fontsize=12)
plt.suptitle('Accept rate across runs, by gamble favorability', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fatigue_by_ev_bin.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: fatigue_by_ev_bin.png")

# ============================================================
# Run 1 vs Run 4 paired t-test per bin
# ============================================================

print("\n=== Run 1 vs Run 4 paired t-test (by bin) ===")
print(f"{'Bin':<22} {'Run1':>8} {'Run4':>8} {'Diff':>8} {'t':>8} {'p':>8} {'d':>8}")
print("-" * 75)

for bin_name in bins:
    bin_data = pivot[pivot['ev_bin'] == bin_name]
    r1 = bin_data[bin_data['run'] == 1].set_index('subject')['accepted']
    r4 = bin_data[bin_data['run'] == 4].set_index('subject')['accepted']
    common = r1.index.intersection(r4.index)
    if len(common) < 10:
        print(f"{bin_name:<22} insufficient data")
        continue
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    print(f"{bin_name:<22} {r1.mean():>8.3f} {r4.mean():>8.3f} "
          f"{diff.mean():>8.3f} {t:>8.2f} {p:>8.4f} {d:>8.2f}")

# ============================================================
# Ambiguous vs extreme gambles
# ============================================================

print("\n=== Fatigue effect: ambiguous vs extreme gambles ===")

extreme_bins = ['Very unfavorable', 'Very favorable']
middle_bins = ['Unfavorable', 'Neutral', 'Favorable']

for label, bin_list in [('Extreme gambles', extreme_bins),
                         ('Ambiguous gambles', middle_bins)]:
    subset = pivot[pivot['ev_bin'].isin(bin_list)]
    r1 = subset[subset['run'] == 1].groupby('subject')['accepted'].mean()
    r4 = subset[subset['run'] == 4].groupby('subject')['accepted'].mean()
    common = r1.index.intersection(r4.index)
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    print(f"  {label}: Run1={r1.mean():.3f}, Run4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, d={d:.2f}")
