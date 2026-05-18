"""
16_response_strength.py
=======================
Response strength analysis: how the distribution of strong vs weak
accept/reject responses changes across runs.

Key finding: participants shift from weak to strong rejection of
unfavorable gambles over time.

Outputs:
  - response_strength_analysis.png
  - Printed Run 1 vs Run 4 tests for mean strength and extreme rate
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('all_subjects_behavior.csv')

# ============================================================
# Encode response strength (1-4 scale)
# ============================================================

strength_map = {
    'strongly_reject': 1, 'weakly_reject': 2,
    'weakly_accept': 3, 'strongly_accept': 4
}
df['strength'] = df['participant_response'].map(strength_map)

print("=== Response distribution ===")
print(df['participant_response'].value_counts().sort_index())

# ============================================================
# Strength metrics by run
# ============================================================

df['extreme_choice'] = df['participant_response'].isin(
    ['strongly_accept', 'strongly_reject']).astype(int)
df['hesitant_choice'] = df['participant_response'].isin(
    ['weakly_accept', 'weakly_reject']).astype(int)

run_strength = df.groupby(['subject', 'run']).agg(
    mean_strength=('strength', 'mean'),
    extreme_rate=('extreme_choice', 'mean'),
    hesitant_rate=('hesitant_choice', 'mean')
).reset_index()

print("\n=== Run-wise response patterns ===")
print(run_strength.groupby('run')[
    ['mean_strength', 'extreme_rate', 'hesitant_rate']
].mean().round(3))

# Run 1 vs Run 4
print("\n=== Run 1 vs Run 4 ===")
for var, label in [('mean_strength', 'Mean strength (1-4)'),
                    ('extreme_rate', 'Extreme choice rate'),
                    ('hesitant_rate', 'Hesitant choice rate')]:
    r1 = run_strength[run_strength['run'] == 1].set_index('subject')[var]
    r4 = run_strength[run_strength['run'] == 4].set_index('subject')[var]
    common = r1.index.intersection(r4.index)
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    print(f"  {label}: R1={r1.mean():.3f}, R4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, d={d:.2f}")

# ============================================================
# Extreme choice rate by EV bin
# ============================================================

df['ev_diff'] = df['gain'] - df['loss']
df['ev_bin'] = pd.qcut(df['ev_diff'], q=5, labels=[
    'Very unfavorable', 'Unfavorable', 'Neutral',
    'Favorable', 'Very favorable'
])

pivot = df.groupby(['subject', 'run', 'ev_bin'], observed=True).agg(
    extreme_rate=('extreme_choice', 'mean'),
    mean_strength=('strength', 'mean')
).reset_index()

bins = ['Very unfavorable', 'Unfavorable', 'Neutral',
        'Favorable', 'Very favorable']

print("\n=== Extreme choice rate: Run 1 vs Run 4 (by bin) ===")
print(f"{'Bin':<22} {'R1_extreme':>10} {'R4_extreme':>10} {'t':>8} {'p':>8}")
print("-" * 60)

for b in bins:
    bd = pivot[pivot['ev_bin'] == b]
    r1 = bd[bd['run'] == 1].set_index('subject')['extreme_rate']
    r4 = bd[bd['run'] == 4].set_index('subject')['extreme_rate']
    common = r1.index.intersection(r4.index)
    if len(common) < 10:
        continue
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    print(f"{b:<22} {r1.mean():>10.3f} {r4.mean():>10.3f} {t:>8.2f} {p:>8.4f}")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: response type proportions across runs
response_types = ['strongly_reject', 'weakly_reject',
                  'weakly_accept', 'strongly_accept']
response_colors = ['#A32D2D', '#D85A30', '#1D9E75', '#0F6E56']

for resp, color in zip(response_types, response_colors):
    df[f'is_{resp}'] = (df['participant_response'] == resp).astype(int)
    means = df.groupby('run')[f'is_{resp}'].mean()
    axes[0].plot(means.index, means.values, marker='o',
                 color=color, linewidth=2, label=resp.replace('_', ' '))
axes[0].set_xlabel('Run')
axes[0].set_ylabel('Proportion')
axes[0].set_title('Response type proportions across runs')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].legend(fontsize=8)

# Panel 2: extreme choice rate across runs
means = run_strength.groupby('run')['extreme_rate'].mean()
sems = run_strength.groupby('run')['extreme_rate'].sem()
axes[1].errorbar(means.index, means.values, yerr=sems.values * 1.96,
                 marker='o', color='#534AB7', linewidth=2, capsize=4)
axes[1].set_xlabel('Run')
axes[1].set_ylabel('Extreme choice rate')
axes[1].set_title('"Strong" response rate across runs')
axes[1].set_xticks([1, 2, 3, 4])

# Panel 3: mean strength by EV bin across runs
for b, color in zip(bins, ['#A32D2D', '#D85A30', '#888780', '#1D9E75', '#0F6E56']):
    bd = pivot[pivot['ev_bin'] == b]
    means = bd.groupby('run')['mean_strength'].mean()
    axes[2].plot(means.index, means.values, marker='o',
                 color=color, linewidth=2, label=b)
axes[2].set_xlabel('Run')
axes[2].set_ylabel('Mean strength (1-4)')
axes[2].set_title('Choice strength by gamble type')
axes[2].set_xticks([1, 2, 3, 4])
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig('response_strength_analysis.png', dpi=150)
print("\nFigure saved: response_strength_analysis.png")
