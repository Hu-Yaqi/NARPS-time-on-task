"""
43_state_space_posthoc.py
=========================
Post-hoc analyses on state-space model results:

1. Pairwise beta_neural contrasts: is the difference between
   positive-group and negative-group ROIs credible?
2. Decompose gap changes into gain and loss channels per region:
   does the gap change because gain increases, loss decreases, or both?
3. Verify functional labels from our own Part C data:
   which regions are "value integration" vs "loss salience"
   based on their gain/loss response profiles?

Prerequisites:
  - state_space_results_v2/trace_shared.nc
  - state_space_results_v2/trace_robustness_{roi}.nc
  - state_space_results/bin_level_data.csv
  - state_space_results/bin_level_data_all_rois.csv

Outputs:
  - state_space_results_v2/posthoc_contrasts.csv
  - state_space_results_v2/channel_decomposition.csv
  - state_space_results_v2/functional_labels.csv
  - state_space_results_v2/figures/contrasts.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

v2_dir = 'state_space_results_v2'
fig_dir = os.path.join(v2_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ============================================================
# 1. Load all traces and extract beta_neural posteriors
# ============================================================

print("=" * 60)
print("Script 43: Post-hoc analyses on state-space results")
print("=" * 60)

rois_other = ['L_insula', 'R_insula', 'dACC', 'L_amygdala', 'R_IFG', 'v_striatum']

# Load vmPFC beta_neural
trace_vmpfc = az.from_netcdf(os.path.join(v2_dir, 'trace_shared.nc'))
betas = {'vmPFC': trace_vmpfc.posterior['beta_neural'].values.flatten()}

# Load other ROI beta_neurals
for roi in rois_other:
    trace_file = os.path.join(v2_dir, f'trace_robustness_{roi}.nc')
    if os.path.exists(trace_file):
        trace = az.from_netcdf(trace_file)
        betas[roi] = trace.posterior['beta_neural'].values.flatten()
    else:
        print(f"  Warning: {trace_file} not found, skipping {roi}")

print(f"Loaded beta_neural posteriors for {len(betas)} regions")

# ============================================================
# 2. Pairwise contrasts between regions
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 1: Pairwise contrasts (beta_roi_A - beta_roi_B)")
print("=" * 60)
print("If HDI excludes zero, the two regions respond differently")
print("to the latent state.\n")

all_rois = ['vmPFC'] + rois_other
contrast_results = []

# Key contrasts: each negative-group ROI vs each positive-group ROI
negative_group = ['vmPFC', 'L_amygdala', 'v_striatum']
positive_group = ['L_insula', 'R_insula', 'R_IFG']

print("--- Negative vs Positive group contrasts ---\n")
print(f"{'ROI_A (neg)':>15s}  {'ROI_B (pos)':>15s}  {'mean diff':>10s}  "
      f"{'HDI_low':>10s}  {'HDI_high':>10s}  {'Credible':>10s}")
print("-" * 80)

for roi_a in negative_group:
    for roi_b in positive_group:
        if roi_a not in betas or roi_b not in betas:
            continue
        diff = betas[roi_a] - betas[roi_b]
        hdi = az.hdi(diff, hdi_prob=0.95)
        credible = not (hdi[0] < 0 < hdi[1])
        credible_str = "YES" if credible else "no"

        contrast_results.append({
            'ROI_A': roi_a, 'ROI_B': roi_b,
            'mean_diff': diff.mean(),
            'hdi_low': hdi[0], 'hdi_high': hdi[1],
            'credible': credible,
        })

        print(f"{roi_a:>15s}  {roi_b:>15s}  {diff.mean():>10.3f}  "
              f"{hdi[0]:>10.3f}  {hdi[1]:>10.3f}  {credible_str:>10s}")

# Within-group contrasts (should NOT be credible if groups are coherent)
print("\n--- Within negative group ---\n")
for i, roi_a in enumerate(negative_group):
    for roi_b in negative_group[i+1:]:
        if roi_a not in betas or roi_b not in betas:
            continue
        diff = betas[roi_a] - betas[roi_b]
        hdi = az.hdi(diff, hdi_prob=0.95)
        credible = not (hdi[0] < 0 < hdi[1])

        contrast_results.append({
            'ROI_A': roi_a, 'ROI_B': roi_b,
            'mean_diff': diff.mean(),
            'hdi_low': hdi[0], 'hdi_high': hdi[1],
            'credible': credible,
        })

        print(f"  {roi_a} vs {roi_b}: diff={diff.mean():.3f}, "
              f"HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}] {'CREDIBLE' if credible else ''}")

print("\n--- Within positive group ---\n")
for i, roi_a in enumerate(positive_group):
    for roi_b in positive_group[i+1:]:
        if roi_a not in betas or roi_b not in betas:
            continue
        diff = betas[roi_a] - betas[roi_b]
        hdi = az.hdi(diff, hdi_prob=0.95)
        credible = not (hdi[0] < 0 < hdi[1])

        contrast_results.append({
            'ROI_A': roi_a, 'ROI_B': roi_b,
            'mean_diff': diff.mean(),
            'hdi_low': hdi[0], 'hdi_high': hdi[1],
            'credible': credible,
        })

        print(f"  {roi_a} vs {roi_b}: diff={diff.mean():.3f}, "
              f"HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}] {'CREDIBLE' if credible else ''}")

contrasts_df = pd.DataFrame(contrast_results)
contrasts_df.to_csv(os.path.join(v2_dir, 'posthoc_contrasts.csv'), index=False)
print(f"\nSaved: posthoc_contrasts.csv")

# ============================================================
# 3. Decompose gap into gain and loss channels
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 2: Channel decomposition (gain vs loss separately)")
print("=" * 60)
print("For each region, test whether the 8-bin trend is driven")
print("by gain changes, loss changes, or both.\n")

# Load bin-level data
bin_df = pd.read_csv('state_space_results/bin_level_data.csv')
bin_all_df = pd.read_csv('state_space_results/bin_level_data_all_rois.csv')

subjects = sorted(bin_df['subject'].unique())
n_subj = len(subjects)
bins = np.arange(1, 9)

decomposition = []

# vmPFC
for channel, col in [('gain', 'gain_vmPFC'), ('loss', 'loss_vmPFC'), ('gap', 'vmPFC_gap')]:
    pivot = bin_df.pivot(index='subject', columns='bin', values=col)
    slopes = []
    for subj in pivot.index:
        s, _, _, _, _ = stats.linregress(bins, pivot.loc[subj].values)
        slopes.append(s)
    slopes = np.array(slopes)
    t, p = stats.ttest_1samp(slopes, 0)
    means = [pivot[b].mean() for b in range(1, 9)]

    decomposition.append({
        'ROI': 'vmPFC', 'channel': channel,
        'bin1_mean': means[0], 'bin8_mean': means[7],
        'mean_slope': slopes.mean(), 't': t, 'p': p,
        'direction': 'increase' if slopes.mean() > 0 else 'decrease',
    })

    sig = '*' if p < 0.05 else ''
    print(f"  vmPFC {channel:>5s}: bin1={means[0]:+.3f} → bin8={means[7]:+.3f}, "
          f"slope={slopes.mean():+.4f}, t={t:.3f}, p={p:.4f} {sig}")

# Other ROIs
for roi in rois_other:
    gain_col = f'{roi}_gain'
    loss_col = f'{roi}_loss'

    if gain_col not in bin_all_df.columns:
        continue

    # Compute gap
    bin_all_df[f'{roi}_gap'] = bin_all_df[gain_col] - bin_all_df[loss_col]

    print(f"\n  {roi}:")
    for channel, col in [('gain', gain_col), ('loss', loss_col), ('gap', f'{roi}_gap')]:
        pivot = bin_all_df.pivot(index='subject', columns='bin', values=col)
        slopes = []
        for subj in pivot.index:
            vals = pivot.loc[subj].values
            if not np.any(np.isnan(vals)):
                s, _, _, _, _ = stats.linregress(bins, vals)
                slopes.append(s)
        slopes = np.array(slopes)
        if len(slopes) < 10:
            continue
        t, p = stats.ttest_1samp(slopes, 0)
        means = [pivot[b].mean() for b in range(1, 9)]

        decomposition.append({
            'ROI': roi, 'channel': channel,
            'bin1_mean': means[0], 'bin8_mean': means[7],
            'mean_slope': slopes.mean(), 't': t, 'p': p,
            'direction': 'increase' if slopes.mean() > 0 else 'decrease',
        })

        sig = '*' if p < 0.05 else ''
        print(f"    {channel:>5s}: bin1={means[0]:+.3f} → bin8={means[7]:+.3f}, "
              f"slope={slopes.mean():+.4f}, t={t:.3f}, p={p:.4f} {sig}")

decomp_df = pd.DataFrame(decomposition)
decomp_df.to_csv(os.path.join(v2_dir, 'channel_decomposition.csv'), index=False)
print(f"\nSaved: channel_decomposition.csv")

# ============================================================
# 4. Functional labels from our own data
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 3: Functional labels from Part C gain/loss profiles")
print("=" * 60)
print("Classifying regions by their gain vs loss response pattern.\n")

# For each region, compute the overall (all-bin average) gain and loss
# sensitivity, then classify:
#   - Value integration: both gain and loss present (gain > 0, loss < 0 or vice versa)
#   - Loss salience: strong loss, weak/absent gain
#   - Gain encoding: strong gain, weak/absent loss

label_results = []

# vmPFC
gain_overall = bin_df['gain_vmPFC'].mean()
loss_overall = bin_df['loss_vmPFC'].mean()

# One-sample t-tests: is the overall mean significantly different from 0?
gain_subj_means = bin_df.groupby('subject')['gain_vmPFC'].mean()
loss_subj_means = bin_df.groupby('subject')['loss_vmPFC'].mean()
t_gain, p_gain = stats.ttest_1samp(gain_subj_means, 0)
t_loss, p_loss = stats.ttest_1samp(loss_subj_means, 0)

gain_sig = p_gain < 0.05
loss_sig = p_loss < 0.05

if gain_sig and loss_sig:
    label = 'Value integration (gain + loss)'
elif loss_sig and not gain_sig:
    label = 'Loss salience'
elif gain_sig and not loss_sig:
    label = 'Gain encoding'
else:
    label = 'Neither significant'

label_results.append({
    'ROI': 'vmPFC',
    'mean_gain': gain_overall, 't_gain': t_gain, 'p_gain': p_gain, 'gain_sig': gain_sig,
    'mean_loss': loss_overall, 't_loss': t_loss, 'p_loss': p_loss, 'loss_sig': loss_sig,
    'functional_label': label,
    'beta_neural_sign': 'negative' if betas.get('vmPFC', np.array([0])).mean() < 0 else 'positive',
})

print(f"  vmPFC: gain={gain_overall:+.3f} (p={p_gain:.4f}), "
      f"loss={loss_overall:+.3f} (p={p_loss:.4f}) → {label}")

# Other ROIs
for roi in rois_other:
    gain_col = f'{roi}_gain'
    loss_col = f'{roi}_loss'

    if gain_col not in bin_all_df.columns:
        continue

    gain_subj = bin_all_df.groupby('subject')[gain_col].mean()
    loss_subj = bin_all_df.groupby('subject')[loss_col].mean()

    t_gain, p_gain = stats.ttest_1samp(gain_subj.dropna(), 0)
    t_loss, p_loss = stats.ttest_1samp(loss_subj.dropna(), 0)

    gain_sig = p_gain < 0.05
    loss_sig = p_loss < 0.05
    gain_mean = gain_subj.mean()
    loss_mean = loss_subj.mean()

    if gain_sig and loss_sig:
        label = 'Value integration (gain + loss)'
    elif loss_sig and not gain_sig:
        label = 'Loss salience'
    elif gain_sig and not loss_sig:
        label = 'Gain encoding'
    else:
        label = 'Neither significant'

    bn_sign = 'negative' if betas.get(roi, np.array([0])).mean() < 0 else 'positive'

    label_results.append({
        'ROI': roi,
        'mean_gain': gain_mean, 't_gain': t_gain, 'p_gain': p_gain, 'gain_sig': gain_sig,
        'mean_loss': loss_mean, 't_loss': t_loss, 'p_loss': p_loss, 'loss_sig': loss_sig,
        'functional_label': label,
        'beta_neural_sign': bn_sign,
    })

    print(f"  {roi:12s}: gain={gain_mean:+.3f} (p={p_gain:.4f}), "
          f"loss={loss_mean:+.3f} (p={p_loss:.4f}) → {label}")

labels_df = pd.DataFrame(label_results)
labels_df.to_csv(os.path.join(v2_dir, 'functional_labels.csv'), index=False)
print(f"\nSaved: functional_labels.csv")

# ============================================================
# 5. Cross-tabulation: functional label × beta sign
# ============================================================

print(f"\n{'=' * 60}")
print("CROSS-TABULATION: Functional label × Beta sign")
print("=" * 60)

print(f"\n{'ROI':>12s}  {'Functional label':>35s}  {'Beta sign':>12s}  {'Beta value':>12s}")
print("-" * 80)

for _, row in labels_df.iterrows():
    bn_val = betas.get(row['ROI'], np.array([0])).mean()
    credible = ''
    if row['ROI'] in betas:
        hdi = az.hdi(betas[row['ROI']], hdi_prob=0.95)
        if not (hdi[0] < 0 < hdi[1]):
            credible = ' *'

    print(f"{row['ROI']:>12s}  {row['functional_label']:>35s}  "
          f"{row['beta_neural_sign']:>12s}  {bn_val:>+10.3f}{credible}")

print("\n* = HDI excludes zero (credible)")

# Check: do the labels predict the sign?
print(f"\nPattern check:")
for label_type in labels_df['functional_label'].unique():
    subset = labels_df[labels_df['functional_label'] == label_type]
    signs = subset['beta_neural_sign'].value_counts()
    print(f"  {label_type}:")
    for sign, count in signs.items():
        rois_list = subset[subset['beta_neural_sign'] == sign]['ROI'].tolist()
        print(f"    {sign}: {count} ({', '.join(rois_list)})")

# ============================================================
# 6. Visualization
# ============================================================

print(f"\n{'=' * 60}")
print("Figures")
print("=" * 60)

# Figure: beta_neural values with HDIs, colored by functional label
fig, ax = plt.subplots(figsize=(10, 6))

roi_order = ['vmPFC', 'L_amygdala', 'v_striatum', 'dACC', 'R_IFG', 'L_insula', 'R_insula']
colors_map = {
    'Value integration (gain + loss)': '#C0392B',
    'Loss salience': '#2E86C1',
    'Gain encoding': '#27AE60',
    'Neither significant': '#888888',
}

y_positions = list(range(len(roi_order)))
for i, roi in enumerate(roi_order):
    if roi not in betas:
        continue
    bn = betas[roi]
    hdi = az.hdi(bn, hdi_prob=0.95)
    label_row = labels_df[labels_df['ROI'] == roi].iloc[0]
    color = colors_map.get(label_row['functional_label'], '#888888')

    ax.errorbar(bn.mean(), i, xerr=[[bn.mean() - hdi[0]], [hdi[1] - bn.mean()]],
                fmt='o', color=color, markersize=10, capsize=5, capthick=2,
                linewidth=2, zorder=5)

ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(roi_order, fontsize=11)
ax.set_xlabel('beta_neural (latent state → value discrimination gap)', fontsize=12)
ax.set_title('State-Space Model: Neural Loading by Region\n'
             'Red = value integration, Blue = loss salience', fontsize=13)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in colors_map.items()
                   if l in labels_df['functional_label'].values]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'contrasts_by_function.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'contrasts_by_function.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/contrasts_by_function.png/.pdf")
plt.close()

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)

n_between_credible = sum(1 for r in contrast_results
                          if r['ROI_A'] in negative_group
                          and r['ROI_B'] in positive_group
                          and r['credible'])
n_between_total = sum(1 for r in contrast_results
                       if r['ROI_A'] in negative_group
                       and r['ROI_B'] in positive_group)

print(f"\n  Between-group contrasts (neg vs pos): "
      f"{n_between_credible}/{n_between_total} credible")

n_within_credible = sum(1 for r in contrast_results
                         if ((r['ROI_A'] in negative_group and r['ROI_B'] in negative_group) or
                             (r['ROI_A'] in positive_group and r['ROI_B'] in positive_group))
                         and r['credible'])
n_within_total = sum(1 for r in contrast_results
                      if ((r['ROI_A'] in negative_group and r['ROI_B'] in negative_group) or
                          (r['ROI_A'] in positive_group and r['ROI_B'] in positive_group)))

print(f"  Within-group contrasts: "
      f"{n_within_credible}/{n_within_total} credible")

print(f"\n  If between-group contrasts are mostly credible AND")
print(f"  within-group contrasts are mostly NOT credible,")
print(f"  the two-group sign pattern is statistically supported.")

print(f"\n  If functional labels predict beta sign,")
print(f"  the reorganization interpretation has internal support")
print(f"  from our own data (not just imported from literature).")

print(f"\nAll results saved in {v2_dir}/")
print("=" * 60)
