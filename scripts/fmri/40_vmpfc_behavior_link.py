"""
40_vmpfc_behavior_link.py
=========================
Tests whether vmPFC value-discrimination collapse is linked to
behavioral loss aversion drift at the individual level.

Three analyses:
  1. Does the initial gain-loss gap in vmPFC predict behavioral delta-lambda?
  2. Do vmPFC trajectory slopes (gain, loss, gap) predict delta-lambda?
  3. Does vmPFC's mediating role in choice weaken from early to late bins?

Prerequisites:
  - sawtooth_statistics/individual_bin_vmPFC.csv (from script 34)
  - runwise_parameters_fixed.csv (from script 11b)
  - all_subjects_behavior.csv

Output:
  - vmpfc_behavior_link/link_results.csv
  - vmpfc_behavior_link/scatter_plots.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = 'vmpfc_behavior_link'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Script 40: vmPFC Collapse — Behavioral Drift Link")
print("=" * 60)

# Per-subject per-bin vmPFC values (from script 34)
vmPFC_file = 'sawtooth_statistics/individual_bin_vmPFC.csv'
if not os.path.exists(vmPFC_file):
    print(f"ERROR: {vmPFC_file} not found. Run script 34 first.")
    exit(1)

ind = pd.read_csv(vmPFC_file)
print(f"Loaded vmPFC bin data: {ind.shape[0]} rows, "
      f"{ind['subject'].nunique()} subjects")

# Run-wise behavioral parameters (from script 11b)
runwise = pd.read_csv('runwise_parameters_fixed.csv')

# Behavioral trial data
behavior = pd.read_csv('all_subjects_behavior.csv')

# Identify subjects with both neural and behavioral data
neural_subs = sorted(ind['subject'].unique())
behav_subs = sorted(runwise['subject'].unique())
common_subs = sorted(set(neural_subs) & set(behav_subs))
n = len(common_subs)
print(f"Subjects with both neural and behavioral data: {n}")

# ============================================================
# Compute per-subject summary variables
# ============================================================

records = []

for subj in common_subs:
    row = {'subject': subj}

    # --- Neural: vmPFC bin values ---
    s_neural = ind[ind['subject'] == subj].sort_values('bin')
    if len(s_neural) != 8:
        continue

    bins = np.arange(1, 9)
    gain_vals = s_neural['gain_vmPFC'].values
    loss_vals = s_neural['loss_vmPFC'].values
    gap_vals = gain_vals - loss_vals

    # Bin 1 values
    row['gain_bin1'] = gain_vals[0]
    row['loss_bin1'] = loss_vals[0]
    row['gap_bin1'] = gap_vals[0]

    # Bin 8 values
    row['gain_bin8'] = gain_vals[7]
    row['loss_bin8'] = loss_vals[7]
    row['gap_bin8'] = gap_vals[7]

    # Trajectory slopes (linear regression across 8 bins)
    slope_gain, _, r_gain, _, _ = stats.linregress(bins, gain_vals)
    slope_loss, _, r_loss, _, _ = stats.linregress(bins, loss_vals)
    slope_gap, _, r_gap, _, _ = stats.linregress(bins, gap_vals)

    row['slope_gain'] = slope_gain
    row['slope_loss'] = slope_loss
    row['slope_gap'] = slope_gap
    row['r_gain'] = r_gain
    row['r_loss'] = r_loss
    row['r_gap'] = r_gap

    # Gap change (bin8 - bin1)
    row['gap_change'] = gap_vals[7] - gap_vals[0]

    # --- Behavioral: delta-lambda ---
    s_behav = runwise[runwise['subject'] == subj].sort_values('run')
    if len(s_behav) != 4:
        continue

    lam_early = s_behav[s_behav['run'].isin([1, 2])]['lambda'].mean()
    lam_late = s_behav[s_behav['run'].isin([3, 4])]['lambda'].mean()
    row['delta_lambda'] = lam_late - lam_early

    # Lambda slope across 4 runs
    slope_lam, _, _, _, _ = stats.linregress(
        s_behav['run'].values, s_behav['lambda'].values)
    row['lambda_slope'] = slope_lam

    records.append(row)

df = pd.DataFrame(records)
df.to_csv(os.path.join(output_dir, 'subject_summary.csv'), index=False)
n = len(df)
print(f"\nComplete data for {n} subjects")

# ============================================================
# Test 1: Does initial gap predict behavioral drift?
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 1: Initial vmPFC gap vs behavioral delta-lambda")
print("=" * 60)
print("If starting-point asymmetry drives behavioral drift,")
print("subjects with larger initial gap should show more lambda drift.")

test1_results = []

for predictor, label in [('gap_bin1', 'Initial gap (gain - loss)'),
                          ('gain_bin1', 'Initial gain sensitivity'),
                          ('loss_bin1', 'Initial loss sensitivity'),
                          ('gap_bin1', 'Initial gap → lambda slope')]:
    outcome = 'delta_lambda' if 'slope' not in label else 'lambda_slope'
    r, p = stats.pearsonr(df[predictor], df[outcome])
    rho, p_sp = stats.spearmanr(df[predictor], df[outcome])
    print(f"  {label}:")
    print(f"    vs delta_lambda: Pearson r={r:.3f}, p={p:.4f}")
    print(f"                     Spearman rho={rho:.3f}, p={p_sp:.4f}")
    test1_results.append({
        'test': 'T1', 'predictor': predictor, 'outcome': outcome,
        'pearson_r': r, 'pearson_p': p,
        'spearman_rho': rho, 'spearman_p': p_sp
    })

# Also test: initial gap magnitude (absolute value)
r, p = stats.pearsonr(df['gap_bin1'].abs(), df['delta_lambda'].abs())
print(f"\n  |Initial gap| vs |delta_lambda|: r={r:.3f}, p={p:.4f}")
test1_results.append({
    'test': 'T1', 'predictor': '|gap_bin1|', 'outcome': '|delta_lambda|',
    'pearson_r': r, 'pearson_p': p,
    'spearman_rho': np.nan, 'spearman_p': np.nan
})

# ============================================================
# Test 2: vmPFC trajectory slopes predict behavioral drift?
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 2: vmPFC trajectory slopes vs behavioral delta-lambda")
print("=" * 60)
print("If vmPFC collapse drives behavioral drift,")
print("subjects with steeper collapse should show more lambda drift.")

test2_results = []

for predictor, label in [('slope_loss', 'Loss trajectory slope'),
                          ('slope_gain', 'Gain trajectory slope'),
                          ('slope_gap', 'Gap trajectory slope'),
                          ('gap_change', 'Gap change (bin8 - bin1)')]:
    for outcome, out_label in [('delta_lambda', 'delta_lambda'),
                                ('lambda_slope', 'lambda_slope')]:
        r, p = stats.pearsonr(df[predictor], df[outcome])
        rho, p_sp = stats.spearmanr(df[predictor], df[outcome])
        sig = '**' if p < .01 else '*' if p < .05 else '†' if p < .1 else ''
        print(f"  {label} vs {out_label}: r={r:.3f}, p={p:.4f} {sig}")
        test2_results.append({
            'test': 'T2', 'predictor': predictor, 'outcome': outcome,
            'pearson_r': r, 'pearson_p': p,
            'spearman_rho': rho, 'spearman_p': p_sp
        })

# ============================================================
# Test 3: vmPFC mediation of choice — early vs late
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 3: vmPFC mediation of choice (early vs late)")
print("=" * 60)
print("Per-subject logistic: P(reject) ~ gain + loss + vmPFC_loss_signal")
print("Compare vmPFC coefficient in early bins (1-4) vs late bins (5-8)")

try:
    from statsmodels.discrete.discrete_model import Logit
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'statsmodels', '--break-system-packages'],
                   capture_output=True)
    from statsmodels.discrete.discrete_model import Logit

# We need trial-level data matched with bin-level vmPFC signal
# Use bin-level vmPFC as a subject-level moderator for each trial's bin

fmri_subjects = sorted(ind['subject'].unique())
behavior_fmri = behavior[behavior['subject'].isin(fmri_subjects)].copy()

# Assign each trial to a bin (same logic as scripts 28/30)
trial_bins = []
for subj in fmri_subjects:
    s_data = behavior_fmri[behavior_fmri['subject'] == subj].copy()
    for run in range(1, 5):
        run_data = s_data[s_data['run'] == run].reset_index(drop=True)
        n_trials = len(run_data)
        half = n_trials // 2
        bin_first = (run - 1) * 2 + 1
        bin_second = bin_first + 1
        for i in range(n_trials):
            b = bin_first if i < half else bin_second
            trial_bins.append({
                'subject': subj, 'run': run,
                'trial_idx': run_data.index[i] if i < len(run_data) else i,
                'bin': b,
                'gain': run_data.iloc[i]['gain'],
                'loss': run_data.iloc[i]['loss'],
                'accepted': run_data.iloc[i]['accepted'],
            })

trial_df = pd.DataFrame(trial_bins)
trial_df['reject'] = 1 - trial_df['accepted']

# Merge vmPFC bin-level signal into trial data
trial_df = trial_df.merge(
    ind[['subject', 'bin', 'loss_vmPFC', 'gain_vmPFC']],
    on=['subject', 'bin'], how='left'
)

# Z-score vmPFC signals within each subject
for col in ['loss_vmPFC', 'gain_vmPFC']:
    trial_df[col + '_z'] = trial_df.groupby('subject')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

# Split into early (bins 1-4) and late (bins 5-8)
early_trials = trial_df[trial_df['bin'] <= 4]
late_trials = trial_df[trial_df['bin'] >= 5]

# Per-subject logistic regression in early and late halves
early_coefs = []
late_coefs = []

for subj in fmri_subjects:
    for trials, store in [(early_trials, early_coefs),
                           (late_trials, late_coefs)]:
        sd = trials[trials['subject'] == subj].copy()
        if len(sd) < 20 or sd['reject'].nunique() < 2:
            continue
        try:
            X = sd[['gain', 'loss', 'loss_vmPFC_z']].copy()
            X['intercept'] = 1.0
            y = sd['reject'].values
            model = Logit(y, X[['intercept', 'gain', 'loss', 'loss_vmPFC_z']])
            result = model.fit(disp=0, maxiter=100, method='bfgs')
            store.append({
                'subject': subj,
                'coef_vmPFC': result.params['loss_vmPFC_z'],
                'p_vmPFC': result.pvalues['loss_vmPFC_z'],
                'coef_gain': result.params['gain'],
                'coef_loss': result.params['loss'],
            })
        except:
            pass

early_df = pd.DataFrame(early_coefs)
late_df = pd.DataFrame(late_coefs)

print(f"\n  Converged: {len(early_df)} early, {len(late_df)} late")

# Match subjects
common = set(early_df['subject']) & set(late_df['subject'])
early_matched = early_df[early_df['subject'].isin(common)].set_index('subject')
late_matched = late_df[late_df['subject'].isin(common)].set_index('subject')
common_sorted = sorted(common)

if len(common_sorted) > 10:
    e_vmPFC = early_matched.loc[common_sorted, 'coef_vmPFC'].values
    l_vmPFC = late_matched.loc[common_sorted, 'coef_vmPFC'].values

    t_med, p_med = stats.ttest_rel(l_vmPFC, e_vmPFC)
    print(f"\n  vmPFC loss coefficient:")
    print(f"    Early: mean = {e_vmPFC.mean():.4f} (SD = {e_vmPFC.std():.4f})")
    print(f"    Late:  mean = {l_vmPFC.mean():.4f} (SD = {l_vmPFC.std():.4f})")
    print(f"    Paired t({len(common_sorted)-1}) = {t_med:.3f}, p = {p_med:.4f}")

    if abs(l_vmPFC.mean()) < abs(e_vmPFC.mean()):
        print("    Direction: vmPFC influence WEAKENS (consistent with collapse)")
    else:
        print("    Direction: vmPFC influence STRENGTHENS or unchanged")

    # Also test gain and loss behavioral coefficients
    e_gain = early_matched.loc[common_sorted, 'coef_gain'].values
    l_gain = late_matched.loc[common_sorted, 'coef_gain'].values
    e_loss = early_matched.loc[common_sorted, 'coef_loss'].values
    l_loss = late_matched.loc[common_sorted, 'coef_loss'].values

    t_g, p_g = stats.ttest_rel(l_gain, e_gain)
    t_l, p_l = stats.ttest_rel(l_loss, e_loss)
    print(f"\n  Gain coefficient:  early={e_gain.mean():.4f}, late={l_gain.mean():.4f}, "
          f"t={t_g:.3f}, p={p_g:.4f}")
    print(f"  Loss coefficient:  early={e_loss.mean():.4f}, late={l_loss.mean():.4f}, "
          f"t={t_l:.3f}, p={p_l:.4f}")

    test3_result = {
        'test': 'T3', 'n_matched': len(common_sorted),
        'early_vmPFC_mean': e_vmPFC.mean(), 'late_vmPFC_mean': l_vmPFC.mean(),
        't': t_med, 'p': p_med,
        'direction': 'weakens' if abs(l_vmPFC.mean()) < abs(e_vmPFC.mean()) else 'strengthens'
    }
else:
    print("  Insufficient matched subjects for Test 3")
    test3_result = {'test': 'T3', 'n_matched': 0}

# ============================================================
# Save all results
# ============================================================

all_results = test1_results + test2_results
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv(os.path.join(output_dir, 'correlation_results.csv'), index=False)

test3_df = pd.DataFrame([test3_result])
test3_df.to_csv(os.path.join(output_dir, 'mediation_results.csv'), index=False)

# ============================================================
# Visualization
# ============================================================

print(f"\nGenerating figures...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Test 1 — initial values vs delta-lambda
for i, (var, label) in enumerate([
    ('gap_bin1', 'Initial Gap\n(gain - loss)'),
    ('gain_bin1', 'Initial Gain\nSensitivity'),
    ('loss_bin1', 'Initial Loss\nSensitivity')
]):
    ax = axes[0, i]
    ax.scatter(df[var], df['delta_lambda'], alpha=0.6, s=40, edgecolors='k',
               linewidth=0.5, color='#2E86C1')
    r, p = stats.pearsonr(df[var], df['delta_lambda'])

    # Regression line
    slope, intercept = np.polyfit(df[var], df['delta_lambda'], 1)
    x_line = np.linspace(df[var].min(), df[var].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Δλ (late - early)', fontsize=10)
    ax.set_title(f'r = {r:.3f}, p = {p:.3f}', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

# Row 2: Test 2 — trajectory slopes vs delta-lambda
for i, (var, label) in enumerate([
    ('slope_gap', 'Gap Trajectory\nSlope'),
    ('slope_loss', 'Loss Trajectory\nSlope'),
    ('slope_gain', 'Gain Trajectory\nSlope')
]):
    ax = axes[1, i]
    ax.scatter(df[var], df['delta_lambda'], alpha=0.6, s=40, edgecolors='k',
               linewidth=0.5, color='#E74C3C')
    r, p = stats.pearsonr(df[var], df['delta_lambda'])

    slope_fit, intercept = np.polyfit(df[var], df['delta_lambda'], 1)
    x_line = np.linspace(df[var].min(), df[var].max(), 100)
    ax.plot(x_line, slope_fit * x_line + intercept, 'r-', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Δλ (late - early)', fontsize=10)
    ax.set_title(f'r = {r:.3f}, p = {p:.3f}', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

plt.suptitle(f'vmPFC Collapse — Behavioral Drift Link (n = {n})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'scatter_plots.png'), dpi=200,
            bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'scatter_plots.pdf'), format='pdf',
            bbox_inches='tight')
print(f"Saved: {output_dir}/scatter_plots.png / .pdf")
plt.close()

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)

print("\nTest 1 — Initial gap predicts behavioral drift?")
r1, p1 = stats.pearsonr(df['gap_bin1'], df['delta_lambda'])
print(f"  Gap_bin1 vs delta_lambda: r = {r1:.3f}, p = {p1:.4f}")
if p1 < 0.05:
    print("  --> SIGNIFICANT: starting-point asymmetry linked to behavioral drift")
else:
    print("  --> Not significant: no evidence for starting-point mechanism")

print("\nTest 2 — vmPFC trajectory slopes predict behavioral drift?")
r2, p2 = stats.pearsonr(df['slope_gap'], df['delta_lambda'])
print(f"  Gap_slope vs delta_lambda: r = {r2:.3f}, p = {p2:.4f}")
if p2 < 0.05:
    print("  --> SIGNIFICANT: individual collapse rate linked to behavioral drift")
else:
    print("  --> Not significant: collapse and behavioral drift are decoupled")

print("\nTest 3 — vmPFC mediation weakens over time?")
if test3_result.get('p') is not None and test3_result['n_matched'] > 0:
    print(f"  Early vmPFC coef = {test3_result['early_vmPFC_mean']:.4f}, "
          f"Late = {test3_result['late_vmPFC_mean']:.4f}")
    print(f"  t = {test3_result['t']:.3f}, p = {test3_result['p']:.4f}")
    if test3_result['p'] < 0.05:
        print(f"  --> SIGNIFICANT: vmPFC influence on choice {test3_result['direction']}")
    else:
        print(f"  --> Not significant")
else:
    print("  --> Could not run (insufficient data)")

print(f"\nIMPLICATIONS FOR PAPER:")
print(f"  If Tests 1-2 are null: remove starting-point asymmetry claim.")
print(f"    Say: 'collapse is consistent with behavioral shift' without mechanism.")
print(f"    Flag joint state-space model as future work.")
print(f"  If Test 3 is significant: vmPFC's influence on choice weakens,")
print(f"    providing direct evidence that value integration degrades.")
print(f"  If all null: the neural and behavioral effects are parallel but")
print(f"    decoupled at individual level — honest limitation, flag future work.")

print(f"\nAll results saved in {output_dir}/")
print("=" * 60)
