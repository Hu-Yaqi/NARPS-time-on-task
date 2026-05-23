"""
44_independent_roi_and_model_comparison.py
==========================================
Two critical robustness analyses:

1. Independent vmPFC ROI: Re-extract 8-bin trajectories using the
   Tom et al. (2007) coordinate [−2, 28, −10] instead of the
   within-dataset peak [−12, 36, −13]. Tests whether the double
   dissociation survives at an independently defined location.

2. State-space model comparison: Compare shared latent state vs
   independent latent states vs no-drift null. Uses pointwise
   log-likelihood comparison since multi-outcome WAIC is problematic.

Prerequisites:
  - Completed 8-bin GLMs (scripts 30/34 must have run)
  - state_space_results_v2/ traces from script 41b

Outputs in robustness_results/:
  - independent_roi_trajectory.csv
  - independent_roi_comparison.png
  - model_comparison_results.csv
  - model_comparison_summary.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiSpheresMasker
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

output_dir = 'robustness_results'
fig_dir = os.path.join(output_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

data_dir = 'data'
fmriprep_base = os.path.join(data_dir, 'derivatives', 'fmriprep')
N_BINS = 8

subjects = sorted([
    d for d in os.listdir(fmriprep_base)
    if d.startswith('sub-') and os.path.isdir(os.path.join(fmriprep_base, d))
])

# ============================================================
# PART 1: Independent vmPFC ROI
# ============================================================

print("=" * 60)
print("PART 1: Independent vmPFC ROI (Tom et al. 2007)")
print("=" * 60)

# Two ROIs to compare
rois = {
    'vmPFC_Tom2007': (-2, 28, -10),      # Tom et al. (2007) loss parametric peak
    'vmPFC_datapeak': (-12, 36, -13),     # Our within-dataset early-vs-late peak
}

maskers = {name: NiftiSpheresMasker(seeds=[coords], radius=8, standardize=False)
           for name, coords in rois.items()}

indep_roi_file = os.path.join(output_dir, 'independent_roi_trajectory.csv')

if os.path.exists(indep_roi_file):
    print(f"Loading cached: {indep_roi_file}")
    indep_df = pd.read_csv(indep_roi_file)
else:
    # We need the 8-bin GLM results. Check if individual bin contrasts
    # are saved, or if we need to rerun the GLM.
    # The most reliable approach: rerun the extraction from the existing
    # GLM framework (same as script 34 but with two ROIs).

    print("Extracting 8-bin trajectories for both ROIs...")
    print("(This requires refitting the 8-bin GLM per subject)")

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

    all_rows = []
    successful = []
    t0 = time.time()

    for subj in subjects:
        print(f"  {subj} ...", end=' ')
        ts = time.time()
        fdir = os.path.join(fmriprep_base, subj, 'func')
        if not os.path.exists(fdir):
            print("skip"); continue

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

            glm = FirstLevelModel(t_r=1.0, hrf_model='spm', drift_model='cosine',
                                  high_pass=0.01, smoothing_fwhm=6, minimize_memory=True)
            glm.fit(imgs, evts, confs)

            for b in range(1, 9):
                zg = glm.compute_contrast(f'gain_bin{b}', output_type='z_score')
                zl = glm.compute_contrast(f'loss_bin{b}', output_type='z_score')

                row = {'subject': subj, 'bin': b}
                for roi_name, masker in maskers.items():
                    row[f'{roi_name}_gain'] = masker.fit_transform(zg).flat[0]
                    row[f'{roi_name}_loss'] = masker.fit_transform(zl).flat[0]
                all_rows.append(row)

            print(f"ok ({time.time()-ts:.0f}s)")
            successful.append(subj)
        except Exception as e:
            print(f"fail: {e}")

    indep_df = pd.DataFrame(all_rows)
    indep_df.to_csv(indep_roi_file, index=False)
    print(f"\nExtracted: {len(successful)} subjects, {len(indep_df)} rows")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

# Compute gaps
for roi_name in rois:
    indep_df[f'{roi_name}_gap'] = indep_df[f'{roi_name}_gain'] - indep_df[f'{roi_name}_loss']

# ============================================================
# Compare trajectories: Tom vs data-driven ROI
# ============================================================

print(f"\n{'=' * 60}")
print("Trajectory comparison: Tom et al. vs data-driven ROI")
print("=" * 60)

bins = np.arange(1, 9)
n_subj = indep_df['subject'].nunique()

for roi_name, coords in rois.items():
    print(f"\n  {roi_name} (MNI {coords}):")

    for channel in ['gain', 'loss', 'gap']:
        col = f'{roi_name}_{channel}'
        pivot = indep_df.pivot(index='subject', columns='bin', values=col)

        slopes = []
        for subj in pivot.index:
            vals = pivot.loc[subj].values
            if not np.any(np.isnan(vals)):
                s, _, _, _, _ = stats.linregress(bins, vals)
                slopes.append(s)
        slopes = np.array(slopes)
        t, p = stats.ttest_1samp(slopes, 0)
        means = [pivot[b].mean() for b in range(1, 9)]

        sig = '*' if p < 0.05 else ''
        print(f"    {channel:>5s}: bin1={means[0]:+.3f}, bin8={means[7]:+.3f}, "
              f"slope={slopes.mean():+.4f}, t={t:.3f}, p={p:.4f} {sig}")

    # Group-level correlation of gap trajectory
    gap_col = f'{roi_name}_gap'
    gap_means = [indep_df[indep_df['bin'] == b][gap_col].mean() for b in range(1, 9)]
    r, p = stats.pearsonr(bins, gap_means)
    print(f"    Gap group-level trend: r={r:.3f}, p={p:.4f}")

# Correlation between the two ROIs' gap values
merged = indep_df[['subject', 'bin', 'vmPFC_Tom2007_gap', 'vmPFC_datapeak_gap']].dropna()
r_rois, p_rois = stats.pearsonr(merged['vmPFC_Tom2007_gap'], merged['vmPFC_datapeak_gap'])
print(f"\nCorrelation between ROIs' gap values: r = {r_rois:.3f}, p = {p_rois:.6f}")

# ============================================================
# Visualization: side by side trajectory comparison
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels = ['R1\n1st', 'R1\n2nd', 'R2\n1st', 'R2\n2nd',
          'R3\n1st', 'R3\n2nd', 'R4\n1st', 'R4\n2nd']

for idx, (roi_name, color, marker) in enumerate([
    ('vmPFC_Tom2007', '#E74C3C', 'o'),
    ('vmPFC_datapeak', '#2E86C1', 's')
]):
    # Gain
    gain_col = f'{roi_name}_gain'
    gain_pivot = indep_df.pivot(index='subject', columns='bin', values=gain_col)
    gm = [gain_pivot[b].mean() for b in range(1, 9)]
    gs = [gain_pivot[b].std() / np.sqrt(n_subj) for b in range(1, 9)]

    # Loss
    loss_col = f'{roi_name}_loss'
    loss_pivot = indep_df.pivot(index='subject', columns='bin', values=loss_col)
    lm = [loss_pivot[b].mean() for b in range(1, 9)]
    ls = [loss_pivot[b].std() / np.sqrt(n_subj) for b in range(1, 9)]

    # Gap
    gap_col = f'{roi_name}_gap'
    gap_pivot = indep_df.pivot(index='subject', columns='bin', values=gap_col)
    gapm = [gap_pivot[b].mean() for b in range(1, 9)]
    gaps = [gap_pivot[b].std() / np.sqrt(n_subj) for b in range(1, 9)]

    # Panel 1: Gain + Loss
    ax = axes[0] if idx == 0 else axes[1]
    short_name = 'Tom et al.' if 'Tom' in roi_name else 'Data-driven'
    ax.errorbar(bins, lm, yerr=np.array(ls)*1.96, color='#E74C3C', marker='o',
                linewidth=2, capsize=3, label='Loss')
    ax.errorbar(bins, gm, yerr=np.array(gs)*1.96, color='#2E86C1', marker='s',
                linewidth=2, capsize=3, label='Gain')
    for b in [2.5, 4.5, 6.5]:
        ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Sensitivity (z-stat)')
    ax.set_title(f'{short_name}\n{rois[roi_name]}')
    ax.set_xticks(bins)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8)

# Panel 3: Gap comparison overlay
ax = axes[2]
for roi_name, color, marker, label in [
    ('vmPFC_Tom2007', '#E74C3C', 'o', 'Tom et al. [-2,28,-10]'),
    ('vmPFC_datapeak', '#2E86C1', 's', 'Data-driven [-12,36,-13]')
]:
    gap_col = f'{roi_name}_gap'
    gap_pivot = indep_df.pivot(index='subject', columns='bin', values=gap_col)
    gapm = [gap_pivot[b].mean() for b in range(1, 9)]
    gaps = [gap_pivot[b].std() / np.sqrt(n_subj) for b in range(1, 9)]
    ax.errorbar(bins, gapm, yerr=np.array(gaps)*1.96, color=color, marker=marker,
                linewidth=2, capsize=3, label=label)

for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Time Bin')
ax.set_ylabel('vmPFC Gap (gain - loss)')
ax.set_title('Gap Trajectory: Independent vs Data-Driven')
ax.set_xticks(bins)
ax.set_xticklabels(labels, fontsize=7)
ax.legend(fontsize=8)

plt.suptitle(f'vmPFC ROI Comparison (n = {n_subj})', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'independent_roi_comparison.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'independent_roi_comparison.pdf'), format='pdf', bbox_inches='tight')
print(f"\nSaved: figures/independent_roi_comparison.png/.pdf")


# ============================================================
# PART 2: State-Space Model Comparison
# ============================================================

print(f"\n{'=' * 60}")
print("PART 2: State-Space Model Comparison")
print("=" * 60)

# Load bin-level data
bin_df = pd.read_csv('state_space_results/bin_level_data.csv')
ssm_subjects = sorted(bin_df['subject'].unique())
n_ssm = len(ssm_subjects)
sub_to_idx = {s: i for i, s in enumerate(ssm_subjects)}

reject_mat = np.full((n_ssm, N_BINS), np.nan)
gap_mat = np.full((n_ssm, N_BINS), np.nan)
for _, row in bin_df.iterrows():
    j = sub_to_idx[row['subject']]
    b = int(row['bin']) - 1
    reject_mat[j, b] = row['reject_rate']
    gap_mat[j, b] = row['vmPFC_gap']

for mat in [reject_mat, gap_mat]:
    for b in range(N_BINS):
        if np.isnan(mat[:, b]).any():
            mat[np.isnan(mat[:, b]), b] = np.nanmean(mat[:, b])

SAMPLE_KWARGS = dict(
    draws=1500, tune=2000, chains=4,
    target_accept=0.95, cores=1, progressbar=True,
)

# --- Model A: Shared latent state (already fit in 41b, load it) ---
print("\nModel A: Shared latent state (from 41b)")
trace_shared_file = 'state_space_results_v2/trace_shared.nc'
if os.path.exists(trace_shared_file):
    trace_shared = az.from_netcdf(trace_shared_file)
    print("  Loaded cached trace")
else:
    print("  ERROR: Run script 41b first")
    trace_shared = None

# --- Model B: Independent latent states ---
print("\nModel B: Independent latent states")
trace_indep_file = os.path.join(output_dir, 'trace_independent_v2.nc')

if os.path.exists(trace_indep_file):
    print(f"  Loading cached: {trace_indep_file}")
    trace_indep = az.from_netcdf(trace_indep_file)
else:
    print("  Building independent-state model...")
    t0 = time.time()

    with pm.Model() as indep_model:
        # Behavioral latent state
        mu_db = pm.Normal('mu_db', mu=0, sigma=0.5)
        sigma_db = pm.HalfNormal('sigma_db', sigma=0.3)
        db_offset = pm.Normal('db_offset', mu=0, sigma=1, shape=n_ssm)
        db = mu_db + sigma_db * db_offset
        sigma_zb = pm.HalfNormal('sigma_zb', sigma=0.5)

        zb_inn = pm.Normal('zb_inn', mu=0, sigma=1, shape=(n_ssm, N_BINS - 1))
        zb_list = [pt.zeros(n_ssm)]
        for b in range(1, N_BINS):
            zb_list.append(zb_list[b-1] + db + sigma_zb * zb_inn[:, b-1])
        zb = pt.stack(zb_list, axis=1)

        mu_B = pm.Normal('mu_B', mu=0.5, sigma=0.3)
        beta_B = pm.HalfNormal('beta_B', sigma=1.0)
        sigma_B = pm.HalfNormal('sigma_B', sigma=0.2)
        pm.Normal('reject_obs', mu=mu_B + beta_B * zb,
                  sigma=sigma_B, observed=reject_mat)

        # Neural latent state (SEPARATE)
        mu_dn = pm.Normal('mu_dn', mu=0, sigma=0.5)
        sigma_dn = pm.HalfNormal('sigma_dn', sigma=0.3)
        dn_offset = pm.Normal('dn_offset', mu=0, sigma=1, shape=n_ssm)
        dn = mu_dn + sigma_dn * dn_offset
        sigma_zn = pm.HalfNormal('sigma_zn', sigma=0.5)

        zn_inn = pm.Normal('zn_inn', mu=0, sigma=1, shape=(n_ssm, N_BINS - 1))
        zn_list = [pt.zeros(n_ssm)]
        for b in range(1, N_BINS):
            zn_list.append(zn_list[b-1] + dn + sigma_zn * zn_inn[:, b-1])
        zn = pt.stack(zn_list, axis=1)

        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        beta_N = pm.Normal('beta_N', mu=0, sigma=1)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N + beta_N * zn,
                  sigma=sigma_N, observed=gap_mat)

        print("  Sampling...")
        trace_indep = pm.sample(**SAMPLE_KWARGS, random_seed=50)

    elapsed = time.time() - t0
    trace_indep.to_netcdf(trace_indep_file)
    print(f"  Done in {elapsed / 60:.1f} min")

# --- Model C: Null (no drift, intercept only) ---
print("\nModel C: Null (no latent state, intercept only)")
trace_null_file = os.path.join(output_dir, 'trace_null.nc')

if os.path.exists(trace_null_file):
    print(f"  Loading cached: {trace_null_file}")
    trace_null = az.from_netcdf(trace_null_file)
else:
    print("  Building null model...")
    t0 = time.time()

    with pm.Model() as null_model:
        mu_B = pm.Normal('mu_B', mu=0.5, sigma=0.3)
        sigma_B = pm.HalfNormal('sigma_B', sigma=0.2)
        pm.Normal('reject_obs', mu=mu_B, sigma=sigma_B, observed=reject_mat)

        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N, sigma=sigma_N, observed=gap_mat)

        print("  Sampling...")
        trace_null = pm.sample(**SAMPLE_KWARGS, random_seed=51)

    elapsed = time.time() - t0
    trace_null.to_netcdf(trace_null_file)
    print(f"  Done in {elapsed / 60:.1f} min")

# ============================================================
# Model comparison via per-outcome WAIC
# ============================================================

print(f"\n{'=' * 60}")
print("Model Comparison")
print("=" * 60)

# Compute WAIC separately for each outcome
comparison_results = []

models = {
    'Shared': trace_shared,
    'Independent': trace_indep,
    'Null': trace_null,
}

for outcome_var in ['reject_obs', 'gap_obs']:
    print(f"\n  Outcome: {outcome_var}")
    for model_name, trace in models.items():
        if trace is None:
            continue
        try:
            waic = az.waic(trace, var_name=outcome_var)
            print(f"    {model_name:15s}: elpd_waic = {waic.elpd_waic:.1f} (SE = {waic.se:.1f})")
            comparison_results.append({
                'outcome': outcome_var,
                'model': model_name,
                'elpd_waic': waic.elpd_waic,
                'se': waic.se,
                'p_waic': waic.p_waic,
            })
        except Exception as e:
            print(f"    {model_name:15s}: WAIC failed: {e}")

# Also compute total (sum of both outcomes)
print(f"\n  Combined (sum of both outcomes):")
for model_name in ['Shared', 'Independent', 'Null']:
    vals = [r for r in comparison_results if r['model'] == model_name]
    if len(vals) == 2:
        total = sum(r['elpd_waic'] for r in vals)
        total_se = np.sqrt(sum(r['se']**2 for r in vals))
        print(f"    {model_name:15s}: total elpd_waic = {total:.1f} (SE ~ {total_se:.1f})")
        comparison_results.append({
            'outcome': 'combined',
            'model': model_name,
            'elpd_waic': total,
            'se': total_se,
        })

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
print(f"\nSaved: model_comparison_results.csv")

# Pairwise differences
print(f"\n  Pairwise differences (positive = first model better):")
combined = comp_df[comp_df['outcome'] == 'combined'].set_index('model')
if 'Shared' in combined.index and 'Independent' in combined.index:
    diff_si = combined.loc['Shared', 'elpd_waic'] - combined.loc['Independent', 'elpd_waic']
    print(f"    Shared - Independent: {diff_si:.1f}")
if 'Shared' in combined.index and 'Null' in combined.index:
    diff_sn = combined.loc['Shared', 'elpd_waic'] - combined.loc['Null', 'elpd_waic']
    print(f"    Shared - Null:        {diff_sn:.1f}")
if 'Independent' in combined.index and 'Null' in combined.index:
    diff_in = combined.loc['Independent', 'elpd_waic'] - combined.loc['Null', 'elpd_waic']
    print(f"    Independent - Null:   {diff_in:.1f}")

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)

print("\nPart 1: Independent ROI")
print("  If Tom et al. coordinate shows the same double dissociation,")
print("  the finding is not circular.")

print("\nPart 2: Model Comparison")
print("  If Shared > Independent > Null:")
print("    One latent state fits better than two, which fits better than none.")
print("  If Shared ~ Independent > Null:")
print("    Both drift models fit similarly; parsimony favors shared.")
print("  If Independent > Shared:")
print("    Two separate processes fit better than one shared process.")

print(f"\nAll results saved in {output_dir}/")
print("=" * 60)
