"""
44b_model_comparison_fix.py
===========================
Fix for script 43's WAIC failure: re-fit independent and null models
with idata_kwargs={"log_likelihood": True}.

Part 1 (independent ROI) is cached from script 43, not re-run.
Only Part 2 (model comparison) is re-done.

Outputs in robustness_results/:
  - trace_independent_v2_fix.nc
  - trace_null_fix.nc
  - model_comparison_results.csv (updated)
"""

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import os
import time
import warnings
warnings.filterwarnings('ignore')

output_dir = 'robustness_results'
os.makedirs(output_dir, exist_ok=True)

N_BINS = 8

# Load bin-level data
bin_df = pd.read_csv('state_space_results/bin_level_data.csv')
subjects = sorted(bin_df['subject'].unique())
n_subj = len(subjects)
sub_to_idx = {s: i for i, s in enumerate(subjects)}

reject_mat = np.full((n_subj, N_BINS), np.nan)
gap_mat = np.full((n_subj, N_BINS), np.nan)
for _, row in bin_df.iterrows():
    j = sub_to_idx[row['subject']]
    b = int(row['bin']) - 1
    reject_mat[j, b] = row['reject_rate']
    gap_mat[j, b] = row['vmPFC_gap']

for mat in [reject_mat, gap_mat]:
    for b in range(N_BINS):
        if np.isnan(mat[:, b]).any():
            mat[np.isnan(mat[:, b]), b] = np.nanmean(mat[:, b])

print("=" * 60)
print("Script 43b: Model Comparison (WAIC fix)")
print(f"Subjects: {n_subj}")
print("=" * 60)

SAMPLE_KWARGS = dict(
    draws=1500, tune=2000, chains=4,
    target_accept=0.95, cores=1, progressbar=True,
    idata_kwargs={"log_likelihood": True},
)

# ============================================================
# Model A: Shared (load from 41b, check log_likelihood)
# ============================================================

print("\nModel A: Shared latent state")
trace_shared_file = 'state_space_results_v2/trace_shared.nc'
trace_shared = az.from_netcdf(trace_shared_file)

# Check if log_likelihood exists
has_ll_shared = hasattr(trace_shared, 'log_likelihood') and trace_shared.log_likelihood is not None
if has_ll_shared:
    ll_vars = list(trace_shared.log_likelihood.data_vars)
    print(f"  Log-likelihood available: {ll_vars}")
else:
    print("  WARNING: No log-likelihood in cached shared trace.")
    print("  Re-fitting shared model with log_likelihood=True...")
    t0 = time.time()

    with pm.Model() as shared_model:
        mu_delta = pm.Normal('mu_delta', mu=0, sigma=0.5)
        sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.3)
        delta_offset = pm.Normal('delta_offset', mu=0, sigma=1, shape=n_subj)
        delta = mu_delta + sigma_delta * delta_offset
        sigma_z = pm.HalfNormal('sigma_z', sigma=0.5)

        z_innovations = pm.Normal('z_innovations', mu=0, sigma=1,
                                   shape=(n_subj, N_BINS - 1))
        z_list = [pt.zeros(n_subj)]
        for b in range(1, N_BINS):
            z_list.append(z_list[b - 1] + delta + sigma_z * z_innovations[:, b - 1])
        z = pt.stack(z_list, axis=1)

        mu_B = pm.Normal('mu_B', mu=0.5, sigma=0.3)
        beta_B = pm.HalfNormal('beta_B', sigma=1.0)
        sigma_B = pm.HalfNormal('sigma_B', sigma=0.2)
        pm.Normal('reject_obs', mu=mu_B + beta_B * z,
                  sigma=sigma_B, observed=reject_mat)

        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        beta_N = pm.Normal('beta_N', mu=0, sigma=1)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N + beta_N * z,
                  sigma=sigma_N, observed=gap_mat)

        trace_shared = pm.sample(**SAMPLE_KWARGS, random_seed=42)

    trace_shared.to_netcdf(os.path.join(output_dir, 'trace_shared_with_ll.nc'))
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# Model B: Independent latent states
# ============================================================

print("\nModel B: Independent latent states")
trace_indep_file = os.path.join(output_dir, 'trace_independent_fix.nc')

if os.path.exists(trace_indep_file):
    print(f"  Loading cached: {trace_indep_file}")
    trace_indep = az.from_netcdf(trace_indep_file)
else:
    print("  Fitting...")
    t0 = time.time()

    with pm.Model() as indep_model:
        # Behavioral latent state
        mu_db = pm.Normal('mu_db', mu=0, sigma=0.5)
        sigma_db = pm.HalfNormal('sigma_db', sigma=0.3)
        db_offset = pm.Normal('db_offset', mu=0, sigma=1, shape=n_subj)
        db = mu_db + sigma_db * db_offset
        sigma_zb = pm.HalfNormal('sigma_zb', sigma=0.5)

        zb_inn = pm.Normal('zb_inn', mu=0, sigma=1, shape=(n_subj, N_BINS - 1))
        zb_list = [pt.zeros(n_subj)]
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
        dn_offset = pm.Normal('dn_offset', mu=0, sigma=1, shape=n_subj)
        dn = mu_dn + sigma_dn * dn_offset
        sigma_zn = pm.HalfNormal('sigma_zn', sigma=0.5)

        zn_inn = pm.Normal('zn_inn', mu=0, sigma=1, shape=(n_subj, N_BINS - 1))
        zn_list = [pt.zeros(n_subj)]
        for b in range(1, N_BINS):
            zn_list.append(zn_list[b-1] + dn + sigma_zn * zn_inn[:, b-1])
        zn = pt.stack(zn_list, axis=1)

        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        beta_N = pm.Normal('beta_N', mu=0, sigma=1)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N + beta_N * zn,
                  sigma=sigma_N, observed=gap_mat)

        trace_indep = pm.sample(**SAMPLE_KWARGS, random_seed=50)

    trace_indep.to_netcdf(trace_indep_file)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# Model C: Null (intercept only)
# ============================================================

print("\nModel C: Null (no latent state)")
trace_null_file = os.path.join(output_dir, 'trace_null_fix.nc')

if os.path.exists(trace_null_file):
    print(f"  Loading cached: {trace_null_file}")
    trace_null = az.from_netcdf(trace_null_file)
else:
    print("  Fitting...")
    t0 = time.time()

    with pm.Model() as null_model:
        mu_B = pm.Normal('mu_B', mu=0.5, sigma=0.3)
        sigma_B = pm.HalfNormal('sigma_B', sigma=0.2)
        pm.Normal('reject_obs', mu=mu_B, sigma=sigma_B, observed=reject_mat)

        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N, sigma=sigma_N, observed=gap_mat)

        trace_null = pm.sample(**SAMPLE_KWARGS, random_seed=51)

    trace_null.to_netcdf(trace_null_file)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# Model D: Shared with reset (load from 41b)
# ============================================================

print("\nModel D: Shared with reset")
trace_reset_file = 'state_space_results_v2/trace_shared_reset.nc'
trace_reset = None
if os.path.exists(trace_reset_file):
    trace_reset = az.from_netcdf(trace_reset_file)
    has_ll_reset = hasattr(trace_reset, 'log_likelihood') and trace_reset.log_likelihood is not None
    if has_ll_reset:
        print(f"  Loaded with log-likelihood")
    else:
        print("  No log-likelihood; re-fitting with reset...")
        t0 = time.time()
        is_boundary = np.array([0, 0, 1, 0, 1, 0, 1, 0])

        with pm.Model() as reset_model:
            mu_delta = pm.Normal('mu_delta', mu=0, sigma=0.5)
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.3)
            delta_offset = pm.Normal('delta_offset', mu=0, sigma=1, shape=n_subj)
            delta = mu_delta + sigma_delta * delta_offset
            sigma_z = pm.HalfNormal('sigma_z', sigma=0.5)
            rho = pm.Beta('rho', alpha=5, beta=2)

            z_innovations = pm.Normal('z_innovations', mu=0, sigma=1,
                                       shape=(n_subj, N_BINS - 1))
            z_list = [pt.zeros(n_subj)]
            for b in range(1, N_BINS):
                if is_boundary[b]:
                    z_list.append(rho * z_list[b-1] + delta + sigma_z * z_innovations[:, b-1])
                else:
                    z_list.append(z_list[b-1] + delta + sigma_z * z_innovations[:, b-1])
            z = pt.stack(z_list, axis=1)

            mu_B = pm.Normal('mu_B', mu=0.5, sigma=0.3)
            beta_B = pm.HalfNormal('beta_B', sigma=1.0)
            sigma_B = pm.HalfNormal('sigma_B', sigma=0.2)
            pm.Normal('reject_obs', mu=mu_B + beta_B * z,
                      sigma=sigma_B, observed=reject_mat)

            mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
            beta_N = pm.Normal('beta_N', mu=0, sigma=1)
            sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
            pm.Normal('gap_obs', mu=mu_N + beta_N * z,
                      sigma=sigma_N, observed=gap_mat)

            trace_reset = pm.sample(**SAMPLE_KWARGS, random_seed=44)

        trace_reset.to_netcdf(os.path.join(output_dir, 'trace_reset_with_ll.nc'))
        print(f"  Done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# WAIC comparison
# ============================================================

print(f"\n{'=' * 60}")
print("WAIC Comparison (per outcome, then combined)")
print("=" * 60)

models = {'Shared': trace_shared, 'Independent': trace_indep, 'Null': trace_null}
if trace_reset is not None:
    models['Shared_reset'] = trace_reset

comparison_results = []

for outcome_var in ['reject_obs', 'gap_obs']:
    print(f"\n  Outcome: {outcome_var}")
    for model_name, trace in models.items():
        if trace is None:
            continue
        try:
            waic = az.waic(trace, var_name=outcome_var)
            elpd = waic.elpd_waic
            se = waic.se
            p_waic = waic.p_waic
            print(f"    {model_name:18s}: elpd_waic = {elpd:.1f} (SE = {se:.1f}, p_waic = {p_waic:.1f})")
            comparison_results.append({
                'outcome': outcome_var, 'model': model_name,
                'elpd_waic': elpd, 'se': se, 'p_waic': p_waic,
            })
        except Exception as e:
            print(f"    {model_name:18s}: FAILED: {e}")

# Combined elpd
print(f"\n  Combined elpd_waic (sum of both outcomes):")
for model_name in models:
    vals = [r for r in comparison_results if r['model'] == model_name]
    if len(vals) == 2:
        total_elpd = sum(r['elpd_waic'] for r in vals)
        total_se = np.sqrt(sum(r['se']**2 for r in vals))
        print(f"    {model_name:18s}: elpd_waic = {total_elpd:.1f} (SE ~ {total_se:.1f})")
        comparison_results.append({
            'outcome': 'combined', 'model': model_name,
            'elpd_waic': total_elpd, 'se': total_se,
        })

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv(os.path.join(output_dir, 'model_comparison_waic.csv'), index=False)
print(f"\nSaved: model_comparison_waic.csv")

# Pairwise differences
print(f"\n{'=' * 60}")
print("Pairwise Differences (positive = first model better)")
print("=" * 60)

combined = comp_df[comp_df['outcome'] == 'combined'].set_index('model')

pairs = [
    ('Shared', 'Null', 'Shared vs Null (drift helps?)'),
    ('Independent', 'Null', 'Independent vs Null (drift helps?)'),
    ('Shared', 'Independent', 'Shared vs Independent (one state vs two?)'),
]
if 'Shared_reset' in combined.index:
    pairs.append(('Shared_reset', 'Shared', 'Reset vs No-reset'))
    pairs.append(('Shared_reset', 'Null', 'Reset vs Null'))

for m1, m2, label in pairs:
    if m1 in combined.index and m2 in combined.index:
        diff = combined.loc[m1, 'elpd_waic'] - combined.loc[m2, 'elpd_waic']
        se1 = combined.loc[m1, 'se']
        se2 = combined.loc[m2, 'se']
        # Approximate SE of difference
        se_diff = np.sqrt(se1**2 + se2**2)
        z_score = diff / se_diff if se_diff > 0 else 0
        print(f"  {label}")
        print(f"    diff = {diff:.1f} (SE ~ {se_diff:.1f}, z ~ {z_score:.2f})")

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 60}")
print("SUMMARY FOR PAPER")
print("=" * 60)

print("""
Report:
  1. elpd_waic for each model (reject_obs and gap_obs separately,
     plus combined)
  2. Pairwise differences with approximate SE
  3. Interpretation:
     - Shared > Null: temporal dynamics improve fit
     - Shared >= Independent: one state is as good as two (parsimony)
     - Shared > Independent: one state is BETTER than two
     - Reset > No-reset: sawtooth dynamics improve fit
""")

print(f"All results saved in {output_dir}/")
print("=" * 60)
