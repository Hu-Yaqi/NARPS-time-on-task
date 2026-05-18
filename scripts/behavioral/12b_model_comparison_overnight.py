"""
12b_model_comparison_overnight.py
=================================
Hierarchical Bayesian model comparison across four nested prospect theory
models (M1: EV, M2: loss aversion, M3: curvature, M4: full PT).
Compares via WAIC. All 108 subjects, non-centered parameterization.

Outputs:
  - trace_model[1-4]_*.nc: MCMC traces
  - model_comparison_results.json: WAIC scores
"""

import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import time
import json

# ============================================================
# Data preparation
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')
subjects = sorted(df['subject'].unique())
sub_to_idx = {s: i for i, s in enumerate(subjects)}
df['sub_idx'] = df['subject'].map(sub_to_idx)
n_subjects = len(subjects)

gain = df['gain'].values.astype('float64')
loss = df['loss'].values.astype('float64')
choice = df['accepted'].values.astype('float64')
sub_idx = df['sub_idx'].values
log_gain = np.log(gain)
log_loss = np.log(loss)

print(f"Subjects: {n_subjects}, Trials: {len(choice)}")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

all_results = {}

# ============================================================
# Shared sampling configuration
# ============================================================

SAMPLE_KWARGS = dict(
    draws=2000, tune=3000, chains=4,
    random_seed=42, target_accept=0.95, cores=1,
    max_treedepth=12,
    idata_kwargs={"log_likelihood": True},
)

# ============================================================
# Model 1: Expected Value (SV = gain - loss)
# ============================================================

print("\n>>> Model 1: Expected Value (SV = gain - loss)")
start = time.time()

with pm.Model() as model1:
    mu_beta = pm.Normal('mu_beta', mu=1.0, sigma=0.5)
    sigma_beta = pm.Exponential('sigma_beta', lam=5)
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
    beta = pm.math.exp(mu_beta + sigma_beta * beta_offset)

    sv = gain - loss
    p_accept = pm.math.sigmoid(beta[sub_idx] * sv)
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)
    y = pm.Bernoulli('y', p=p_accept, observed=choice)
    trace1 = pm.sample(**SAMPLE_KWARGS)

elapsed1 = time.time() - start
trace1.to_netcdf('trace_model1_EV.nc')
waic1 = az.waic(trace1)
print(f"    Done in {elapsed1/60:.1f} min — WAIC = {waic1.elpd_waic:.1f} (SE={waic1.se:.1f})")

all_results['Model1_EV'] = {
    'waic': float(waic1.elpd_waic), 'waic_se': float(waic1.se),
    'p_waic': float(waic1.p_waic), 'time_min': elapsed1 / 60,
    'n_params': 1, 'description': 'SV = gain - loss'
}

# ============================================================
# Model 2: Loss Aversion (SV = gain - lambda * loss)
# ============================================================

print("\n>>> Model 2: Loss Aversion (SV = gain - λ·loss)")
start = time.time()

with pm.Model() as model2:
    mu_lam = pm.Normal('mu_lam', mu=0.2, sigma=0.3)
    sigma_lam = pm.Exponential('sigma_lam', lam=5)
    mu_beta = pm.Normal('mu_beta', mu=1.0, sigma=0.5)
    sigma_beta = pm.Exponential('sigma_beta', lam=5)

    lam_offset = pm.Normal('lam_offset', mu=0, sigma=1, shape=n_subjects)
    lam = pm.math.exp(mu_lam + sigma_lam * lam_offset)
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
    beta = pm.math.exp(mu_beta + sigma_beta * beta_offset)

    sv = gain - lam[sub_idx] * loss
    p_accept = pm.math.sigmoid(beta[sub_idx] * sv)
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)
    y = pm.Bernoulli('y', p=p_accept, observed=choice)
    trace2 = pm.sample(**SAMPLE_KWARGS)

elapsed2 = time.time() - start
trace2.to_netcdf('trace_model2_LA.nc')
waic2 = az.waic(trace2)
print(f"    Done in {elapsed2/60:.1f} min — WAIC = {waic2.elpd_waic:.1f} (SE={waic2.se:.1f})")

all_results['Model2_LA'] = {
    'waic': float(waic2.elpd_waic), 'waic_se': float(waic2.se),
    'p_waic': float(waic2.p_waic), 'time_min': elapsed2 / 60,
    'n_params': 2, 'description': 'SV = gain - λ·loss'
}

# ============================================================
# Model 3: Curvature Only (SV = gain^alpha - loss^alpha)
# ============================================================

print("\n>>> Model 3: Curvature Only (SV = gain^α - loss^α)")
start = time.time()

with pm.Model() as model3:
    mu_alpha = pm.Normal('mu_alpha', mu=0.5, sigma=0.5)
    sigma_alpha = pm.Exponential('sigma_alpha', lam=5)
    mu_beta = pm.Normal('mu_beta', mu=1.0, sigma=0.5)
    sigma_beta = pm.Exponential('sigma_beta', lam=5)

    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_subjects)
    alpha = pm.math.sigmoid(mu_alpha + sigma_alpha * alpha_offset)
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
    beta = pm.math.exp(mu_beta + sigma_beta * beta_offset)

    alpha_i = alpha[sub_idx]
    sv = pm.math.exp(alpha_i * log_gain) - pm.math.exp(alpha_i * log_loss)
    p_accept = pm.math.sigmoid(beta[sub_idx] * sv)
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)
    y = pm.Bernoulli('y', p=p_accept, observed=choice)
    trace3 = pm.sample(**SAMPLE_KWARGS)

elapsed3 = time.time() - start
trace3.to_netcdf('trace_model3_curv.nc')
waic3 = az.waic(trace3)
print(f"    Done in {elapsed3/60:.1f} min — WAIC = {waic3.elpd_waic:.1f} (SE={waic3.se:.1f})")

all_results['Model3_Curv'] = {
    'waic': float(waic3.elpd_waic), 'waic_se': float(waic3.se),
    'p_waic': float(waic3.p_waic), 'time_min': elapsed3 / 60,
    'n_params': 2, 'description': 'SV = gain^α - loss^α'
}

# ============================================================
# Model 4: Full Prospect Theory (SV = gain^alpha - lambda * loss^alpha)
# ============================================================

print("\n>>> Model 4: Full Prospect Theory (SV = gain^α - λ·loss^α)")
start = time.time()

with pm.Model() as model4:
    mu_lam = pm.Normal('mu_lam', mu=0.2, sigma=0.3)
    sigma_lam = pm.Exponential('sigma_lam', lam=5)
    mu_alpha = pm.Normal('mu_alpha', mu=0.5, sigma=0.5)
    sigma_alpha = pm.Exponential('sigma_alpha', lam=5)
    mu_beta = pm.Normal('mu_beta', mu=1.0, sigma=0.5)
    sigma_beta = pm.Exponential('sigma_beta', lam=5)

    lam_offset = pm.Normal('lam_offset', mu=0, sigma=1, shape=n_subjects)
    lam = pm.math.exp(mu_lam + sigma_lam * lam_offset)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_subjects)
    alpha = pm.math.sigmoid(mu_alpha + sigma_alpha * alpha_offset)
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
    beta = pm.math.exp(mu_beta + sigma_beta * beta_offset)

    alpha_i = alpha[sub_idx]
    sv = pm.math.exp(alpha_i * log_gain) - lam[sub_idx] * pm.math.exp(alpha_i * log_loss)
    p_accept = pm.math.sigmoid(beta[sub_idx] * sv)
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)
    y = pm.Bernoulli('y', p=p_accept, observed=choice)
    trace4 = pm.sample(**SAMPLE_KWARGS)

elapsed4 = time.time() - start
trace4.to_netcdf('trace_model4_PT.nc')
waic4 = az.waic(trace4)
print(f"    Done in {elapsed4/60:.1f} min — WAIC = {waic4.elpd_waic:.1f} (SE={waic4.se:.1f})")

all_results['Model4_PT'] = {
    'waic': float(waic4.elpd_waic), 'waic_se': float(waic4.se),
    'p_waic': float(waic4.p_waic), 'time_min': elapsed4 / 60,
    'n_params': 3, 'description': 'SV = gain^α - λ·loss^α'
}

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
for name, r in all_results.items():
    print(f"  {name:15s}: WAIC = {r['waic']:.1f} (SE={r['waic_se']:.1f}), "
          f"p_waic = {r['p_waic']:.1f}, time = {r['time_min']:.1f} min")

with open('model_comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved: model_comparison_results.json")
