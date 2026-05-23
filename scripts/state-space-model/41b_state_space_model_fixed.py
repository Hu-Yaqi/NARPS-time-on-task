"""
41b_state_space_model_fixed.py
==============================
Fixed version of script 41 — addresses sign-flipping non-identifiability.

The issue in 41:
  The latent state z_t had no inherent scale or sign. Flipping the sign of z
  and the signs of (beta_behav, beta_neural) together left likelihood unchanged.
  Different chains converged on different sign conventions, producing
  r_hat ~ 1.7 and ESS < 10 for the betas.

The fix:
  Constrain beta_behav > 0 using a HalfNormal prior. This anchors the
  interpretation: a higher latent state always corresponds to more rejection
  (behavioral loss aversion drift). beta_neural can then be positive or
  negative; its sign indicates whether vmPFC moves with or against the
  behavioral drift.

All other model structure is identical to script 41.
Existing caches from script 41 are NOT reused (different priors, different models).

Outputs in state_space_results_v2/:
  - trace_shared.nc, trace_independent.nc, trace_shared_reset.nc
  - trace_robustness_{roi}.nc
  - posterior_summary.csv
  - state_trajectories.csv
  - robustness_results.csv
  - figures/*.png

Prerequisites:
  - state_space_results/bin_level_data.csv (reused from script 41)
  - state_space_results/bin_level_data_all_rois.csv (reused from script 41)
"""

import pandas as pd
import numpy as np
from scipy import stats
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

# New output directory so we don't clobber the v1 traces
output_dir = 'state_space_results_v2'
fig_dir = os.path.join(output_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

N_BINS = 8
is_boundary = np.array([0, 0, 1, 0, 1, 0, 1, 0])  # bins 3, 5, 7 are run starts

# ============================================================
# Load cached bin-level data from script 41
# ============================================================

print("=" * 60)
print("Script 41b: State-Space Model (sign-identifiable)")
print("=" * 60)

bin_data_file = 'state_space_results/bin_level_data.csv'
bin_all_file = 'state_space_results/bin_level_data_all_rois.csv'

if not os.path.exists(bin_data_file):
    print(f"ERROR: {bin_data_file} not found. Run script 41 first.")
    exit(1)

bin_df = pd.read_csv(bin_data_file)
print(f"Loaded: {bin_df['subject'].nunique()} subjects, {len(bin_df)} rows")

if os.path.exists(bin_all_file):
    bin_all_df = pd.read_csv(bin_all_file)
    print(f"All-ROI data: {len(bin_all_df)} rows")
else:
    bin_all_df = None

# Prepare arrays
subjects = sorted(bin_df['subject'].unique())
n_subj = len(subjects)
sub_to_idx = {s: i for i, s in enumerate(subjects)}
bin_df['sub_idx'] = bin_df['subject'].map(sub_to_idx)

reject_mat = np.full((n_subj, N_BINS), np.nan)
gap_mat = np.full((n_subj, N_BINS), np.nan)
for _, row in bin_df.iterrows():
    j = int(row['sub_idx'])
    b = int(row['bin']) - 1
    reject_mat[j, b] = row['reject_rate']
    gap_mat[j, b] = row['vmPFC_gap']

# Fill any missing
for b in range(N_BINS):
    if np.isnan(reject_mat[:, b]).any():
        reject_mat[np.isnan(reject_mat[:, b]), b] = np.nanmean(reject_mat[:, b])
    if np.isnan(gap_mat[:, b]).any():
        gap_mat[np.isnan(gap_mat[:, b]), b] = np.nanmean(gap_mat[:, b])

# ============================================================
# Sampling settings (macOS-friendly)
# ============================================================

SAMPLE_KWARGS = dict(
    draws=1500, tune=2000, chains=4,
    target_accept=0.95,
    cores=1,  # sequential to avoid macOS multiprocessing issues
    progressbar=True,
)

# ============================================================
# Primary model — shared latent state with sign anchor
# ============================================================

print(f"\n{'=' * 60}")
print("Model 1: Shared latent state (beta_behav > 0 anchor)")
print("=" * 60)

trace_file = os.path.join(output_dir, 'trace_shared.nc')

if os.path.exists(trace_file):
    print(f"Loading cached trace: {trace_file}")
    trace_shared = az.from_netcdf(trace_file)
else:
    print("Building model...")
    t0 = time.time()

    with pm.Model() as shared_model:
        # Hyperpriors for latent state drift
        mu_delta = pm.Normal('mu_delta', mu=0, sigma=0.5)
        sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.3)
        delta_offset = pm.Normal('delta_offset', mu=0, sigma=1, shape=n_subj)
        delta = mu_delta + sigma_delta * delta_offset

        sigma_z = pm.HalfNormal('sigma_z', sigma=0.5)

        # Latent state evolution
        z_innovations = pm.Normal('z_innovations', mu=0, sigma=1,
                                   shape=(n_subj, N_BINS - 1))
        z_list = [pm.math.zeros_like(delta)]
        for b in range(1, N_BINS):
            z_next = z_list[b - 1] + delta + sigma_z * z_innovations[:, b - 1]
            z_list.append(z_next)
        z = pm.math.stack(z_list, axis=1)

        # --- Behavioral outcome: SIGN-ANCHORED (beta_behav > 0) ---
        # Anchors interpretation: higher z corresponds to more rejection
        mu_behav = pm.Normal('mu_behav', mu=0.5, sigma=0.3)
        beta_behav = pm.HalfNormal('beta_behav', sigma=1.0)  # constrained positive
        sigma_behav = pm.HalfNormal('sigma_behav', sigma=0.2)

        behav_pred = mu_behav + beta_behav * z
        pm.Normal('reject_obs', mu=behav_pred, sigma=sigma_behav,
                  observed=reject_mat)

        # --- Neural outcome (unconstrained sign) ---
        mu_neural = pm.Normal('mu_neural', mu=0.3, sigma=0.5)
        beta_neural = pm.Normal('beta_neural', mu=0, sigma=1)
        sigma_neural = pm.HalfNormal('sigma_neural', sigma=0.5)

        neural_pred = mu_neural + beta_neural * z
        pm.Normal('gap_obs', mu=neural_pred, sigma=sigma_neural,
                  observed=gap_mat)

        print("Sampling primary model...")
        trace_shared = pm.sample(**SAMPLE_KWARGS, random_seed=42)

    elapsed = time.time() - t0
    trace_shared.to_netcdf(trace_file)
    print(f"Done in {elapsed / 60:.1f} min")

# ============================================================
# Results: primary model
# ============================================================

print(f"\n{'=' * 60}")
print("Primary model results")
print("=" * 60)

summary_vars = ['mu_delta', 'sigma_delta', 'sigma_z',
                'mu_behav', 'beta_behav', 'sigma_behav',
                'mu_neural', 'beta_neural', 'sigma_neural']
summary = az.summary(trace_shared, var_names=summary_vars,
                      hdi_prob=0.95, round_to=4)
print(summary)
summary.to_csv(os.path.join(output_dir, 'posterior_summary.csv'))

# Check convergence
max_rhat = summary['r_hat'].max()
min_ess = summary['ess_bulk'].min()
print(f"\nConvergence check: max r_hat = {max_rhat:.3f}, min ess_bulk = {min_ess:.0f}")
if max_rhat > 1.05 or min_ess < 200:
    print("  WARNING: convergence may still be poor")
else:
    print("  Convergence looks good")

# Key test
beta_b_samples = trace_shared.posterior['beta_behav'].values.flatten()
beta_n_samples = trace_shared.posterior['beta_neural'].values.flatten()

beta_b_hdi = az.hdi(beta_b_samples, hdi_prob=0.95)
beta_n_hdi = az.hdi(beta_n_samples, hdi_prob=0.95)

print(f"\n--- KEY TEST: Shared latent state ---")
print(f"  beta_behav:  mean={beta_b_samples.mean():.4f}, "
      f"95% HDI=[{beta_b_hdi[0]:.4f}, {beta_b_hdi[1]:.4f}]")
print(f"  beta_neural: mean={beta_n_samples.mean():.4f}, "
      f"95% HDI=[{beta_n_hdi[0]:.4f}, {beta_n_hdi[1]:.4f}]")

# beta_behav is constrained > 0, so we only test if it differs meaningfully from 0
# (using whether the lower HDI bound is well above 0)
behav_meaningful = beta_b_hdi[0] > 0.01
neural_credible = not (beta_n_hdi[0] < 0 < beta_n_hdi[1])

if behav_meaningful and neural_credible:
    direction = "negative" if beta_n_samples.mean() < 0 else "positive"
    print(f"  ==> BOTH effects credible. Neural link is {direction}:")
    print(f"      higher latent state -> {direction} vmPFC gap change")
    if beta_n_samples.mean() < 0:
        print(f"      This is consistent with vmPFC value-discrimination collapse")
        print(f"      tracking behavioral loss aversion drift.")
elif behav_meaningful and not neural_credible:
    print(f"  ==> Behavioral link credible; neural link includes zero.")
    print(f"      Latent state captures behavioral drift but not vmPFC collapse.")
else:
    print(f"  ==> No clear evidence for the latent state mechanism.")

# Drift rate
delta_samples = trace_shared.posterior['mu_delta'].values.flatten()
delta_hdi = az.hdi(delta_samples, hdi_prob=0.95)
print(f"\n  mu_delta (drift rate): mean={delta_samples.mean():.4f}, "
      f"95% HDI=[{delta_hdi[0]:.4f}, {delta_hdi[1]:.4f}]")

# ============================================================
# Extract latent state trajectories
# ============================================================

print(f"\nExtracting latent state trajectories...")

mu_delta_post = trace_shared.posterior['mu_delta'].mean(dim=['chain', 'draw']).values
sigma_delta_post = trace_shared.posterior['sigma_delta'].mean(dim=['chain', 'draw']).values
sigma_z_post = trace_shared.posterior['sigma_z'].mean(dim=['chain', 'draw']).values
delta_offset_post = trace_shared.posterior['delta_offset'].mean(dim=['chain', 'draw']).values
z_innov_post = trace_shared.posterior['z_innovations'].mean(dim=['chain', 'draw']).values

delta_vals = mu_delta_post + sigma_delta_post * delta_offset_post

z_post = np.zeros((n_subj, N_BINS))
for b in range(1, N_BINS):
    z_post[:, b] = z_post[:, b-1] + delta_vals + sigma_z_post * z_innov_post[:, b-1]

traj_rows = []
for j, subj in enumerate(subjects):
    for b in range(N_BINS):
        traj_rows.append({
            'subject': subj, 'bin': b + 1,
            'z_posterior_mean': z_post[j, b],
            'reject_rate': reject_mat[j, b],
            'vmPFC_gap': gap_mat[j, b],
            'delta_j': delta_vals[j],
        })

traj_df = pd.DataFrame(traj_rows)
traj_df.to_csv(os.path.join(output_dir, 'state_trajectories.csv'), index=False)

z_group = z_post.mean(axis=0)
print(f"Group mean z trajectory: {np.round(z_group, 3)}")

# ============================================================
# Reset model (run-boundary partial reset)
# ============================================================

print(f"\n{'=' * 60}")
print("Model 2: Shared latent state with run-boundary reset")
print("=" * 60)

trace_reset_file = os.path.join(output_dir, 'trace_shared_reset.nc')

if os.path.exists(trace_reset_file):
    print(f"Loading cached trace: {trace_reset_file}")
    trace_reset = az.from_netcdf(trace_reset_file)
else:
    print("Building reset model...")
    t0 = time.time()

    with pm.Model() as reset_model:
        mu_delta = pm.Normal('mu_delta', mu=0, sigma=0.5)
        sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.3)
        delta_offset = pm.Normal('delta_offset', mu=0, sigma=1, shape=n_subj)
        delta = mu_delta + sigma_delta * delta_offset
        sigma_z = pm.HalfNormal('sigma_z', sigma=0.5)

        rho = pm.Beta('rho', alpha=5, beta=2)

        z_innovations = pm.Normal('z_innovations', mu=0, sigma=1,
                                   shape=(n_subj, N_BINS - 1))
        z_list = [pm.math.zeros_like(delta)]
        for b in range(1, N_BINS):
            if is_boundary[b]:
                z_next = rho * z_list[b-1] + delta + sigma_z * z_innovations[:, b-1]
            else:
                z_next = z_list[b-1] + delta + sigma_z * z_innovations[:, b-1]
            z_list.append(z_next)
        z = pm.math.stack(z_list, axis=1)

        mu_behav = pm.Normal('mu_behav', mu=0.5, sigma=0.3)
        beta_behav = pm.HalfNormal('beta_behav', sigma=1.0)  # sign anchor
        sigma_behav = pm.HalfNormal('sigma_behav', sigma=0.2)
        pm.Normal('reject_obs', mu=mu_behav + beta_behav * z,
                  sigma=sigma_behav, observed=reject_mat)

        mu_neural = pm.Normal('mu_neural', mu=0.3, sigma=0.5)
        beta_neural = pm.Normal('beta_neural', mu=0, sigma=1)
        sigma_neural = pm.HalfNormal('sigma_neural', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_neural + beta_neural * z,
                  sigma=sigma_neural, observed=gap_mat)

        print("Sampling reset model...")
        trace_reset = pm.sample(**SAMPLE_KWARGS, random_seed=44)

    elapsed = time.time() - t0
    trace_reset.to_netcdf(trace_reset_file)
    print(f"Done in {elapsed / 60:.1f} min")

# Reset model results
rho_samples = trace_reset.posterior['rho'].values.flatten()
rho_hdi = az.hdi(rho_samples, hdi_prob=0.95)
beta_b_reset = trace_reset.posterior['beta_behav'].values.flatten()
beta_n_reset = trace_reset.posterior['beta_neural'].values.flatten()
beta_b_reset_hdi = az.hdi(beta_b_reset, hdi_prob=0.95)
beta_n_reset_hdi = az.hdi(beta_n_reset, hdi_prob=0.95)

reset_summary = az.summary(trace_reset, var_names=['mu_delta', 'sigma_z', 'rho',
                                                     'beta_behav', 'beta_neural'])
print("\nReset model summary:")
print(reset_summary.round(4))

print(f"\nrho (reset parameter): mean={rho_samples.mean():.3f}, "
      f"95% HDI=[{rho_hdi[0]:.3f}, {rho_hdi[1]:.3f}]")
print("  rho=1 means no reset; rho=0 means full reset")
print(f"beta_behav (reset model):  {beta_b_reset.mean():.4f} "
      f"[{beta_b_reset_hdi[0]:.4f}, {beta_b_reset_hdi[1]:.4f}]")
print(f"beta_neural (reset model): {beta_n_reset.mean():.4f} "
      f"[{beta_n_reset_hdi[0]:.4f}, {beta_n_reset_hdi[1]:.4f}]")

# ============================================================
# Robustness: other ROIs
# ============================================================

print(f"\n{'=' * 60}")
print("Model 3+: Robustness — other ROIs as neural outcome")
print("=" * 60)

other_rois = ['L_insula', 'R_insula', 'dACC', 'L_amygdala', 'R_IFG', 'v_striatum']
robustness_results = []

if bin_all_df is not None:
    for roi in other_rois:
        loss_col = f'{roi}_loss'
        gain_col = f'{roi}_gain'

        if loss_col not in bin_all_df.columns or gain_col not in bin_all_df.columns:
            print(f"  {roi}: columns not found, skipping")
            continue

        trace_roi_file = os.path.join(output_dir, f'trace_robustness_{roi}.nc')

        if os.path.exists(trace_roi_file):
            print(f"\n{roi}: loading cached trace")
            trace_roi = az.from_netcdf(trace_roi_file)
        else:
            print(f"\n{roi}: fitting model...")
            t0 = time.time()

            roi_gap_mat = np.full((n_subj, N_BINS), np.nan)
            for _, row in bin_all_df.iterrows():
                if row['subject'] in sub_to_idx:
                    j = sub_to_idx[row['subject']]
                    b = int(row['bin']) - 1
                    roi_gap_mat[j, b] = row[gain_col] - row[loss_col]
            for b in range(N_BINS):
                if np.isnan(roi_gap_mat[:, b]).any():
                    roi_gap_mat[np.isnan(roi_gap_mat[:, b]), b] = np.nanmean(roi_gap_mat[:, b])

            with pm.Model() as roi_model:
                mu_delta = pm.Normal('mu_delta', mu=0, sigma=0.5)
                sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.3)
                delta_offset = pm.Normal('delta_offset', mu=0, sigma=1, shape=n_subj)
                delta = mu_delta + sigma_delta * delta_offset
                sigma_z = pm.HalfNormal('sigma_z', sigma=0.5)

                z_innovations = pm.Normal('z_innovations', mu=0, sigma=1,
                                           shape=(n_subj, N_BINS - 1))
                z_list = [pm.math.zeros_like(delta)]
                for b in range(1, N_BINS):
                    z_list.append(z_list[b-1] + delta + sigma_z * z_innovations[:, b-1])
                z = pm.math.stack(z_list, axis=1)

                mu_behav = pm.Normal('mu_behav', mu=0.5, sigma=0.3)
                beta_behav = pm.HalfNormal('beta_behav', sigma=1.0)  # sign anchor
                sigma_behav = pm.HalfNormal('sigma_behav', sigma=0.2)
                pm.Normal('reject_obs', mu=mu_behav + beta_behav * z,
                          sigma=sigma_behav, observed=reject_mat)

                mu_neural = pm.Normal('mu_neural', mu=0, sigma=0.5)
                beta_neural = pm.Normal('beta_neural', mu=0, sigma=1)
                sigma_neural = pm.HalfNormal('sigma_neural', sigma=0.5)
                pm.Normal('gap_obs', mu=mu_neural + beta_neural * z,
                          sigma=sigma_neural, observed=roi_gap_mat)

                trace_roi = pm.sample(**SAMPLE_KWARGS, random_seed=45)

            elapsed = time.time() - t0
            trace_roi.to_netcdf(trace_roi_file)
            print(f"  Done in {elapsed / 60:.1f} min")

        # Extract
        bn = trace_roi.posterior['beta_neural'].values.flatten()
        bn_hdi = az.hdi(bn, hdi_prob=0.95)
        credible = not (bn_hdi[0] < 0 < bn_hdi[1])

        # Convergence check
        roi_summary = az.summary(trace_roi, var_names=['beta_neural', 'beta_behav'])
        max_rhat = roi_summary['r_hat'].max()

        robustness_results.append({
            'ROI': roi,
            'beta_neural_mean': bn.mean(),
            'beta_neural_hdi_low': bn_hdi[0],
            'beta_neural_hdi_high': bn_hdi[1],
            'credibly_nonzero': credible,
            'max_rhat': max_rhat,
        })

        sig_str = "CREDIBLE" if credible else "not credible"
        conv_str = "OK" if max_rhat < 1.05 else f"r_hat={max_rhat:.2f}"
        print(f"  {roi:12s}: beta_neural={bn.mean():.4f}, "
              f"HDI=[{bn_hdi[0]:.4f}, {bn_hdi[1]:.4f}] — {sig_str} ({conv_str})")

    rob_df = pd.DataFrame(robustness_results)
    rob_df.to_csv(os.path.join(output_dir, 'robustness_results.csv'), index=False)
    print(f"\nSaved: robustness_results.csv")

    print(f"\n--- Comparison ---")
    print(f"vmPFC:        beta_neural={beta_n_samples.mean():.4f}, "
          f"HDI=[{beta_n_hdi[0]:.4f}, {beta_n_hdi[1]:.4f}]")
    for _, row in rob_df.iterrows():
        tag = " <-- CREDIBLE" if row['credibly_nonzero'] else ""
        print(f"{row['ROI']:12s}: beta_neural={row['beta_neural_mean']:.4f}, "
              f"HDI=[{row['beta_neural_hdi_low']:.4f}, {row['beta_neural_hdi_high']:.4f}]{tag}")

# ============================================================
# Figures
# ============================================================

print(f"\n{'=' * 60}")
print("Figures")
print("=" * 60)

bins_x = np.arange(1, 9)
labels = ['R1\n1st', 'R1\n2nd', 'R2\n1st', 'R2\n2nd',
          'R3\n1st', 'R3\n2nd', 'R4\n1st', 'R4\n2nd']

# Figure 1: latent state + observed vs predicted
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
z_sem = z_post.std(axis=0) / np.sqrt(n_subj)
ax.plot(bins_x, z_group, 'k-o', linewidth=2.5, markersize=8, label='z (group mean)')
ax.fill_between(bins_x, z_group - 1.96 * z_sem, z_group + 1.96 * z_sem,
                alpha=0.2, color='gray')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('Latent State z')
ax.set_title('A. Latent State Trajectory')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5)

ax = axes[1]
rej_group = reject_mat.mean(axis=0)
rej_sem = reject_mat.std(axis=0) / np.sqrt(n_subj)
beta_b_mean = beta_b_samples.mean()
mu_b_mean = trace_shared.posterior['mu_behav'].mean().values
pred_behav = mu_b_mean + beta_b_mean * z_group

ax.errorbar(bins_x, rej_group, yerr=1.96 * rej_sem,
            color='#E74C3C', marker='o', linewidth=2, capsize=3, label='Observed')
ax.plot(bins_x, pred_behav, 'k--', linewidth=2, label='Model')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('Rejection Rate')
ax.set_title('B. Behavioral Outcome')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=9)

ax = axes[2]
gap_group = gap_mat.mean(axis=0)
gap_sem = gap_mat.std(axis=0) / np.sqrt(n_subj)
beta_n_mean = beta_n_samples.mean()
mu_n_mean = trace_shared.posterior['mu_neural'].mean().values
pred_neural = mu_n_mean + beta_n_mean * z_group

ax.errorbar(bins_x, gap_group, yerr=1.96 * gap_sem,
            color='#2E86C1', marker='s', linewidth=2, capsize=3, label='Observed')
ax.plot(bins_x, pred_neural, 'k--', linewidth=2, label='Model')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('vmPFC Gap (gain - loss)')
ax.set_title('C. Neural Outcome')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=9)
ax.axhline(0, color='gray', linewidth=0.5)

plt.suptitle(f'State-Space Model: Sign-Identified (n = {n_subj})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'state_space_fit.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'state_space_fit.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/state_space_fit.png/.pdf")
plt.close()

# Figure 2: posteriors
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

az.plot_posterior(trace_shared, var_names=['beta_behav'], ax=axes[0],
                  hdi_prob=0.95, ref_val=0)
axes[0].set_title('beta_behav\n(z → rejection rate)')

az.plot_posterior(trace_shared, var_names=['beta_neural'], ax=axes[1],
                  hdi_prob=0.95, ref_val=0)
axes[1].set_title('beta_neural\n(z → vmPFC gap)')

az.plot_posterior(trace_shared, var_names=['mu_delta'], ax=axes[2],
                  hdi_prob=0.95, ref_val=0)
axes[2].set_title('mu_delta\n(drift rate)')

az.plot_posterior(trace_reset, var_names=['rho'], ax=axes[3],
                  hdi_prob=0.95, ref_val=1)
axes[3].set_title('rho (reset)\n1=no reset, 0=full')

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'posteriors.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'posteriors.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/posteriors.png/.pdf")
plt.close()

# ============================================================
# Final summary
# ============================================================

print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nPrimary model (sign-anchored):")
print(f"  beta_behav  = {beta_b_samples.mean():.4f} "
      f"[{beta_b_hdi[0]:.4f}, {beta_b_hdi[1]:.4f}] (constrained > 0)")
print(f"  beta_neural = {beta_n_samples.mean():.4f} "
      f"[{beta_n_hdi[0]:.4f}, {beta_n_hdi[1]:.4f}] "
      f"{'<-- CREDIBLE' if neural_credible else '(includes zero)'}")
print(f"  mu_delta    = {delta_samples.mean():.4f} "
      f"[{delta_hdi[0]:.4f}, {delta_hdi[1]:.4f}]")
print(f"  rho (reset) = {rho_samples.mean():.3f} "
      f"[{rho_hdi[0]:.3f}, {rho_hdi[1]:.3f}]")

print(f"\nFOR PAPER:")
if behav_meaningful and neural_credible:
    sign = "negatively" if beta_n_samples.mean() < 0 else "positively"
    print(f"  A shared latent state credibly drives both behavioral rejection")
    print(f"  rate AND vmPFC gap. The neural link is {sign} signed:")
    if beta_n_samples.mean() < 0:
        print(f"  as the latent state increases, vmPFC value discrimination")
        print(f"  decreases (gap shrinks toward zero).")
elif behav_meaningful and not neural_credible:
    print(f"  The latent state captures behavioral drift, but the link to")
    print(f"  vmPFC is not credible. Behavioral and neural effects are")
    print(f"  parallel at group level but not jointly explained by one state.")
else:
    print(f"  No clear evidence for the latent state mechanism.")

print(f"\nAll outputs saved in {output_dir}/")
print("=" * 60)
