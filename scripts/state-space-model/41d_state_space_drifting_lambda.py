"""
41d_state_space_drifting_lambda.py
==================================
State-space model where the latent state directly modulates loss
aversion (lambda) within a prospect theory framework.

Key difference from 41b:
  41b: z -> rejection_rate (bin-level Gaussian)
  41d: z -> lambda -> P(accept) per trial (trial-level Bernoulli)

Model:
  Latent state (same as 41b):
    z_{j,1} = 0
    z_{j,b} = z_{j,b-1} + delta_j + sigma_z * eps

  Behavioral (trial-level):
    lambda_{j,b} = exp(log_lambda0_j + beta_B * z_{j,b})
    SV_t = gain_t - lambda_{j,b(t)} * loss_t        [alpha=1, per M2]
    P(accept_t) = sigmoid(tau_j * SV_t)

  Neural (bin-level, same as 41b):
    vmPFC_gap_{j,b} ~ Normal(mu_N + beta_N * z_{j,b}, sigma_N)

This embeds the prospect theory model directly: the latent state
governs how lambda changes over time, and lambda governs choice
through the standard SV/softmax framework. Since M2 (alpha=1)
won the model comparison, we fix alpha=1 and only let lambda drift.

Prerequisites:
  - all_subjects_behavior.csv (trial-level choices)
  - sawtooth_statistics/individual_bin_vmPFC.csv (bin-level neural)

Outputs in state_space_results_v4/:
  - trace_drifting_lambda.nc
  - trace_drifting_lambda_reset.nc
  - posterior_summary.csv
  - state_trajectories.csv
  - lambda_trajectories.csv
  - figures/*.pdf
"""

import pandas as pd
import numpy as np
from scipy import stats
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

output_dir = 'state_space_results_v4'
fig_dir = os.path.join(output_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

N_BINS = 8
is_boundary = np.array([0, 0, 1, 0, 1, 0, 1, 0])

# ============================================================
# Load and prepare data
# ============================================================

print("=" * 60)
print("Script 41d: State-Space with Drifting Lambda")
print("=" * 60)

# --- Trial-level behavioral data ---
behavior = pd.read_csv('all_subjects_behavior.csv')

# --- Bin-level neural data ---
vmPFC = pd.read_csv('sawtooth_statistics/individual_bin_vmPFC.csv')
vmPFC['vmPFC_gap'] = vmPFC['gain_vmPFC'] - vmPFC['loss_vmPFC']

# Use only subjects with both behavioral and neural data
neural_subs = sorted(vmPFC['subject'].unique())
behavior = behavior[behavior['subject'].isin(neural_subs)].copy()

subjects = sorted(behavior['subject'].unique())
n_subj = len(subjects)
sub_to_idx = {s: i for i, s in enumerate(subjects)}

print(f"Subjects: {n_subj}")
print(f"Total trials: {len(behavior)}")

# --- Assign each trial to a time bin ---
trial_rows = []
for subj in subjects:
    s_data = behavior[behavior['subject'] == subj].copy()
    for run in range(1, 5):
        run_data = s_data[s_data['run'] == run].reset_index(drop=True)
        n = len(run_data)
        half = n // 2
        bin_first = (run - 1) * 2 + 1
        bin_second = bin_first + 1
        for i in range(n):
            b = bin_first if i < half else bin_second
            trial_rows.append({
                'subject': subj,
                'sub_idx': sub_to_idx[subj],
                'bin': b,
                'bin_idx': b - 1,  # 0-indexed
                'gain': run_data.iloc[i]['gain'],
                'loss': run_data.iloc[i]['loss'],
                'accepted': run_data.iloc[i]['accepted'],
            })

trial_df = pd.DataFrame(trial_rows)
print(f"Trial data with bins: {len(trial_df)} rows")

# --- Prepare arrays ---
sub_idx_trial = trial_df['sub_idx'].values.astype(int)
bin_idx_trial = trial_df['bin_idx'].values.astype(int)
gain_trial = trial_df['gain'].values.astype(float)
loss_trial = trial_df['loss'].values.astype(float)
choice_trial = trial_df['accepted'].values.astype(float)

# Neural: bin-level gap matrix (n_subj x N_BINS)
gap_mat = np.full((n_subj, N_BINS), np.nan)
for _, row in vmPFC.iterrows():
    if row['subject'] in sub_to_idx:
        j = sub_to_idx[row['subject']]
        b = int(row['bin']) - 1
        gap_mat[j, b] = row['vmPFC_gap']

for b in range(N_BINS):
    if np.isnan(gap_mat[:, b]).any():
        gap_mat[np.isnan(gap_mat[:, b]), b] = np.nanmean(gap_mat[:, b])

print(f"Neural gap matrix: {gap_mat.shape}")

# ============================================================
# Model 1: Drifting lambda, no reset
# ============================================================

print(f"\n{'=' * 60}")
print("Model 1: Drifting lambda (no reset)")
print("=" * 60)

trace_file = os.path.join(output_dir, 'trace_drifting_lambda.nc')

if os.path.exists(trace_file):
    print(f"Loading cached trace: {trace_file}")
    trace = az.from_netcdf(trace_file)
else:
    print("Building model...")
    t0 = time.time()

    with pm.Model() as model:
        # --- Latent state ---
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
        z = pt.stack(z_list, axis=1)  # (n_subj, N_BINS)

        # --- Behavioral: drifting lambda through prospect theory ---
        # Per-subject baseline log-lambda
        mu_log_lam = pm.Normal('mu_log_lam', mu=0.1, sigma=0.5)
        sigma_log_lam = pm.HalfNormal('sigma_log_lam', sigma=0.3)
        log_lam_offset = pm.Normal('log_lam_offset', mu=0, sigma=1, shape=n_subj)
        log_lambda0 = mu_log_lam + sigma_log_lam * log_lam_offset

        # beta_B > 0: higher z means higher lambda (more loss averse)
        beta_B = pm.HalfNormal('beta_B', sigma=1.0)

        # Lambda at each subject x bin
        # log(lambda_{j,b}) = log_lambda0_j + beta_B * z_{j,b}
        log_lambda = log_lambda0[:, None] + beta_B * z  # (n_subj, N_BINS)
        lam = pt.exp(log_lambda)

        # Per-subject inverse temperature
        mu_log_tau = pm.Normal('mu_log_tau', mu=0.5, sigma=0.5)
        sigma_log_tau = pm.HalfNormal('sigma_log_tau', sigma=0.3)
        log_tau_offset = pm.Normal('log_tau_offset', mu=0, sigma=1, shape=n_subj)
        log_tau = mu_log_tau + sigma_log_tau * log_tau_offset
        tau = pt.exp(log_tau)

        # Trial-level SV and choice probability (alpha = 1, per M2)
        lam_trial = lam[sub_idx_trial, bin_idx_trial]
        tau_trial = tau[sub_idx_trial]

        sv = gain_trial - lam_trial * loss_trial
        p_accept = pm.math.sigmoid(tau_trial * sv)
        p_accept = pt.clip(p_accept, 0.01, 0.99)

        pm.Bernoulli('choice', p=p_accept, observed=choice_trial)

        # --- Neural: bin-level vmPFC gap (same as 41b) ---
        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        beta_N = pm.Normal('beta_N', mu=0, sigma=1)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)

        neural_pred = mu_N + beta_N * z
        pm.Normal('gap_obs', mu=neural_pred, sigma=sigma_N, observed=gap_mat)

        # --- Sample ---
        print("Sampling (this will take longer due to trial-level likelihood)...")
        print(f"  {len(choice_trial)} binary observations + {n_subj * N_BINS} neural observations")

        trace = pm.sample(
            draws=1000, tune=2000, chains=4,
            target_accept=0.95,
            cores=1,
            progressbar=True,
            random_seed=48,
        )

    elapsed = time.time() - t0
    trace.to_netcdf(trace_file)
    print(f"Done in {elapsed / 60:.1f} min")

# ============================================================
# Results
# ============================================================

print(f"\n{'=' * 60}")
print("Results: Drifting lambda model")
print("=" * 60)

summary_vars = ['mu_delta', 'sigma_delta', 'sigma_z',
                'mu_log_lam', 'sigma_log_lam', 'beta_B',
                'mu_log_tau', 'sigma_log_tau',
                'mu_N', 'beta_N', 'sigma_N']
summary = az.summary(trace, var_names=summary_vars, hdi_prob=0.95, round_to=4)
print(summary)
summary.to_csv(os.path.join(output_dir, 'posterior_summary.csv'))

max_rhat = summary['r_hat'].max()
min_ess = summary['ess_bulk'].min()
print(f"\nConvergence: max r_hat = {max_rhat:.3f}, min ESS = {min_ess:.0f}")

# Key parameters
beta_B_samples = trace.posterior['beta_B'].values.flatten()
beta_N_samples = trace.posterior['beta_N'].values.flatten()
mu_log_lam_samples = trace.posterior['mu_log_lam'].values.flatten()

beta_B_hdi = az.hdi(beta_B_samples, hdi_prob=0.95)
beta_N_hdi = az.hdi(beta_N_samples, hdi_prob=0.95)

B_meaningful = beta_B_hdi[0] > 0.01
N_credible = not (beta_N_hdi[0] < 0 < beta_N_hdi[1])

print(f"\n--- KEY TEST ---")
print(f"  beta_B (z -> log lambda): {beta_B_samples.mean():.4f}, "
      f"HDI [{beta_B_hdi[0]:.4f}, {beta_B_hdi[1]:.4f}] "
      f"{'CREDIBLE' if B_meaningful else ''}")
print(f"  beta_N (z -> vmPFC gap):  {beta_N_samples.mean():.4f}, "
      f"HDI [{beta_N_hdi[0]:.4f}, {beta_N_hdi[1]:.4f}] "
      f"{'CREDIBLE' if N_credible else ''}")
print(f"  mu_log_lambda (baseline): {mu_log_lam_samples.mean():.4f} "
      f"-> lambda ~ {np.exp(mu_log_lam_samples.mean()):.3f}")

delta_samples = trace.posterior['mu_delta'].values.flatten()
delta_hdi = az.hdi(delta_samples, hdi_prob=0.95)
print(f"  mu_delta: {delta_samples.mean():.4f}, HDI [{delta_hdi[0]:.4f}, {delta_hdi[1]:.4f}]")

# Compute implied lambda trajectory
mu_delta_post = trace.posterior['mu_delta'].mean(dim=['chain', 'draw']).values
sigma_delta_post = trace.posterior['sigma_delta'].mean(dim=['chain', 'draw']).values
sigma_z_post = trace.posterior['sigma_z'].mean(dim=['chain', 'draw']).values
delta_offset_post = trace.posterior['delta_offset'].mean(dim=['chain', 'draw']).values
z_innov_post = trace.posterior['z_innovations'].mean(dim=['chain', 'draw']).values
mu_log_lam_post = trace.posterior['mu_log_lam'].mean(dim=['chain', 'draw']).values
sigma_log_lam_post = trace.posterior['sigma_log_lam'].mean(dim=['chain', 'draw']).values
log_lam_offset_post = trace.posterior['log_lam_offset'].mean(dim=['chain', 'draw']).values
beta_B_post = trace.posterior['beta_B'].mean(dim=['chain', 'draw']).values

delta_vals = mu_delta_post + sigma_delta_post * delta_offset_post
log_lam0_vals = mu_log_lam_post + sigma_log_lam_post * log_lam_offset_post

z_post = np.zeros((n_subj, N_BINS))
for b in range(1, N_BINS):
    z_post[:, b] = z_post[:, b-1] + delta_vals + sigma_z_post * z_innov_post[:, b-1]

log_lam_post = log_lam0_vals[:, None] + beta_B_post * z_post
lam_post = np.exp(log_lam_post)

z_group = z_post.mean(axis=0)
lam_group = lam_post.mean(axis=0)
lam_sem = lam_post.std(axis=0) / np.sqrt(n_subj)

print(f"\nGroup mean z trajectory: {np.round(z_group, 3)}")
print(f"Group mean lambda trajectory: {np.round(lam_group, 3)}")

# Save trajectories
traj_rows = []
for j, subj in enumerate(subjects):
    for b in range(N_BINS):
        traj_rows.append({
            'subject': subj, 'bin': b + 1,
            'z': z_post[j, b],
            'lambda': lam_post[j, b],
            'vmPFC_gap': gap_mat[j, b],
        })
traj_df = pd.DataFrame(traj_rows)
traj_df.to_csv(os.path.join(output_dir, 'state_trajectories.csv'), index=False)
traj_df.pivot(index='subject', columns='bin', values='lambda').to_csv(
    os.path.join(output_dir, 'lambda_trajectories.csv'))

# ============================================================
# Model 2: With run-boundary reset
# ============================================================

print(f"\n{'=' * 60}")
print("Model 2: Drifting lambda with reset")
print("=" * 60)

trace_reset_file = os.path.join(output_dir, 'trace_drifting_lambda_reset.nc')

if os.path.exists(trace_reset_file):
    print(f"Loading cached trace: {trace_reset_file}")
    trace_reset = az.from_netcdf(trace_reset_file)
else:
    print("Building reset model...")
    t0 = time.time()

    with pm.Model() as model_reset:
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

        mu_log_lam = pm.Normal('mu_log_lam', mu=0.1, sigma=0.5)
        sigma_log_lam = pm.HalfNormal('sigma_log_lam', sigma=0.3)
        log_lam_offset = pm.Normal('log_lam_offset', mu=0, sigma=1, shape=n_subj)
        log_lambda0 = mu_log_lam + sigma_log_lam * log_lam_offset

        beta_B = pm.HalfNormal('beta_B', sigma=1.0)
        log_lambda = log_lambda0[:, None] + beta_B * z
        lam = pt.exp(log_lambda)

        mu_log_tau = pm.Normal('mu_log_tau', mu=0.5, sigma=0.5)
        sigma_log_tau = pm.HalfNormal('sigma_log_tau', sigma=0.3)
        log_tau_offset = pm.Normal('log_tau_offset', mu=0, sigma=1, shape=n_subj)
        log_tau = mu_log_tau + sigma_log_tau * log_tau_offset
        tau = pt.exp(log_tau)

        lam_trial = lam[sub_idx_trial, bin_idx_trial]
        tau_trial = tau[sub_idx_trial]
        sv = gain_trial - lam_trial * loss_trial
        p_accept = pm.math.sigmoid(tau_trial * sv)
        p_accept = pt.clip(p_accept, 0.01, 0.99)
        pm.Bernoulli('choice', p=p_accept, observed=choice_trial)

        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        beta_N = pm.Normal('beta_N', mu=0, sigma=1)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N + beta_N * z, sigma=sigma_N, observed=gap_mat)

        print("Sampling reset model...")
        trace_reset = pm.sample(
            draws=1000, tune=2000, chains=4,
            target_accept=0.95,
            cores=1,
            progressbar=True,
            random_seed=49,
        )

    elapsed = time.time() - t0
    trace_reset.to_netcdf(trace_reset_file)
    print(f"Done in {elapsed / 60:.1f} min")

# Reset results
reset_vars = ['mu_delta', 'rho', 'beta_B', 'beta_N', 'mu_log_lam']
reset_summary = az.summary(trace_reset, var_names=reset_vars, hdi_prob=0.95, round_to=4)
print("\nReset model summary:")
print(reset_summary)

rho_samples = trace_reset.posterior['rho'].values.flatten()
rho_hdi = az.hdi(rho_samples, hdi_prob=0.95)
beta_B_r = trace_reset.posterior['beta_B'].values.flatten()
beta_N_r = trace_reset.posterior['beta_N'].values.flatten()
beta_B_r_hdi = az.hdi(beta_B_r, hdi_prob=0.95)
beta_N_r_hdi = az.hdi(beta_N_r, hdi_prob=0.95)

print(f"\nReset model key parameters:")
print(f"  rho:    {rho_samples.mean():.3f}, HDI [{rho_hdi[0]:.3f}, {rho_hdi[1]:.3f}]")
print(f"  beta_B: {beta_B_r.mean():.4f}, HDI [{beta_B_r_hdi[0]:.4f}, {beta_B_r_hdi[1]:.4f}]")
print(f"  beta_N: {beta_N_r.mean():.4f}, HDI [{beta_N_r_hdi[0]:.4f}, {beta_N_r_hdi[1]:.4f}]")

# ============================================================
# Figures
# ============================================================

print(f"\n{'=' * 60}")
print("Figures")
print("=" * 60)

bins_x = np.arange(1, 9)
labels = ['R1\n1st', 'R1\n2nd', 'R2\n1st', 'R2\n2nd',
          'R3\n1st', 'R3\n2nd', 'R4\n1st', 'R4\n2nd']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Latent state
ax = axes[0]
z_sem = z_post.std(axis=0) / np.sqrt(n_subj)
ax.plot(bins_x, z_group, 'k-o', linewidth=2.5, markersize=8)
ax.fill_between(bins_x, z_group - 1.96 * z_sem, z_group + 1.96 * z_sem,
                alpha=0.2, color='gray')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('Latent State z')
ax.set_title('A. Latent State')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5)

# Panel B: Lambda trajectory (model-implied vs MLE)
ax = axes[1]
ax.errorbar(bins_x, lam_group, yerr=1.96 * lam_sem,
            color='#E74C3C', marker='o', linewidth=2, capsize=3, label='Model-implied')

# Compare with MLE run-wise estimates (if available)
try:
    runwise = pd.read_csv('runwise_parameters_fixed.csv')
    runwise_fmri = runwise[runwise['subject'].isin(subjects)]

    # Approximate: map runs to bin pairs
    for run in range(1, 5):
        r_data = runwise_fmri[runwise_fmri['run'] == run]
        run_mean = r_data['lambda'].mean()
        bin_first = (run - 1) * 2 + 1
        bin_second = bin_first + 1
        for b in [bin_first, bin_second]:
            ax.plot(b, run_mean, 'ks', markersize=6, alpha=0.5,
                    zorder=3, label='MLE (run-wise)' if run == 1 and b == bin_first else '')
except:
    pass

for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('Loss Aversion (λ)')
ax.set_title('B. Lambda Trajectory')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=8)

# Panel C: vmPFC gap
ax = axes[2]
gap_group = gap_mat.mean(axis=0)
gap_sem = gap_mat.std(axis=0) / np.sqrt(n_subj)
beta_N_mean = beta_N_samples.mean()
mu_N_mean = trace.posterior['mu_N'].mean().values
pred_N = mu_N_mean + beta_N_mean * z_group

ax.errorbar(bins_x, gap_group, yerr=1.96 * gap_sem,
            color='#2E86C1', marker='s', linewidth=2, capsize=3, label='Observed')
ax.plot(bins_x, pred_N, 'k--', linewidth=2, label='Model')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('vmPFC Gap (gain - loss)')
ax.set_title('C. Neural: vmPFC')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5)

plt.suptitle(f'Drifting-Lambda State-Space Model (n = {n_subj})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'drifting_lambda_fit.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'drifting_lambda_fit.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/drifting_lambda_fit.png/.pdf")
plt.close()

# Posteriors
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

az.plot_posterior(trace, var_names=['beta_B'], ax=axes[0], hdi_prob=0.95, ref_val=0)
axes[0].set_title('beta_B\n(z -> log lambda)')

az.plot_posterior(trace, var_names=['beta_N'], ax=axes[1], hdi_prob=0.95, ref_val=0)
axes[1].set_title('beta_N\n(z -> vmPFC gap)')

az.plot_posterior(trace, var_names=['mu_log_lam'], ax=axes[2], hdi_prob=0.95)
axes[2].set_title('mu_log_lambda\n(baseline)')

az.plot_posterior(trace, var_names=['mu_delta'], ax=axes[3], hdi_prob=0.95, ref_val=0)
axes[3].set_title('mu_delta\n(drift rate)')

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'posteriors_drifting_lambda.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'posteriors_drifting_lambda.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/posteriors_drifting_lambda.png/.pdf")
plt.close()

# ============================================================
# Final summary
# ============================================================

print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nDrifting-lambda model (no reset):")
print(f"  beta_B (z -> log lambda): {beta_B_samples.mean():.4f} "
      f"[{beta_B_hdi[0]:.4f}, {beta_B_hdi[1]:.4f}] "
      f"{'CREDIBLE' if B_meaningful else ''}")
print(f"  beta_N (z -> vmPFC gap):  {beta_N_samples.mean():.4f} "
      f"[{beta_N_hdi[0]:.4f}, {beta_N_hdi[1]:.4f}] "
      f"{'CREDIBLE' if N_credible else ''}")
print(f"  baseline lambda:          {np.exp(mu_log_lam_samples.mean()):.3f}")
print(f"  lambda trajectory:        {np.round(lam_group, 3)}")

if os.path.exists(trace_reset_file):
    print(f"\nDrifting-lambda model (with reset):")
    print(f"  rho:    {rho_samples.mean():.3f} [{rho_hdi[0]:.3f}, {rho_hdi[1]:.3f}]")
    print(f"  beta_B: {beta_B_r.mean():.4f} [{beta_B_r_hdi[0]:.4f}, {beta_B_r_hdi[1]:.4f}]")
    print(f"  beta_N: {beta_N_r.mean():.4f} [{beta_N_r_hdi[0]:.4f}, {beta_N_r_hdi[1]:.4f}]")

print(f"\nCOMPARISON WITH 41b (bin-level rejection rate):")
print(f"  41b beta_B (z -> reject rate): 0.2356 [0.0507, 0.5622]")
print(f"  41d beta_B (z -> log lambda):  {beta_B_samples.mean():.4f} [{beta_B_hdi[0]:.4f}, {beta_B_hdi[1]:.4f}]")
print(f"  41b beta_N: -0.4499 [-1.0668, -0.0840]")
print(f"  41d beta_N: {beta_N_samples.mean():.4f} [{beta_N_hdi[0]:.4f}, {beta_N_hdi[1]:.4f}]")

print(f"\nFOR PAPER:")
if B_meaningful and N_credible:
    print(f"  The drifting-lambda model embeds prospect theory directly:")
    print(f"  a shared latent state governs both how lambda changes over")
    print(f"  time (driving trial-level choice) and how vmPFC value")
    print(f"  discrimination collapses. This formalizes the connection")
    print(f"  between SV as a trial-level latent variable and z as a")
    print(f"  session-level latent variable governing SV computation quality.")
else:
    print(f"  Check convergence and individual results above.")

print(f"\nAll outputs saved in {output_dir}/")
print("=" * 60)
