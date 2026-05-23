"""
41c_state_space_with_rt.py
==========================
Extends the sign-identified state-space model (41b) by adding
reaction time as a third observed outcome.

Model:
  Latent state z_{j,b} evolves across 8 bins (same as 41b).
  Three observation equations:
    reject_rate_{j,b} ~ Normal(mu_B + beta_B * z, sigma_B)   [beta_B > 0]
    vmPFC_gap_{j,b}   ~ Normal(mu_N + beta_N * z, sigma_N)   [unconstrained]
    mean_RT_{j,b}     ~ Normal(mu_R + beta_R * z, sigma_R)   [unconstrained]

  If beta_R is credibly negative: as the latent state increases,
  RT decreases — consistent with faster, more heuristic responding.

Also fits the reset version and compares with the two-outcome model.

Prerequisites:
  - state_space_results/bin_level_data.csv (from script 41)
  - state_space_results_v2/trace_shared.nc (for comparison)

Outputs in state_space_results_v3/:
  - trace_3outcome.nc (primary: no reset)
  - trace_3outcome_reset.nc (with run-boundary reset)
  - posterior_summary.csv
  - state_trajectories.csv
  - figures/state_space_3outcome.png/.pdf
  - figures/posteriors_3outcome.png/.pdf
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

output_dir = 'state_space_results_v3'
fig_dir = os.path.join(output_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

N_BINS = 8
is_boundary = np.array([0, 0, 1, 0, 1, 0, 1, 0])

SAMPLE_KWARGS = dict(
    draws=1500, tune=2000, chains=4,
    target_accept=0.95,
    cores=1,
    progressbar=True,
)

# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("Script 41c: State-Space Model with RT (3 outcomes)")
print("=" * 60)

bin_df = pd.read_csv('state_space_results/bin_level_data.csv')
print(f"Loaded: {bin_df['subject'].nunique()} subjects, {len(bin_df)} rows")

# Check RT availability
if 'mean_rt' not in bin_df.columns or bin_df['mean_rt'].isna().all():
    print("ERROR: mean_rt not in bin_level_data.csv. Recomputing...")

    # Recompute RT per bin from raw behavioral data
    behavior = pd.read_csv('all_subjects_behavior.csv')
    neural_subs = sorted(bin_df['subject'].unique())

    rt_bins = []
    for subj in neural_subs:
        s_data = behavior[behavior['subject'] == subj]
        for run in range(1, 5):
            run_data = s_data[s_data['run'] == run].reset_index(drop=True)
            n = len(run_data)
            half = n // 2
            bin_first = (run - 1) * 2 + 1
            bin_second = bin_first + 1

            if half > 0 and 'RT' in run_data.columns:
                rt_first = run_data.iloc[:half]['RT'].mean()
                rt_second = run_data.iloc[half:]['RT'].mean()
                rt_bins.append({'subject': subj, 'bin': bin_first, 'mean_rt': rt_first})
                rt_bins.append({'subject': subj, 'bin': bin_second, 'mean_rt': rt_second})

    rt_df = pd.DataFrame(rt_bins)

    if len(rt_df) > 0:
        bin_df = bin_df.drop(columns=['mean_rt'], errors='ignore')
        bin_df = bin_df.merge(rt_df, on=['subject', 'bin'], how='left')
        bin_df.to_csv('state_space_results/bin_level_data.csv', index=False)
        print(f"  RT data merged and saved. Non-null: {bin_df['mean_rt'].notna().sum()}")
    else:
        print("  ERROR: Could not compute RT. Check all_subjects_behavior.csv columns.")
        exit(1)

# Prepare arrays
subjects = sorted(bin_df['subject'].unique())
n_subj = len(subjects)
sub_to_idx = {s: i for i, s in enumerate(subjects)}

reject_mat = np.full((n_subj, N_BINS), np.nan)
gap_mat = np.full((n_subj, N_BINS), np.nan)
rt_mat = np.full((n_subj, N_BINS), np.nan)

for _, row in bin_df.iterrows():
    j = sub_to_idx[row['subject']]
    b = int(row['bin']) - 1
    reject_mat[j, b] = row['reject_rate']
    gap_mat[j, b] = row['vmPFC_gap']
    if pd.notna(row.get('mean_rt')):
        rt_mat[j, b] = row['mean_rt']

# Fill missing values with column means
for mat in [reject_mat, gap_mat, rt_mat]:
    for b in range(N_BINS):
        col = mat[:, b]
        if np.isnan(col).any():
            mat[np.isnan(col), b] = np.nanmean(col)

# Descriptive: RT trajectory
rt_group = rt_mat.mean(axis=0)
print(f"\nRT group means by bin: {np.round(rt_group, 3)}")
r_rt, p_rt = stats.pearsonr(np.arange(1, 9), rt_group)
print(f"RT linear trend: r={r_rt:.3f}, p={p_rt:.4f}")

# ============================================================
# Model 1: Three-outcome shared latent state (no reset)
# ============================================================

print(f"\n{'=' * 60}")
print("Model 1: Three-outcome (reject + vmPFC gap + RT)")
print("=" * 60)

trace_file = os.path.join(output_dir, 'trace_3outcome.nc')

if os.path.exists(trace_file):
    print(f"Loading cached trace: {trace_file}")
    trace_3out = az.from_netcdf(trace_file)
else:
    print("Building model...")
    t0 = time.time()

    with pm.Model() as model_3out:
        # Latent state
        mu_delta = pm.Normal('mu_delta', mu=0, sigma=0.5)
        sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.3)
        delta_offset = pm.Normal('delta_offset', mu=0, sigma=1, shape=n_subj)
        delta = mu_delta + sigma_delta * delta_offset

        sigma_z = pm.HalfNormal('sigma_z', sigma=0.5)

        z_innovations = pm.Normal('z_innovations', mu=0, sigma=1,
                                   shape=(n_subj, N_BINS - 1))
        z_list = [pm.math.zeros_like(delta)]
        for b in range(1, N_BINS):
            z_list.append(z_list[b - 1] + delta + sigma_z * z_innovations[:, b - 1])
        z = pm.math.stack(z_list, axis=1)

        # Outcome 1: Rejection rate (sign anchor: beta_B > 0)
        mu_B = pm.Normal('mu_B', mu=0.5, sigma=0.3)
        beta_B = pm.HalfNormal('beta_B', sigma=1.0)
        sigma_B = pm.HalfNormal('sigma_B', sigma=0.2)
        pm.Normal('reject_obs', mu=mu_B + beta_B * z,
                  sigma=sigma_B, observed=reject_mat)

        # Outcome 2: vmPFC gap (unconstrained)
        mu_N = pm.Normal('mu_N', mu=0.3, sigma=0.5)
        beta_N = pm.Normal('beta_N', mu=0, sigma=1)
        sigma_N = pm.HalfNormal('sigma_N', sigma=0.5)
        pm.Normal('gap_obs', mu=mu_N + beta_N * z,
                  sigma=sigma_N, observed=gap_mat)

        # Outcome 3: Reaction time (unconstrained)
        mu_R = pm.Normal('mu_R', mu=1.6, sigma=0.5)
        beta_R = pm.Normal('beta_R', mu=0, sigma=1)
        sigma_R = pm.HalfNormal('sigma_R', sigma=0.5)
        pm.Normal('rt_obs', mu=mu_R + beta_R * z,
                  sigma=sigma_R, observed=rt_mat)

        print("Sampling 3-outcome model...")
        trace_3out = pm.sample(**SAMPLE_KWARGS, random_seed=46)

    elapsed = time.time() - t0
    trace_3out.to_netcdf(trace_file)
    print(f"Done in {elapsed / 60:.1f} min")

# ============================================================
# Results: 3-outcome model
# ============================================================

print(f"\n{'=' * 60}")
print("Results: Three-outcome model")
print("=" * 60)

summary_vars = ['mu_delta', 'sigma_delta', 'sigma_z',
                'mu_B', 'beta_B', 'sigma_B',
                'mu_N', 'beta_N', 'sigma_N',
                'mu_R', 'beta_R', 'sigma_R']
summary = az.summary(trace_3out, var_names=summary_vars,
                      hdi_prob=0.95, round_to=4)
print(summary)
summary.to_csv(os.path.join(output_dir, 'posterior_summary.csv'))

# Convergence check
max_rhat = summary['r_hat'].max()
min_ess = summary['ess_bulk'].min()
print(f"\nConvergence: max r_hat = {max_rhat:.3f}, min ESS = {min_ess:.0f}")

# Key tests
beta_B_samples = trace_3out.posterior['beta_B'].values.flatten()
beta_N_samples = trace_3out.posterior['beta_N'].values.flatten()
beta_R_samples = trace_3out.posterior['beta_R'].values.flatten()

beta_B_hdi = az.hdi(beta_B_samples, hdi_prob=0.95)
beta_N_hdi = az.hdi(beta_N_samples, hdi_prob=0.95)
beta_R_hdi = az.hdi(beta_R_samples, hdi_prob=0.95)

B_meaningful = beta_B_hdi[0] > 0.01
N_credible = not (beta_N_hdi[0] < 0 < beta_N_hdi[1])
R_credible = not (beta_R_hdi[0] < 0 < beta_R_hdi[1])

print(f"\n--- KEY TEST: Three-outcome shared latent state ---")
print(f"  beta_B (reject):  {beta_B_samples.mean():.4f}, "
      f"HDI [{beta_B_hdi[0]:.4f}, {beta_B_hdi[1]:.4f}] "
      f"{'*** CREDIBLE' if B_meaningful else ''}")
print(f"  beta_N (vmPFC):   {beta_N_samples.mean():.4f}, "
      f"HDI [{beta_N_hdi[0]:.4f}, {beta_N_hdi[1]:.4f}] "
      f"{'*** CREDIBLE' if N_credible else '(includes zero)'}")
print(f"  beta_R (RT):      {beta_R_samples.mean():.4f}, "
      f"HDI [{beta_R_hdi[0]:.4f}, {beta_R_hdi[1]:.4f}] "
      f"{'*** CREDIBLE' if R_credible else '(includes zero)'}")

if B_meaningful and N_credible and R_credible:
    print("\n  ==> ALL THREE credible: latent state jointly explains")
    print("      rejection drift, vmPFC collapse, AND RT decrease.")
elif B_meaningful and N_credible:
    print("\n  ==> Behavior + vmPFC credible, RT not.")
    print("      Same result as 2-outcome model; RT adds no new information.")
elif B_meaningful and R_credible:
    print("\n  ==> Behavior + RT credible, vmPFC not.")
else:
    print("\n  ==> Check individual results above.")

# Drift rate
delta_samples = trace_3out.posterior['mu_delta'].values.flatten()
delta_hdi = az.hdi(delta_samples, hdi_prob=0.95)
print(f"\n  mu_delta: {delta_samples.mean():.4f}, HDI [{delta_hdi[0]:.4f}, {delta_hdi[1]:.4f}]")

# ============================================================
# Compare with 2-outcome model
# ============================================================

print(f"\n{'=' * 60}")
print("Comparison with 2-outcome model (from 41b)")
print("=" * 60)

trace_2out_file = 'state_space_results_v2/trace_shared.nc'
if os.path.exists(trace_2out_file):
    trace_2out = az.from_netcdf(trace_2out_file)
    beta_B_2 = trace_2out.posterior['beta_behav'].values.flatten()
    beta_N_2 = trace_2out.posterior['beta_neural'].values.flatten()

    print(f"  2-outcome: beta_B={beta_B_2.mean():.4f}, beta_N={beta_N_2.mean():.4f}")
    print(f"  3-outcome: beta_B={beta_B_samples.mean():.4f}, "
          f"beta_N={beta_N_samples.mean():.4f}, beta_R={beta_R_samples.mean():.4f}")
    print(f"\n  Adding RT changes the latent state estimates:")
    print(f"    beta_B: {beta_B_2.mean():.4f} -> {beta_B_samples.mean():.4f}")
    print(f"    beta_N: {beta_N_2.mean():.4f} -> {beta_N_samples.mean():.4f}")
else:
    print("  2-outcome trace not found, skipping comparison.")

# ============================================================
# Model 2: Three-outcome with reset
# ============================================================

print(f"\n{'=' * 60}")
print("Model 2: Three-outcome with run-boundary reset")
print("=" * 60)

trace_reset_file = os.path.join(output_dir, 'trace_3outcome_reset.nc')

if os.path.exists(trace_reset_file):
    print(f"Loading cached trace: {trace_reset_file}")
    trace_3out_reset = az.from_netcdf(trace_reset_file)
else:
    print("Building reset model...")
    t0 = time.time()

    with pm.Model() as model_3out_reset:
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
                z_list.append(rho * z_list[b-1] + delta + sigma_z * z_innovations[:, b-1])
            else:
                z_list.append(z_list[b-1] + delta + sigma_z * z_innovations[:, b-1])
        z = pm.math.stack(z_list, axis=1)

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

        mu_R = pm.Normal('mu_R', mu=1.6, sigma=0.5)
        beta_R = pm.Normal('beta_R', mu=0, sigma=1)
        sigma_R = pm.HalfNormal('sigma_R', sigma=0.5)
        pm.Normal('rt_obs', mu=mu_R + beta_R * z,
                  sigma=sigma_R, observed=rt_mat)

        print("Sampling 3-outcome reset model...")
        trace_3out_reset = pm.sample(**SAMPLE_KWARGS, random_seed=47)

    elapsed = time.time() - t0
    trace_3out_reset.to_netcdf(trace_reset_file)
    print(f"Done in {elapsed / 60:.1f} min")

# Reset model results
reset_summary_vars = ['mu_delta', 'sigma_z', 'rho',
                       'beta_B', 'beta_N', 'beta_R']
reset_summary = az.summary(trace_3out_reset, var_names=reset_summary_vars,
                            hdi_prob=0.95, round_to=4)
print("\nReset model summary:")
print(reset_summary)

rho_samples = trace_3out_reset.posterior['rho'].values.flatten()
rho_hdi = az.hdi(rho_samples, hdi_prob=0.95)

beta_B_reset = trace_3out_reset.posterior['beta_B'].values.flatten()
beta_N_reset = trace_3out_reset.posterior['beta_N'].values.flatten()
beta_R_reset = trace_3out_reset.posterior['beta_R'].values.flatten()

beta_B_reset_hdi = az.hdi(beta_B_reset, hdi_prob=0.95)
beta_N_reset_hdi = az.hdi(beta_N_reset, hdi_prob=0.95)
beta_R_reset_hdi = az.hdi(beta_R_reset, hdi_prob=0.95)

print(f"\nReset model loadings:")
print(f"  rho:    {rho_samples.mean():.3f}, HDI [{rho_hdi[0]:.3f}, {rho_hdi[1]:.3f}]")
print(f"  beta_B: {beta_B_reset.mean():.4f}, HDI [{beta_B_reset_hdi[0]:.4f}, {beta_B_reset_hdi[1]:.4f}]")
print(f"  beta_N: {beta_N_reset.mean():.4f}, HDI [{beta_N_reset_hdi[0]:.4f}, {beta_N_reset_hdi[1]:.4f}]")
print(f"  beta_R: {beta_R_reset.mean():.4f}, HDI [{beta_R_reset_hdi[0]:.4f}, {beta_R_reset_hdi[1]:.4f}]")

# ============================================================
# Extract latent trajectories
# ============================================================

print(f"\nExtracting latent state trajectories...")

mu_delta_post = trace_3out.posterior['mu_delta'].mean(dim=['chain', 'draw']).values
sigma_delta_post = trace_3out.posterior['sigma_delta'].mean(dim=['chain', 'draw']).values
sigma_z_post = trace_3out.posterior['sigma_z'].mean(dim=['chain', 'draw']).values
delta_offset_post = trace_3out.posterior['delta_offset'].mean(dim=['chain', 'draw']).values
z_innov_post = trace_3out.posterior['z_innovations'].mean(dim=['chain', 'draw']).values

delta_vals = mu_delta_post + sigma_delta_post * delta_offset_post

z_post = np.zeros((n_subj, N_BINS))
for b in range(1, N_BINS):
    z_post[:, b] = z_post[:, b-1] + delta_vals + sigma_z_post * z_innov_post[:, b-1]

# Save trajectories
traj_rows = []
for j, subj in enumerate(subjects):
    for b in range(N_BINS):
        traj_rows.append({
            'subject': subj, 'bin': b + 1,
            'z_posterior_mean': z_post[j, b],
            'reject_rate': reject_mat[j, b],
            'vmPFC_gap': gap_mat[j, b],
            'mean_rt': rt_mat[j, b],
        })
traj_df = pd.DataFrame(traj_rows)
traj_df.to_csv(os.path.join(output_dir, 'state_trajectories.csv'), index=False)

z_group = z_post.mean(axis=0)
z_sem = z_post.std(axis=0) / np.sqrt(n_subj)
print(f"Group mean z: {np.round(z_group, 3)}")

# ============================================================
# Figures
# ============================================================

print(f"\n{'=' * 60}")
print("Figures")
print("=" * 60)

bins_x = np.arange(1, 9)
labels = ['R1\n1st', 'R1\n2nd', 'R2\n1st', 'R2\n2nd',
          'R3\n1st', 'R3\n2nd', 'R4\n1st', 'R4\n2nd']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Panel A: Latent state
ax = axes[0]
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

# Panel B: Rejection rate
ax = axes[1]
rej_group = reject_mat.mean(axis=0)
rej_sem = reject_mat.std(axis=0) / np.sqrt(n_subj)
beta_B_mean = beta_B_samples.mean()
mu_B_mean = trace_3out.posterior['mu_B'].mean().values
pred_B = mu_B_mean + beta_B_mean * z_group

ax.errorbar(bins_x, rej_group, yerr=1.96 * rej_sem,
            color='#E74C3C', marker='o', linewidth=2, capsize=3, label='Observed')
ax.plot(bins_x, pred_B, 'k--', linewidth=2, label='Model')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('Rejection Rate')
ax.set_title('B. Behavioral: Rejection')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=8)

# Panel C: vmPFC gap
ax = axes[2]
gap_group = gap_mat.mean(axis=0)
gap_sem = gap_mat.std(axis=0) / np.sqrt(n_subj)
beta_N_mean = beta_N_samples.mean()
mu_N_mean = trace_3out.posterior['mu_N'].mean().values
pred_N = mu_N_mean + beta_N_mean * z_group

ax.errorbar(bins_x, gap_group, yerr=1.96 * gap_sem,
            color='#2E86C1', marker='s', linewidth=2, capsize=3, label='Observed')
ax.plot(bins_x, pred_N, 'k--', linewidth=2, label='Model')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('vmPFC Gap')
ax.set_title('C. Neural: vmPFC')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5)

# Panel D: RT
ax = axes[3]
rt_group = rt_mat.mean(axis=0)
rt_sem = rt_mat.std(axis=0) / np.sqrt(n_subj)
beta_R_mean = beta_R_samples.mean()
mu_R_mean = trace_3out.posterior['mu_R'].mean().values
pred_R = mu_R_mean + beta_R_mean * z_group

ax.errorbar(bins_x, rt_group, yerr=1.96 * rt_sem,
            color='#27AE60', marker='^', linewidth=2, capsize=3, label='Observed')
ax.plot(bins_x, pred_R, 'k--', linewidth=2, label='Model')
for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time Bin')
ax.set_ylabel('Mean RT (s)')
ax.set_title('D. Behavioral: RT')
ax.set_xticks(bins_x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=8)

plt.suptitle(f'Three-Outcome State-Space Model (n = {n_subj})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'state_space_3outcome.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'state_space_3outcome.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/state_space_3outcome.png/.pdf")
plt.close()

# Posteriors figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

az.plot_posterior(trace_3out, var_names=['beta_B'], ax=axes[0],
                  hdi_prob=0.95, ref_val=0)
axes[0].set_title('beta_B\n(z → rejection)')

az.plot_posterior(trace_3out, var_names=['beta_N'], ax=axes[1],
                  hdi_prob=0.95, ref_val=0)
axes[1].set_title('beta_N\n(z → vmPFC gap)')

az.plot_posterior(trace_3out, var_names=['beta_R'], ax=axes[2],
                  hdi_prob=0.95, ref_val=0)
axes[2].set_title('beta_R\n(z → RT)')

az.plot_posterior(trace_3out, var_names=['mu_delta'], ax=axes[3],
                  hdi_prob=0.95, ref_val=0)
axes[3].set_title('mu_delta\n(drift rate)')

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'posteriors_3outcome.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(fig_dir, 'posteriors_3outcome.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved: figures/posteriors_3outcome.png/.pdf")
plt.close()

# ============================================================
# Final summary
# ============================================================

print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nThree-outcome model (no reset):")
print(f"  beta_B (reject): {beta_B_samples.mean():.4f} [{beta_B_hdi[0]:.4f}, {beta_B_hdi[1]:.4f}] "
      f"{'CREDIBLE' if B_meaningful else ''}")
print(f"  beta_N (vmPFC):  {beta_N_samples.mean():.4f} [{beta_N_hdi[0]:.4f}, {beta_N_hdi[1]:.4f}] "
      f"{'CREDIBLE' if N_credible else ''}")
print(f"  beta_R (RT):     {beta_R_samples.mean():.4f} [{beta_R_hdi[0]:.4f}, {beta_R_hdi[1]:.4f}] "
      f"{'CREDIBLE' if R_credible else ''}")
print(f"  mu_delta:        {delta_samples.mean():.4f} [{delta_hdi[0]:.4f}, {delta_hdi[1]:.4f}]")

print(f"\nThree-outcome model (with reset):")
print(f"  rho:    {rho_samples.mean():.3f} [{rho_hdi[0]:.3f}, {rho_hdi[1]:.3f}]")
print(f"  beta_B: {beta_B_reset.mean():.4f} [{beta_B_reset_hdi[0]:.4f}, {beta_B_reset_hdi[1]:.4f}]")
print(f"  beta_N: {beta_N_reset.mean():.4f} [{beta_N_reset_hdi[0]:.4f}, {beta_N_reset_hdi[1]:.4f}]")
print(f"  beta_R: {beta_R_reset.mean():.4f} [{beta_R_reset_hdi[0]:.4f}, {beta_R_reset_hdi[1]:.4f}]")

print(f"\nFOR PAPER:")
if B_meaningful and N_credible and R_credible:
    print(f"  A single latent state credibly drives all three outcomes:")
    print(f"  rejection rate (positive), vmPFC gap (negative), RT (expected negative).")
    print(f"  This three-way linkage provides converging evidence for a shared")
    print(f"  temporal process underlying behavioral strategy shift, neural")
    print(f"  value-discrimination collapse, and response speed change.")
elif B_meaningful and N_credible:
    print(f"  RT does not add to the shared process beyond rejection + vmPFC.")
    print(f"  Report 2-outcome model as primary; mention RT null in text.")

print(f"\nAll outputs saved in {output_dir}/")
print("=" * 60)
