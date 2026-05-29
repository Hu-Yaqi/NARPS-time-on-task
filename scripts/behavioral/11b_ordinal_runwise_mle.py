"""
11b_ordinal_runwise_mle.py
==========================
Run-wise MLE with ordinal response model (4 levels).

Instead of collapsing responses to binary (accept/reject), models
all four levels: strongly_reject, weakly_reject, weakly_accept,
strongly_accept via ordered logistic regression.

Model:
  SV = gain - lambda * loss   (alpha=1, per M2)
  P(response <= k) = sigmoid(c_k - tau * SV)
  Three ordered cutpoints: c1 < c2 < c3

Outputs:
  - runwise_parameters_ordinal.csv
  - runwise_ordinal_comparison.png (ordinal vs binary trajectories)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Ordinal response coding
# ============================================================

RESPONSE_MAP = {
    'strongly_reject': 0,
    'weakly_reject': 1,
    'weakly_accept': 2,
    'strongly_accept': 3,
}

def ordinal_log_likelihood(params, gain, loss, response):
    """
    Negative log-likelihood for ordered logistic prospect theory.

    params: [log_lam, log_tau, c1, c2_delta, c3_delta]
      c2 = c1 + exp(c2_delta)   (ensures c2 > c1)
      c3 = c2 + exp(c3_delta)   (ensures c3 > c2)
    """
    log_lam, log_tau, c1, c2_delta, c3_delta = params

    lam = np.exp(log_lam)
    tau = np.exp(log_tau)
    c2 = c1 + np.exp(c2_delta)
    c3 = c2 + np.exp(c3_delta)
    cutpoints = np.array([c1, c2, c3])

    # Subjective value (M2: alpha = 1)
    sv = gain - lam * loss
    eta = tau * sv  # linear predictor

    # Cumulative probabilities: P(response <= k) = sigmoid(c_k - eta)
    # P(response = k) = P(<=k) - P(<=k-1)
    n = len(response)
    log_lik = np.zeros(n)

    for i in range(n):
        k = int(response[i])
        if k == 0:
            p = expit(cutpoints[0] - eta[i])
        elif k == 3:
            p = 1.0 - expit(cutpoints[2] - eta[i])
        else:
            p = expit(cutpoints[k] - eta[i]) - expit(cutpoints[k-1] - eta[i])

        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        log_lik[i] = np.log(p)

    # L2 regularization on lambda and tau
    penalty = 0.5 * (log_lam ** 2) + 0.5 * (log_tau ** 2)

    return -np.sum(log_lik) + penalty


def fit_ordinal_block(gain, loss, response):
    """Fit ordinal prospect theory to one block via MLE."""
    # Initial values: c1=-1, c2=0, c3=1 (evenly spaced)
    x0 = [np.log(1.2), np.log(1.0), -1.0, np.log(1.0), np.log(1.0)]

    result = minimize(
        ordinal_log_likelihood, x0,
        args=(gain, loss, response),
        method='L-BFGS-B',
        bounds=[(-3, 3), (-3, 3), (-5, 5), (-3, 5), (-3, 5)],
        options={'maxiter': 5000}
    )

    log_lam, log_tau, c1, c2_delta, c3_delta = result.x
    lam = np.exp(log_lam)
    tau = np.exp(log_tau)
    c2 = c1 + np.exp(c2_delta)
    c3 = c2 + np.exp(c3_delta)

    return lam, tau, c1, c2, c3, result.fun


# Also fit binary for comparison
def binary_neg_ll(params, gain, loss, choice):
    log_lam, log_tau = params
    lam = np.exp(log_lam)
    tau = np.exp(log_tau)
    sv = gain - lam * loss
    p = expit(tau * sv)
    p = np.clip(p, 0.001, 0.999)
    ll = choice * np.log(p) + (1 - choice) * np.log(1 - p)
    penalty = 0.5 * (log_lam ** 2) + 0.5 * (log_tau ** 2)
    return -np.sum(ll) + penalty


def fit_binary_block(gain, loss, choice):
    result = minimize(
        binary_neg_ll, [np.log(1.2), np.log(1.0)],
        args=(gain, loss, choice),
        method='L-BFGS-B',
        bounds=[(-3, 3), (-3, 3)],
        options={'maxiter': 5000}
    )
    return np.exp(result.x[0]), np.exp(result.x[1])


# ============================================================
# Fit each subject x run
# ============================================================

print("=" * 60)
print("Ordinal Run-wise MLE (4 response levels)")
print("=" * 60)

df = pd.read_csv('all_subjects_behavior.csv')

# Map responses to ordinal codes
df['response_ord'] = df['participant_response'].map(RESPONSE_MAP)
valid = df[df['response_ord'].notna()].copy()
print(f"Valid trials with 4-level response: {len(valid)} / {len(df)}")
print(f"Response distribution:\n{valid['response_ord'].value_counts().sort_index()}")

subjects = sorted(valid['subject'].unique())

results = []
for sub in subjects:
    for run in range(1, 5):
        block = valid[(valid['subject'] == sub) & (valid['run'] == run)]
        if len(block) < 10:
            continue

        gain = block['gain'].values.astype(float)
        loss = block['loss'].values.astype(float)
        response = block['response_ord'].values.astype(float)
        choice = block['accepted'].values.astype(float)

        # Ordinal fit
        lam_ord, tau_ord, c1, c2, c3, nll_ord = fit_ordinal_block(gain, loss, response)

        # Binary fit for comparison
        lam_bin, tau_bin = fit_binary_block(gain, loss, choice)

        results.append({
            'subject': sub, 'run': run,
            'lambda_ordinal': lam_ord,
            'tau_ordinal': tau_ord,
            'c1': c1, 'c2': c2, 'c3': c3,
            'nll_ordinal': nll_ord,
            'lambda_binary': lam_bin,
            'tau_binary': tau_bin,
        })

    if (subjects.index(sub) + 1) % 20 == 0:
        print(f"Completed {subjects.index(sub)+1}/{len(subjects)} subjects")

rw = pd.DataFrame(results)
rw.to_csv('runwise_parameters_ordinal.csv', index=False)
print(f"\nFitted {len(rw)} subject x run combinations")

# ============================================================
# Compare ordinal vs binary estimates
# ============================================================

print(f"\n{'=' * 60}")
print("Comparison: Ordinal vs Binary lambda estimates")
print("=" * 60)

print("\n--- Run-wise means ---")
for method, col in [('Ordinal', 'lambda_ordinal'), ('Binary', 'lambda_binary')]:
    means = rw.groupby('run')[col].mean()
    print(f"  {method}: Run1={means[1]:.3f}, Run2={means[2]:.3f}, "
          f"Run3={means[3]:.3f}, Run4={means[4]:.3f}")

# Correlation between ordinal and binary lambda
r, p = stats.pearsonr(rw['lambda_ordinal'], rw['lambda_binary'])
print(f"\n  Correlation (ordinal vs binary lambda): r = {r:.3f}, p = {p:.6f}")

# Run 1 vs Run 4 tests
print("\n--- Run 1 vs Run 4 paired t-tests ---")
for method, col in [('Ordinal', 'lambda_ordinal'), ('Binary', 'lambda_binary')]:
    r1 = rw[rw['run'] == 1].set_index('subject')[col]
    r4 = rw[rw['run'] == 4].set_index('subject')[col]
    common = r1.index.intersection(r4.index)
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    print(f"  {method:8s} lambda: R1={r1.mean():.3f}, R4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, d={d:.2f}")

# Linear trend
print("\n--- Linear trend tests ---")
for method, col in [('Ordinal', 'lambda_ordinal'), ('Binary', 'lambda_binary')]:
    slopes = []
    for sub in subjects:
        sub_data = rw[rw['subject'] == sub].sort_values('run')
        if len(sub_data) == 4:
            slope, _, _, _, _ = stats.linregress(sub_data['run'], sub_data[col])
            slopes.append(slope)
    slopes = np.array(slopes)
    t, p = stats.ttest_1samp(slopes, 0)
    print(f"  {method:8s}: mean slope={slopes.mean():.4f}, t={t:.2f}, p={p:.4f}")

# Cutpoint trajectory
print("\n--- Cutpoint trajectories ---")
for cp in ['c1', 'c2', 'c3']:
    means = rw.groupby('run')[cp].mean()
    r1 = rw[rw['run'] == 1].set_index('subject')[cp]
    r4 = rw[rw['run'] == 4].set_index('subject')[cp]
    common = r1.index.intersection(r4.index)
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    print(f"  {cp}: R1={means[1]:.3f}, R4={means[4]:.3f}, t={t:.2f}, p={p:.4f}")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Lambda trajectories (ordinal vs binary)
ax = axes[0]
for sub in subjects:
    sub_data = rw[rw['subject'] == sub].sort_values('run')
    ax.plot(sub_data['run'], sub_data['lambda_ordinal'],
            color='gray', alpha=0.05, linewidth=0.5)

means_ord = rw.groupby('run')['lambda_ordinal'].mean()
sems_ord = rw.groupby('run')['lambda_ordinal'].sem()
means_bin = rw.groupby('run')['lambda_binary'].mean()
sems_bin = rw.groupby('run')['lambda_binary'].sem()

ax.errorbar(means_ord.index, means_ord.values, yerr=sems_ord.values * 1.96,
            color='#E74C3C', linewidth=2.5, marker='o', capsize=5, label='Ordinal', zorder=10)
ax.errorbar(means_bin.index + 0.05, means_bin.values, yerr=sems_bin.values * 1.96,
            color='#2E86C1', linewidth=2.5, marker='s', capsize=5, label='Binary', zorder=10)
ax.set_xlabel('Run')
ax.set_ylabel('Lambda')
ax.set_title('Lambda: Ordinal vs Binary')
ax.set_xticks([1, 2, 3, 4])
ax.legend()

# Panel 2: Ordinal vs binary scatter
ax = axes[1]
ax.scatter(rw['lambda_binary'], rw['lambda_ordinal'], alpha=0.3, s=20, color='#27AE60')
lims = [rw[['lambda_binary', 'lambda_ordinal']].min().min(),
        rw[['lambda_binary', 'lambda_ordinal']].max().max()]
ax.plot(lims, lims, 'k--', alpha=0.3)
ax.set_xlabel('Lambda (binary)')
ax.set_ylabel('Lambda (ordinal)')
ax.set_title(f'Ordinal vs Binary (r = {r:.3f})')

# Panel 3: Cutpoint trajectories
ax = axes[2]
colors_cp = ['#E74C3C', '#F39C12', '#27AE60']
for cp, color, label in zip(['c1', 'c2', 'c3'], colors_cp,
                              ['c1 (reject/weak_reject)', 'c2 (weak_reject/weak_accept)',
                               'c3 (weak_accept/accept)']):
    means = rw.groupby('run')[cp].mean()
    sems = rw.groupby('run')[cp].sem()
    ax.errorbar(means.index, means.values, yerr=sems.values * 1.96,
                color=color, linewidth=2, marker='o', capsize=4, label=label)
ax.set_xlabel('Run')
ax.set_ylabel('Cutpoint value')
ax.set_title('Ordinal cutpoint trajectories')
ax.set_xticks([1, 2, 3, 4])
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('runwise_ordinal_comparison.png', dpi=200, bbox_inches='tight')
print("\nSaved: runwise_ordinal_comparison.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
