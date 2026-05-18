"""
11b_runwise_parameters_fixed.py
===============================
Run-wise maximum likelihood estimation of prospect theory parameters.
For each subject and run, estimates lambda, alpha, and tau via
regularized MLE (L2 penalty on transformed parameters).

Outputs:
  - runwise_parameters_fixed.csv
  - runwise_trajectories_fixed.png
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Regularized prospect theory MLE
# ============================================================

def neg_log_likelihood_reg(params, gain, loss, choice):
    """Negative log-likelihood with L2 regularization."""
    log_lam, logit_alpha, log_beta = params

    lam = np.exp(log_lam)
    alpha = expit(logit_alpha)
    beta = np.exp(log_beta)

    sv = np.power(gain, alpha) - lam * np.power(loss, alpha)
    p = expit(beta * sv)
    p = np.clip(p, 0.001, 0.999)

    log_lik = choice * np.log(p) + (1 - choice) * np.log(1 - p)
    penalty = 0.5 * (log_lam ** 2) + 0.5 * (log_beta ** 2)

    return -np.sum(log_lik) + penalty


def fit_one_block(gain, loss, choice):
    """Fit prospect theory to a single run via regularized MLE."""
    x0 = [np.log(1.2), 0.85, np.log(1.0)]
    result = minimize(
        neg_log_likelihood_reg, x0,
        args=(gain, loss, choice),
        method='L-BFGS-B',
        bounds=[(-3, 3), (-5, 5), (-5, 5)],
        options={'maxiter': 5000}
    )
    lam = np.exp(result.x[0])
    alpha = expit(result.x[1])
    beta = np.exp(result.x[2])
    return lam, alpha, beta


# ============================================================
# Fit each subject x run
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')
subjects = sorted(df['subject'].unique())

results = []
for sub in subjects:
    for run in range(1, 5):
        block = df[(df['subject'] == sub) & (df['run'] == run)]
        if len(block) < 10:
            continue

        gain = block['gain'].values.astype(float)
        loss = block['loss'].values.astype(float)
        choice = block['accepted'].values.astype(float)

        lam, alpha, beta = fit_one_block(gain, loss, choice)
        results.append({
            'subject': sub, 'run': run,
            'lambda': lam, 'alpha': alpha, 'beta': beta
        })

    if (subjects.index(sub) + 1) % 20 == 0:
        print(f"Completed {subjects.index(sub)+1}/{len(subjects)} subjects")

rw = pd.DataFrame(results)
rw.to_csv('runwise_parameters_fixed.csv', index=False)
print(f"Fitted {len(rw)} subject x run combinations")

# ============================================================
# Parameter range check
# ============================================================

print("\n=== Parameter ranges ===")
for param in ['lambda', 'alpha', 'beta']:
    print(f"  {param}: min={rw[param].min():.2f}, "
          f"median={rw[param].median():.2f}, "
          f"max={rw[param].max():.2f}")

# ============================================================
# Run-wise means
# ============================================================

print("\n=== Parameter means by run ===")
run_means = rw.groupby('run')[['lambda', 'alpha', 'beta']].agg(['mean', 'sem'])
print(run_means.round(3))

# ============================================================
# Statistical tests
# ============================================================

print("\n=== Run 1 vs Run 4 paired t-test ===")
for param in ['lambda', 'alpha', 'beta']:
    r1 = rw[rw['run'] == 1].set_index('subject')[param]
    r4 = rw[rw['run'] == 4].set_index('subject')[param]
    common = r1.index.intersection(r4.index)
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std()
    print(f"  {param}: Run1={r1.mean():.3f}, Run4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, Cohen's d={d:.2f}")

print("\n=== Linear trend test ===")
for param in ['lambda', 'alpha', 'beta']:
    slopes = []
    for sub in subjects:
        sub_data = rw[rw['subject'] == sub].sort_values('run')
        if len(sub_data) == 4:
            slope, _, _, _, _ = stats.linregress(sub_data['run'], sub_data[param])
            slopes.append(slope)
    slopes = np.array(slopes)
    t, p = stats.ttest_1samp(slopes, 0)
    print(f"  {param}: mean slope={slopes.mean():.4f}, t={t:.2f}, p={p:.4f}")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
param_names = ['lambda', 'alpha', 'beta']
param_labels = ['λ (loss aversion)', 'α (curvature)', 'τ (consistency)']
colors = ['#534AB7', '#1D9E75', '#D85A30']

for i, (param, label, color) in enumerate(zip(param_names, param_labels, colors)):
    ax = axes[i]
    for sub in subjects:
        sub_data = rw[rw['subject'] == sub].sort_values('run')
        ax.plot(sub_data['run'], sub_data[param],
                color='gray', alpha=0.08, linewidth=0.5)

    means = rw.groupby('run')[param].mean()
    sems = rw.groupby('run')[param].sem()
    ax.errorbar(means.index, means.values, yerr=sems.values * 1.96,
                color=color, linewidth=2.5, marker='o', markersize=8,
                capsize=5, capthick=2, zorder=10)
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title(f'{label} across runs', fontsize=13)

plt.tight_layout()
plt.savefig('runwise_trajectories_fixed.png', dpi=150)
print("\nTrajectory plot saved: runwise_trajectories_fixed.png")
