"""
05_prospect_theory_model.py
===========================
Single-subject Bayesian prospect theory estimation (sub-001).
Estimates lambda (loss aversion), alpha (diminishing sensitivity),
and tau (inverse temperature) using PyMC NUTS sampling.

Outputs:
  - prospect_theory_sub001.png: posterior distributions
  - model_vs_data_sub001.png: predicted vs actual choice map
"""

import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data preparation: subject 001
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')
sub001 = df[df['subject'] == 'sub-001'].copy()

gain = sub001['gain'].values
loss = sub001['loss'].values
choice = sub001['accepted'].values

n_trials = len(choice)
print(f"Subject sub-001: {n_trials} trials")
print(f"Accept count: {choice.sum()}, accept rate: {choice.mean():.1%}")

# ============================================================
# Bayesian prospect theory model
# ============================================================

with pm.Model() as pt_model:

    # Priors
    lam = pm.LogNormal('lambda', mu=0.5, sigma=0.5)      # loss aversion
    alpha = pm.Beta('alpha', alpha=2, beta=2)              # value function curvature
    beta = pm.HalfNormal('beta', sigma=1)                  # inverse temperature

    # Subjective value: SV = gain^alpha - lambda * loss^alpha
    sv = gain ** alpha - lam * (loss ** alpha)

    # Choice probability via softmax
    p_accept = pm.math.sigmoid(beta * sv)
    p_accept = pm.math.clip(p_accept, 0.01, 0.99)

    # Likelihood
    y = pm.Bernoulli('y', p=p_accept, observed=choice)

    # MCMC sampling
    print("\nStarting MCMC sampling...")
    trace = pm.sample(
        draws=2000, chains=4, random_seed=42, target_accept=0.9
    )

# ============================================================
# Results
# ============================================================

print("\n=== Posterior summary ===")
summary = az.summary(trace, var_names=['lambda', 'alpha', 'beta'])
print(summary)

# Posterior distribution plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
az.plot_posterior(trace, var_names=['lambda'], ax=axes[0])
axes[0].set_title('λ (loss aversion)')
az.plot_posterior(trace, var_names=['alpha'], ax=axes[1])
axes[1].set_title('α (curvature)')
az.plot_posterior(trace, var_names=['beta'], ax=axes[2])
axes[2].set_title('τ (consistency)')
plt.tight_layout()
plt.savefig('prospect_theory_sub001.png', dpi=150)
print("\nPosterior plot saved: prospect_theory_sub001.png")

# ============================================================
# Decision map: model prediction vs actual data
# ============================================================

lam_est = float(trace.posterior['lambda'].mean())
alpha_est = float(trace.posterior['alpha'].mean())
beta_est = float(trace.posterior['beta'].mean())

print(f"\nPosterior means: λ={lam_est:.2f}, α={alpha_est:.2f}, τ={beta_est:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: actual choices
ax = axes[0]
acc = sub001[sub001['accepted'] == 1]
rej = sub001[sub001['accepted'] == 0]
ax.scatter(acc['gain'], acc['loss'], color='#1D9E75', alpha=0.6, s=40, label='Accepted')
ax.scatter(rej['gain'], rej['loss'], color='#D85A30', alpha=0.6, s=40, label='Rejected')
ax.set_xlabel('Potential gain')
ax.set_ylabel('Potential loss')
ax.set_title('Actual choices (sub-001)')
ax.legend()

# Right: model prediction
ax = axes[1]
gain_grid = np.linspace(10, 40, 30)
loss_grid = np.linspace(5, 20, 30)
G, L = np.meshgrid(gain_grid, loss_grid)
SV = G ** alpha_est - lam_est * (L ** alpha_est)
P = 1 / (1 + np.exp(-beta_est * SV))
c = ax.contourf(G, L, P, levels=20, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(c, ax=ax, label='P(accept)')
ax.set_xlabel('Potential gain')
ax.set_ylabel('Potential loss')
ax.set_title('Model prediction (sub-001)')

plt.tight_layout()
plt.savefig('model_vs_data_sub001.png', dpi=150)
print("Decision map saved: model_vs_data_sub001.png")
