"""
17_model_comparison_by_half.py
===============================
Compare Model 2 (LA) vs Model 4 (full PT) separately for early
(Runs 1-2) and late (Runs 3-4) halves. Tests whether alpha's
contribution shrinks in the second half.

Outputs:
  - model_comparison_by_half.csv
  - model_comparison_by_half.png
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
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')

# ============================================================
# ============================================================

def nll_model2(params, gain, loss, choice):
    """Model 2: SV = gain - λ*loss"""
    log_lam, log_beta = params
    lam = np.exp(log_lam)
    beta = np.exp(log_beta)

    sv = gain - lam * loss
    p = expit(beta * sv)
    p = np.clip(p, 0.001, 0.999)
    ll = choice * np.log(p) + (1 - choice) * np.log(1 - p)

    penalty = 0.5 * (log_lam ** 2) + 0.5 * (log_beta ** 2)
    return -np.sum(ll) + penalty


def nll_model4(params, gain, loss, choice):
    """Model 4: SV = gain^α - λ*loss^α"""
    log_lam, logit_alpha, log_beta = params
    lam = np.exp(log_lam)
    alpha = expit(logit_alpha)
    beta = np.exp(log_beta)

    sv = np.power(gain, alpha) - lam * np.power(loss, alpha)
    p = expit(beta * sv)
    p = np.clip(p, 0.001, 0.999)
    ll = choice * np.log(p) + (1 - choice) * np.log(1 - p)

    penalty = 0.5 * (log_lam ** 2) + 0.5 * (log_beta ** 2)
    return -np.sum(ll) + penalty


def fit_model2(gain, loss, choice):
    result = minimize(nll_model2, [np.log(1.2), np.log(1.0)],
                      args=(gain, loss, choice),
                      method='L-BFGS-B', bounds=[(-3, 3), (-5, 5)], options={'maxiter': 5000})
    # BIC = k*ln(n) - 2*ln(L)
    k = 2
    n = len(choice)
    nll = nll_model2(result.x, gain, loss, choice)
    bic = k * np.log(n) + 2 * nll
    return bic, nll


def fit_model4(gain, loss, choice):
    result = minimize(nll_model4, [np.log(1.2), 0.85, np.log(1.0)],
                      args=(gain, loss, choice),
                      method='L-BFGS-B', bounds=[(-3, 3), (-5, 5), (-5, 5)], options={'maxiter': 5000})
    k = 3
    n = len(choice)
    nll = nll_model4(result.x, gain, loss, choice)
    bic = k * np.log(n) + 2 * nll

    alpha = expit(result.x[1])
    return bic, nll, alpha


# ============================================================
# ============================================================

subjects = sorted(df['subject'].unique())
results = []

for i, sub in enumerate(subjects):
    sub_data = df[df['subject'] == sub]

    for half_label, runs in [('Early (Run 1-2)', [1, 2]),
                              ('Late (Run 3-4)', [3, 4])]:
        half_data = sub_data[sub_data['run'].isin(runs)]

        gain = half_data['gain'].values.astype(float)
        loss = half_data['loss'].values.astype(float)
        choice = half_data['accepted'].values.astype(float)

        bic2, nll2 = fit_model2(gain, loss, choice)
        bic4, nll4, alpha_est = fit_model4(gain, loss, choice)

        results.append({
            'subject': sub,
            'half': half_label,
            'BIC_model2': bic2,
            'BIC_model4': bic4,
            'BIC_advantage_M4': bic2 - bic4,
            'alpha': alpha_est
        })

    if (i + 1) % 20 == 0:
        print(f"Completed {i+1}/{len(subjects)}  subjects")

res = pd.DataFrame(results)
res.to_csv('model_comparison_by_half.csv', index=False)

# ============================================================
# ============================================================

print("\n=== Alpha estimates (early vs late)===")
for half in ['Early (Run 1-2)', 'Late (Run 3-4)']:
    alphas = res[res['half'] == half]['alpha']
    print(f"  {half}: median α = {alphas.median():.3f}, "
          f"mean α = {alphas.mean():.3f}")

print("\n=== Model 4 BIC advantage over Model 2 ===")
print("  (Positive = M4 better, Negative = M2 better)")
for half in ['Early (Run 1-2)', 'Late (Run 3-4)']:
    advantages = res[res['half'] == half]['BIC_advantage_M4']
    n_m4_wins = (advantages > 0).sum()
    n_total = len(advantages)
    print(f"\n  {half}:")
    print(f"    Mean BIC advantage: {advantages.mean():.1f}")
    print(f"    Median BIC advantage: {advantages.median():.1f}")
    print(f"    Subjects where M4 wins: {n_m4_wins}/{n_total} ({n_m4_wins/n_total:.0%})")

early_adv = res[res['half'] == 'Early (Run 1-2)'].set_index('subject')['BIC_advantage_M4']
late_adv = res[res['half'] == 'Late (Run 3-4)'].set_index('subject')['BIC_advantage_M4']
common = early_adv.index.intersection(late_adv.index)

t, p = stats.ttest_rel(early_adv.loc[common], late_adv.loc[common])
print(f"\n=== M4 advantage: early vs late half ===")
print(f"  Early half mean BIC advantage: {early_adv.mean():.1f}")
print(f"  Late half mean BIC advantage: {late_adv.mean():.1f}")
print(f"  Paired t-test: t={t:.2f}, p={p:.4f}")

if early_adv.mean() > late_adv.mean() and p < 0.05:
    print("  → Confirmed: M4 advantage comes mainly from early half")
elif p >= 0.05:
    print("  → Early vs late difference not significant")
else:
    print("  → M4 advantage is larger in late half")

# ============================================================
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

early_alpha = res[res['half'] == 'Early (Run 1-2)']['alpha']
late_alpha = res[res['half'] == 'Late (Run 3-4)']['alpha']
axes[0].hist(early_alpha, bins=20, alpha=0.6, color='#534AB7', label='Run 1-2', edgecolor='white')
axes[0].hist(late_alpha, bins=20, alpha=0.6, color='#D85A30', label='Run 3-4', edgecolor='white')
axes[0].set_xlabel('α (curvature)')
axes[0].set_title('α distribution: early vs late')
axes[0].legend()

axes[1].hist(early_adv, bins=20, alpha=0.6, color='#534AB7', label='Run 1-2', edgecolor='white')
axes[1].hist(late_adv, bins=20, alpha=0.6, color='#D85A30', label='Run 3-4', edgecolor='white')
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('BIC advantage of Model 4 over Model 2')
axes[1].set_title('Does α help? Early vs late')
axes[1].legend()

early_a = res[res['half'] == 'Early (Run 1-2)'].set_index('subject')['alpha']
late_a = res[res['half'] == 'Late (Run 3-4)'].set_index('subject')['alpha']
axes[2].scatter(early_a.loc[common], late_a.loc[common],
                alpha=0.5, color='#1D9E75', s=30)
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2].set_xlabel('α in Run 1-2')
axes[2].set_ylabel('α in Run 3-4')
axes[2].set_title('α: early vs late (each dot = 1 subject)')

plt.tight_layout()
plt.savefig('model_comparison_by_half.png', dpi=150)
print("\nFigure saved: model_comparison_by_half.png")