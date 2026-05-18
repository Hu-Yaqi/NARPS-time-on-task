"""
20_cross_validated_accuracy.py
=============================
Leave-one-run-out cross-validation for four prospect theory models.
Each run is held out once; parameters trained on remaining three runs.
Key result: M2 and M4 achieve identical CV accuracy (90.2%).

Outputs:
  - cross_validated_accuracy.png
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt

# ============================================================
# ============================================================
#

df = pd.read_csv('all_subjects_behavior.csv')
subjects = sorted(df['subject'].unique())

# ============================================================
# ============================================================

def predict_model1(gain, loss, params):
    beta = np.exp(params[0])
    sv = gain - loss
    return expit(beta * sv)

def predict_model2(gain, loss, params):
    lam = np.exp(params[0])
    beta = np.exp(params[1])
    sv = gain - lam * loss
    return expit(beta * sv)

def predict_model3(gain, loss, params):
    alpha = expit(params[0])
    beta = np.exp(params[1])
    sv = np.power(gain, alpha) - np.power(loss, alpha)
    return expit(beta * sv)

def predict_model4(gain, loss, params):
    lam = np.exp(params[0])
    alpha = expit(params[1])
    beta = np.exp(params[2])
    sv = np.power(gain, alpha) - lam * np.power(loss, alpha)
    return expit(beta * sv)

def nll(params, gain, loss, choice, predict_func):
    p = predict_func(gain, loss, params)
    p = np.clip(p, 0.001, 0.999)
    ll = choice * np.log(p) + (1 - choice) * np.log(1 - p)
    penalty = sum(0.5 * x**2 for x in params)
    return -np.sum(ll) + penalty

models = {
    'Model 1: EV': {
        'func': predict_model1,
        'x0': [np.log(1.0)],
        'n_params': 1
    },
    'Model 2: LA': {
        'func': predict_model2,
        'x0': [np.log(1.2), np.log(1.0)],
        'n_params': 2
    },
    'Model 3: Curv': {
        'func': predict_model3,
        'x0': [0.85, np.log(1.0)],
        'n_params': 2
    },
    'Model 4: PT': {
        'func': predict_model4,
        'x0': [np.log(1.2), 0.85, np.log(1.0)],
        'n_params': 3
    }
}

# ============================================================
# ============================================================

results = []

for i, sub in enumerate(subjects):
    sub_data = df[df['subject'] == sub]

    for model_name, config in models.items():

        correct_predictions = []

        for held_out_run in [1, 2, 3, 4]:

            train = sub_data[sub_data['run'] != held_out_run]
            test = sub_data[sub_data['run'] == held_out_run]

            train_gain = train['gain'].values.astype(float)
            train_loss = train['loss'].values.astype(float)
            train_choice = train['accepted'].values.astype(float)

            test_gain = test['gain'].values.astype(float)
            test_loss = test['loss'].values.astype(float)
            test_choice = test['accepted'].values.astype(float)

            result = minimize(
                nll, config['x0'],
                args=(train_gain, train_loss, train_choice, config['func']),
                method='L-BFGS-B',
                bounds=[(-3, 3)] + [(-5, 5)] * (len(config['x0']) - 1),
                options={'maxiter': 5000}
            )

            p_pred = config['func'](test_gain, test_loss, result.x)
            predicted = (p_pred > 0.5).astype(float)

            correct = (predicted == test_choice)
            correct_predictions.extend(correct.tolist())

        cv_accuracy = np.mean(correct_predictions)

        results.append({
            'subject': sub,
            'model': model_name,
            'cv_accuracy': cv_accuracy
        })

    if (i + 1) % 20 == 0:
        print(f"Completed {i+1}/{len(subjects)}  subjects")

res = pd.DataFrame(results)

# ============================================================
# ============================================================

print("\n=== Cross-validated accuracy (out-of-sample) ===")
cv_summary = res.groupby('model')['cv_accuracy'].agg(['mean', 'std', 'median'])
cv_summary = cv_summary.sort_values('mean', ascending=False)
print(cv_summary.round(3))

insample = {'Model 4: PT': 0.909, 'Model 2: LA': 0.909,
            'Model 1: EV': 0.811, 'Model 3: Curv': 0.811}

print("\n=== In-sample vs Cross-validated ===")
print(f"{'Model':<20} {'In-sample':>10} {'CV':>10} {'Drop':>10}")
print("-" * 52)
for model in cv_summary.index:
    cv = cv_summary.loc[model, 'mean']
    ins = insample.get(model, 0)
    drop = ins - cv
    print(f"{model:<20} {ins:>10.3f} {cv:>10.3f} {drop:>10.3f}")

# ============================================================
# ============================================================

from scipy import stats

m2_cv = res[res['model'] == 'Model 2: LA'].set_index('subject')['cv_accuracy']
m4_cv = res[res['model'] == 'Model 4: PT'].set_index('subject')['cv_accuracy']

t, p = stats.ttest_rel(m4_cv, m2_cv)
m4_better = (m4_cv > m2_cv).sum()
m2_better = (m2_cv > m4_cv).sum()
tied = (m2_cv == m4_cv).sum()

print(f"\n=== Model 2 vs Model 4 (cross-validation) ===")
print(f"  Model 2 CV accuracy: {m2_cv.mean():.3f}")
print(f"  Model 4 CV accuracy: {m4_cv.mean():.3f}")
print(f"  Paired t-test: t={t:.2f}, p={p:.4f}")
print(f"  Model 4 better: {m4_better}, Model 2 better: {m2_better}, Tied: {tied}")

if m2_cv.mean() >= m4_cv.mean():
    print("  → Model 2 CV performance >= Model 4")
    print("  → alpha adds nothing; possible overfitting")

# ============================================================
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_order = ['Model 1: EV', 'Model 2: LA', 'Model 3: Curv', 'Model 4: PT']
short_names = ['M1:EV', 'M2:LA', 'M3:Curv', 'M4:PT']
colors = ['#D3D1C7', '#D85A30', '#1D9E75', '#534AB7']

x = np.arange(len(model_order))
width = 0.35

insample_vals = [insample[m] for m in model_order]
cv_vals = [res[res['model'] == m]['cv_accuracy'].mean() for m in model_order]

axes[0].bar(x - width/2, insample_vals, width, label='In-sample',
            color=[c + '99' for c in colors], edgecolor=colors)
axes[0].bar(x + width/2, cv_vals, width, label='Cross-validated',
            color=colors, edgecolor=colors)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('In-sample vs cross-validated accuracy')
axes[0].legend()
axes[0].set_ylim(0.7, 0.95)

box_data = [res[res['model'] == m]['cv_accuracy'].values for m in model_order]
bp = axes[1].boxplot(box_data, labels=short_names,
                      patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1].set_ylabel('Cross-validated accuracy')
axes[1].set_title('CV accuracy across 108 subjects')

plt.tight_layout()
plt.savefig('cross_validated_accuracy.png', dpi=150)
print("\nFigure saved: cross_validated_accuracy.png")