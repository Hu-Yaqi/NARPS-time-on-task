import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt

# ============================================================
# Leave-one-run-out 交叉验证
# ============================================================
# 思路：
#   轮次1: 用 Run 2,3,4 训练 → 在 Run 1 上测试
#   轮次2: 用 Run 1,3,4 训练 → 在 Run 2 上测试
#   轮次3: 用 Run 1,2,4 训练 → 在 Run 3 上测试
#   轮次4: 用 Run 1,2,3 训练 → 在 Run 4 上测试
#   最终准确率 = 四轮测试集准确率的平均
#
# 这样每个trial都恰好被预测了一次，而且是被
# "没见过它的模型"预测的，不存在过拟合

df = pd.read_csv('all_subjects_behavior.csv')
subjects = sorted(df['subject'].unique())

# ============================================================
# 模型定义（和之前一样）
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
# 交叉验证主循环
# ============================================================

results = []

for i, sub in enumerate(subjects):
    sub_data = df[df['subject'] == sub]

    for model_name, config in models.items():

        correct_predictions = []

        # 四轮交叉验证：每次留出一个run
        for held_out_run in [1, 2, 3, 4]:

            # 训练集：除了held_out_run之外的三个run
            train = sub_data[sub_data['run'] != held_out_run]
            # 测试集：被留出的那个run
            test = sub_data[sub_data['run'] == held_out_run]

            train_gain = train['gain'].values.astype(float)
            train_loss = train['loss'].values.astype(float)
            train_choice = train['accepted'].values.astype(float)

            test_gain = test['gain'].values.astype(float)
            test_loss = test['loss'].values.astype(float)
            test_choice = test['accepted'].values.astype(float)

            # 用训练集拟合参数
            result = minimize(
                nll, config['x0'],
                args=(train_gain, train_loss, train_choice, config['func']),
                method='Nelder-Mead', options={'maxiter': 5000}
            )

            # 用拟合好的参数在测试集上预测
            p_pred = config['func'](test_gain, test_loss, result.x)
            predicted = (p_pred > 0.5).astype(float)

            # 记录每个trial是否预测对了
            correct = (predicted == test_choice)
            correct_predictions.extend(correct.tolist())

        # 四轮合起来的准确率
        cv_accuracy = np.mean(correct_predictions)

        results.append({
            'subject': sub,
            'model': model_name,
            'cv_accuracy': cv_accuracy
        })

    if (i + 1) % 20 == 0:
        print(f"已完成 {i+1}/{len(subjects)} 个被试")

res = pd.DataFrame(results)

# ============================================================
# 和之前的in-sample准确率对比
# ============================================================

print("\n=== 交叉验证准确率 (out-of-sample) ===")
cv_summary = res.groupby('model')['cv_accuracy'].agg(['mean', 'std', 'median'])
cv_summary = cv_summary.sort_values('mean', ascending=False)
print(cv_summary.round(3))

# 之前的in-sample准确率
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
# 关键比较：Model 2 vs Model 4 的CV准确率
# ============================================================

from scipy import stats

m2_cv = res[res['model'] == 'Model 2: LA'].set_index('subject')['cv_accuracy']
m4_cv = res[res['model'] == 'Model 4: PT'].set_index('subject')['cv_accuracy']

t, p = stats.ttest_rel(m4_cv, m2_cv)
m4_better = (m4_cv > m2_cv).sum()
m2_better = (m2_cv > m4_cv).sum()
tied = (m2_cv == m4_cv).sum()

print(f"\n=== Model 2 vs Model 4 (交叉验证) ===")
print(f"  Model 2 CV accuracy: {m2_cv.mean():.3f}")
print(f"  Model 4 CV accuracy: {m4_cv.mean():.3f}")
print(f"  配对t检验: t={t:.2f}, p={p:.4f}")
print(f"  Model 4 better: {m4_better}, Model 2 better: {m2_better}, Tied: {tied}")

if m2_cv.mean() >= m4_cv.mean():
    print("  → Model 2 在交叉验证中表现 ≥ Model 4")
    print("  → α不仅没帮助，可能还在过拟合")

# ============================================================
# 可视化
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：in-sample vs CV 对比
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

# 图2：CV准确率的箱线图
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
print("\n图已保存为 cross_validated_accuracy.png")