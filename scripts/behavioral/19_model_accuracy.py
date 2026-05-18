import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt

# ============================================================
# 对每个被试用MLE拟合四个模型，计算预测准确率
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')
subjects = sorted(df['subject'].unique())

# 四个模型的SV计算函数
def predict_model1(gain, loss, params):
    """EV: SV = gain - loss"""
    beta = np.exp(params[0])
    sv = gain - loss
    return expit(beta * sv)

def predict_model2(gain, loss, params):
    """LA: SV = gain - λ*loss"""
    lam = np.exp(params[0])
    beta = np.exp(params[1])
    sv = gain - lam * loss
    return expit(beta * sv)

def predict_model3(gain, loss, params):
    """Curvature: SV = gain^α - loss^α"""
    alpha = expit(params[0])
    beta = np.exp(params[1])
    sv = np.power(gain, alpha) - np.power(loss, alpha)
    return expit(beta * sv)

def predict_model4(gain, loss, params):
    """PT: SV = gain^α - λ*loss^α"""
    lam = np.exp(params[0])
    alpha = expit(params[1])
    beta = np.exp(params[2])
    sv = np.power(gain, alpha) - lam * np.power(loss, alpha)
    return expit(beta * sv)

def nll(params, gain, loss, choice, predict_func):
    """通用的负对数似然函数"""
    p = predict_func(gain, loss, params)
    p = np.clip(p, 0.001, 0.999)
    ll = choice * np.log(p) + (1 - choice) * np.log(1 - p)
    # 正则化
    penalty = sum(0.5 * x**2 for x in params)
    return -np.sum(ll) + penalty

# 模型配置
models = {
    'Model 1: EV': {
        'func': predict_model1,
        'x0': [np.log(1.0)],          # 只有beta
        'n_params': 1
    },
    'Model 2: LA': {
        'func': predict_model2,
        'x0': [np.log(1.2), np.log(1.0)],  # lambda, beta
        'n_params': 2
    },
    'Model 3: Curv': {
        'func': predict_model3,
        'x0': [0.85, np.log(1.0)],     # logit_alpha, beta
        'n_params': 2
    },
    'Model 4: PT': {
        'func': predict_model4,
        'x0': [np.log(1.2), 0.85, np.log(1.0)],  # lambda, logit_alpha, beta
        'n_params': 3
    }
}

# ============================================================
# 拟合并计算准确率
# ============================================================

results = []

for i, sub in enumerate(subjects):
    sub_data = df[df['subject'] == sub]
    gain = sub_data['gain'].values.astype(float)
    loss = sub_data['loss'].values.astype(float)
    choice = sub_data['accepted'].values.astype(float)

    for model_name, config in models.items():
        # 拟合
        result = minimize(
            nll, config['x0'],
            args=(gain, loss, choice, config['func']),
            method='Nelder-Mead', options={'maxiter': 5000}
        )

        # 预测
        p_pred = config['func'](gain, loss, result.x)
        # 如果P > 0.5就预测"接受"，否则预测"拒绝"
        predicted_choice = (p_pred > 0.5).astype(float)

        # 准确率 = 预测对了的trial数 / 总trial数
        accuracy = (predicted_choice == choice).mean()

        results.append({
            'subject': sub,
            'model': model_name,
            'accuracy': accuracy
        })

    if (i + 1) % 20 == 0:
        print(f"已完成 {i+1}/{len(subjects)} 个被试")

res = pd.DataFrame(results)

# ============================================================
# 汇总
# ============================================================

print("\n=== 各模型的预测准确率 ===")
summary = res.groupby('model')['accuracy'].agg(['mean', 'std', 'min', 'max', 'median'])
summary = summary.sort_values('mean', ascending=False)
print(summary.round(3))

# 和"无脑基线"比较
# 基线策略：对每个被试，总是预测他更常做的那个选择
# 比如某人接受率80%，就总是预测"接受"，准确率就是80%
baseline_acc = []
for sub in subjects:
    sub_data = df[df['subject'] == sub]
    accept_rate = sub_data['accepted'].mean()
    # 基线准确率 = max(accept_rate, 1-accept_rate)
    baseline_acc.append(max(accept_rate, 1 - accept_rate))

baseline_mean = np.mean(baseline_acc)
print(f"\n无脑基线准确率: {baseline_mean:.3f}")
print("（总是预测每个被试更常选的那个选项）")

print("\n=== 各模型 vs 基线的提升 ===")
for model_name in summary.index:
    model_acc = res[res['model'] == model_name]['accuracy'].mean()
    lift = model_acc - baseline_mean
    print(f"  {model_name}: {model_acc:.3f} (比基线高 {lift:+.3f})")

# ============================================================
# 可视化
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：四个模型的准确率箱线图
model_order = ['Model 1: EV', 'Model 2: LA', 'Model 3: Curv', 'Model 4: PT']
colors = ['#D3D1C7', '#D85A30', '#1D9E75', '#534AB7']

box_data = [res[res['model'] == m]['accuracy'].values for m in model_order]
bp = axes[0].boxplot(box_data, labels=['M1:EV', 'M2:LA', 'M3:Curv', 'M4:PT'],
                      patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0].axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline ({baseline_mean:.2f})')
axes[0].set_ylabel('Prediction accuracy')
axes[0].set_title('Model accuracy across 108 subjects')
axes[0].legend(fontsize=9)

# 图2：Model 2 vs Model 4 的逐被试比较
m2_acc = res[res['model'] == 'Model 2: LA'].set_index('subject')['accuracy']
m4_acc = res[res['model'] == 'Model 4: PT'].set_index('subject')['accuracy']

axes[1].scatter(m2_acc, m4_acc, alpha=0.5, s=30, color='#534AB7')
axes[1].plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)
axes[1].set_xlabel('Model 2 (LA) accuracy')
axes[1].set_ylabel('Model 4 (PT) accuracy')
axes[1].set_title('Model 2 vs Model 4 per subject')

# 统计在对角线上方和下方的点数
m4_better = (m4_acc > m2_acc).sum()
m2_better = (m2_acc > m4_acc).sum()
same = (m2_acc == m4_acc).sum()
axes[1].text(0.55, 0.95, f'M4 better: {m4_better}\nM2 better: {m2_better}\nTied: {same}',
             fontsize=10, transform=axes[1].transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig('model_accuracy.png', dpi=150)
print("\n图已保存为 model_accuracy.png")