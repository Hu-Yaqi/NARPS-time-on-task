import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 带正则化的前景理论MLE
# ============================================================

def neg_log_likelihood_reg(params, gain, loss, choice):
    """
    带正则化的负对数似然
    正则化 = 对参数加一个轻微的惩罚，防止跑到极端值
    效果类似于贝叶斯里的prior，但计算快得多
    """
    log_lam, logit_alpha, log_beta = params

    lam = np.exp(log_lam)
    alpha = expit(logit_alpha)
    beta = np.exp(log_beta)

    # 前景理论
    sv = np.power(gain, alpha) - lam * np.power(loss, alpha)
    p = expit(beta * sv)
    p = np.clip(p, 0.001, 0.999)

    # 数据的对数似然
    log_lik = choice * np.log(p) + (1 - choice) * np.log(1 - p)

    # 正则化惩罚项（相当于给参数加弱先验）
    # log_lam 的惩罚：鼓励 lambda 接近 1（即 log_lam 接近 0）
    # log_beta 的惩罚：鼓励 beta 不要太大
    # 系数 0.5 控制惩罚的强度，越大越强
    penalty = 0.5 * (log_lam ** 2) + 0.5 * (log_beta ** 2)

    return -np.sum(log_lik) + penalty


def fit_one_block(gain, loss, choice):
    """对一块数据做正则化MLE"""
    x0 = [np.log(1.2), 0.85, np.log(1.0)]

    result = minimize(
        neg_log_likelihood_reg,
        x0,
        args=(gain, loss, choice),
        method='Nelder-Mead',
        options={'maxiter': 5000}
    )

    lam = np.exp(result.x[0])
    alpha = expit(result.x[1])
    beta = np.exp(result.x[2])

    return lam, alpha, beta


# ============================================================
# 对每个被试的每个run分别做MLE
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
        print(f"已完成 {subjects.index(sub)+1}/{len(subjects)} 个被试")

rw = pd.DataFrame(results)
rw.to_csv('runwise_parameters_fixed.csv', index=False)
print(f"共拟合 {len(rw)} 个 被试×run 组合")

# ============================================================
# 检查参数范围是否合理
# ============================================================

print("\n=== 参数范围检查 ===")
for param in ['lambda', 'alpha', 'beta']:
    print(f"  {param}: min={rw[param].min():.2f}, "
          f"median={rw[param].median():.2f}, "
          f"max={rw[param].max():.2f}")

# ============================================================
# 按run的统计
# ============================================================

print("\n=== 各参数按run的平均值 ===")
run_means = rw.groupby('run')[['lambda', 'alpha', 'beta']].agg(['mean', 'sem'])
print(run_means.round(3))

# ============================================================
# 统计检验
# ============================================================

print("\n=== Run 1 vs Run 4 配对t检验 ===")
for param in ['lambda', 'alpha', 'beta']:
    r1 = rw[rw['run'] == 1].set_index('subject')[param]
    r4 = rw[rw['run'] == 4].set_index('subject')[param]
    common = r1.index.intersection(r4.index)

    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std()
    print(f"  {param}: Run1={r1.mean():.3f}, Run4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, Cohen's d={d:.2f}")

print("\n=== 线性趋势检验 ===")
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
# 可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

param_names = ['lambda', 'alpha', 'beta']
param_labels = ['λ (loss aversion)', 'α (curvature)', 'β (consistency)']
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
print("\n参数轨迹图已保存为 runwise_trajectories_fixed.png")