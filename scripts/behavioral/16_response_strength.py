import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('all_subjects_behavior.csv')

# ============================================================
# 把四级选择编码成数值
# ============================================================

# 编码为 1-4 的量表
# 1 = strongly_reject, 2 = weakly_reject, 3 = weakly_accept, 4 = strongly_accept
strength_map = {
    'strongly_reject': 1,
    'weakly_reject': 2,
    'weakly_accept': 3,
    'strongly_accept': 4
}
df['strength'] = df['participant_response'].map(strength_map)

# 检查有没有映射失败的（NoResp之类的——但我们用的是clean数据，应该没有）
print("=== 选择强度分布 ===")
print(df['participant_response'].value_counts().sort_index())
print(f"\n映射成功率: {df['strength'].notna().mean():.1%}")

# ============================================================
# 分析1：选择强度随run的变化
# ============================================================

# 计算"极端选择比例"：strongly_accept 或 strongly_reject 的比例
df['extreme_choice'] = df['participant_response'].isin(
    ['strongly_accept', 'strongly_reject']
).astype(int)

# 计算"犹豫比例"：weakly_accept 或 weakly_reject 的比例
df['hesitant_choice'] = df['participant_response'].isin(
    ['weakly_accept', 'weakly_reject']
).astype(int)

run_strength = df.groupby(['subject', 'run']).agg(
    mean_strength=('strength', 'mean'),
    extreme_rate=('extreme_choice', 'mean'),
    hesitant_rate=('hesitant_choice', 'mean')
).reset_index()

print("\n=== 按run的选择模式 ===")
print(run_strength.groupby('run')[
    ['mean_strength', 'extreme_rate', 'hesitant_rate']
].mean().round(3))

# Run 1 vs Run 4 检验
print("\n=== Run 1 vs Run 4 ===")
for var, label in [('mean_strength', 'Mean strength (1-4)'),
                    ('extreme_rate', 'Extreme choice rate'),
                    ('hesitant_rate', 'Hesitant choice rate')]:
    r1 = run_strength[run_strength['run'] == 1].set_index('subject')[var]
    r4 = run_strength[run_strength['run'] == 4].set_index('subject')[var]
    common = r1.index.intersection(r4.index)
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    print(f"  {label}: R1={r1.mean():.3f}, R4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, d={d:.2f}")

# ============================================================
# 分析2：按EV档位看选择强度的变化
# ============================================================

df['ev_diff'] = df['gain'] - df['loss']
df['ev_bin'] = pd.qcut(df['ev_diff'], q=5, labels=[
    'Very unfavorable', 'Unfavorable', 'Neutral',
    'Favorable', 'Very favorable'
])

# 对每个档位，看Run 1 vs Run 4的extreme_choice变化
pivot = df.groupby(['subject', 'run', 'ev_bin'], observed=True).agg(
    extreme_rate=('extreme_choice', 'mean'),
    mean_strength=('strength', 'mean')
).reset_index()

bins = ['Very unfavorable', 'Unfavorable', 'Neutral',
        'Favorable', 'Very favorable']

print("\n=== 极端选择比例：Run 1 vs Run 4（按档位）===")
print(f"{'Bin':<22} {'R1_extreme':>10} {'R4_extreme':>10} {'t':>8} {'p':>8}")
print("-" * 60)

for b in bins:
    bd = pivot[pivot['ev_bin'] == b]
    r1 = bd[bd['run'] == 1].set_index('subject')['extreme_rate']
    r4 = bd[bd['run'] == 4].set_index('subject')['extreme_rate']
    common = r1.index.intersection(r4.index)
    if len(common) < 10:
        continue
    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    print(f"{b:<22} {r1.mean():>10.3f} {r4.mean():>10.3f} {t:>8.2f} {p:>8.4f}")

# ============================================================
# 可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 图1：四种选择的比例随run变化
response_types = ['strongly_reject', 'weakly_reject',
                  'weakly_accept', 'strongly_accept']
response_colors = ['#A32D2D', '#D85A30', '#1D9E75', '#0F6E56']

for resp, color in zip(response_types, response_colors):
    df[f'is_{resp}'] = (df['participant_response'] == resp).astype(int)
    means = df.groupby('run')[f'is_{resp}'].mean()
    axes[0].plot(means.index, means.values, marker='o',
                 color=color, linewidth=2, label=resp.replace('_', ' '))

axes[0].set_xlabel('Run')
axes[0].set_ylabel('Proportion')
axes[0].set_title('Response type proportions across runs')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].legend(fontsize=8)

# 图2：极端选择比例随run变化
means = run_strength.groupby('run')['extreme_rate'].mean()
sems = run_strength.groupby('run')['extreme_rate'].sem()
axes[1].errorbar(means.index, means.values, yerr=sems.values * 1.96,
                 marker='o', color='#534AB7', linewidth=2, capsize=4)
axes[1].set_xlabel('Run')
axes[1].set_ylabel('Extreme choice rate')
axes[1].set_title('"Strong" response rate across runs')
axes[1].set_xticks([1, 2, 3, 4])

# 图3：平均选择强度随run变化（按EV档位）
for b, color in zip(bins, ['#A32D2D', '#D85A30', '#888780', '#1D9E75', '#0F6E56']):
    bd = pivot[pivot['ev_bin'] == b]
    means = bd.groupby('run')['mean_strength'].mean()
    axes[2].plot(means.index, means.values, marker='o',
                 color=color, linewidth=2, label=b)

axes[2].set_xlabel('Run')
axes[2].set_ylabel('Mean strength (1-4)')
axes[2].set_title('Choice strength by gamble type')
axes[2].set_xticks([1, 2, 3, 4])
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig('response_strength_analysis.png', dpi=150)
print("\n图已保存为 response_strength_analysis.png")