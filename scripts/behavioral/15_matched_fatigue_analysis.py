import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('all_subjects_behavior.csv')

# ============================================================
# 思路：把每个trial按 "gain - loss" 的差值分成几档
# 然后看：同一档位的赌博，在Run 1 vs Run 4的接受率是否不同
# ============================================================

# 计算每个trial的 gain - loss 差值
df['ev_diff'] = df['gain'] - df['loss']

# 看看差值的分布范围
print("=== gain - loss 差值的范围 ===")
print(f"最小: {df['ev_diff'].min()}, 最大: {df['ev_diff'].max()}")
print(f"唯一值数量: {df['ev_diff'].nunique()}")

# 把差值分成5档
# pd.qcut 按数据的分位数来切，确保每档的数据量大致相等
# 比如最差的20%赌博一档、次差的20%一档，等等
df['ev_bin'] = pd.qcut(df['ev_diff'], q=5, labels=[
    'Very unfavorable',    # 差值最小（输远大于赢）
    'Unfavorable',
    'Neutral',             # 差值接近0
    'Favorable',
    'Very favorable'       # 差值最大（赢远大于输）
])

print("\n=== 各档位的trial数和平均差值 ===")
bin_summary = df.groupby('ev_bin', observed=True).agg(
    n_trials=('ev_diff', 'count'),
    mean_diff=('ev_diff', 'mean'),
    accept_rate=('accepted', 'mean')
)
print(bin_summary.round(2))

# ============================================================
# 核心分析：每个被试 × 每个档位 × 每个run 的接受率
# ============================================================

# 计算每个被试在每个run、每个档位的接受率
pivot = df.groupby(['subject', 'run', 'ev_bin'], observed=True)['accepted'].mean()
pivot = pivot.reset_index()

# ============================================================
# 可视化：每个档位在4个run里的接受率变化
# ============================================================

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

bins = ['Very unfavorable', 'Unfavorable', 'Neutral', 'Favorable', 'Very favorable']
colors = ['#A32D2D', '#D85A30', '#888780', '#1D9E75', '#0F6E56']

for i, (bin_name, color) in enumerate(zip(bins, colors)):
    ax = axes[i]
    bin_data = pivot[pivot['ev_bin'] == bin_name]

    # 每个被试画淡线
    for sub in df['subject'].unique():
        sub_data = bin_data[bin_data['subject'] == sub].sort_values('run')
        ax.plot(sub_data['run'], sub_data['accepted'],
                color='gray', alpha=0.05, linewidth=0.5)

    # 群体平均
    means = bin_data.groupby('run')['accepted'].mean()
    sems = bin_data.groupby('run')['accepted'].sem()
    ax.errorbar(means.index, means.values, yerr=sems.values * 1.96,
                color=color, linewidth=2.5, marker='o', markersize=7,
                capsize=4, capthick=2, zorder=10)

    ax.set_xlabel('Run', fontsize=11)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title(bin_name, fontsize=11, color=color)
    ax.set_ylim(-0.05, 1.05)

axes[0].set_ylabel('Accept rate', fontsize=12)

plt.suptitle('Accept rate across runs, by gamble favorability', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fatigue_by_ev_bin.png', dpi=150, bbox_inches='tight')
print("\n图已保存为 fatigue_by_ev_bin.png")

# ============================================================
# 统计检验：每个档位的 Run 1 vs Run 4 配对t检验
# ============================================================

print("\n=== Run 1 vs Run 4 配对t检验（按档位）===")
print(f"{'Bin':<22} {'Run1':>8} {'Run4':>8} {'Diff':>8} {'t':>8} {'p':>8} {'d':>8}")
print("-" * 75)

for bin_name in bins:
    bin_data = pivot[pivot['ev_bin'] == bin_name]

    r1 = bin_data[bin_data['run'] == 1].set_index('subject')['accepted']
    r4 = bin_data[bin_data['run'] == 4].set_index('subject')['accepted']
    common = r1.index.intersection(r4.index)

    if len(common) < 10:
        print(f"{bin_name:<22} 数据不足")
        continue

    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0

    print(f"{bin_name:<22} {r1.mean():>8.3f} {r4.mean():>8.3f} "
          f"{diff.mean():>8.3f} {t:>8.2f} {p:>8.4f} {d:>8.2f}")

# ============================================================
# 关键检验：疲劳效应在"模糊地带"更强吗？
# ============================================================

print("\n=== 疲劳效应的档位交互：中间档 vs 极端档 ===")

# 合并极端档（Very unfavorable + Very favorable）
# 合并中间档（Unfavorable + Neutral + Favorable）
# 逻辑：极端赌博不需要仔细想（明显好/明显差），
# 而中间地带的赌博需要精细评估，疲劳应该更影响后者

extreme_bins = ['Very unfavorable', 'Very favorable']
middle_bins = ['Unfavorable', 'Neutral', 'Favorable']

for label, bin_list in [('Extreme gambles', extreme_bins),
                         ('Ambiguous gambles', middle_bins)]:
    subset = pivot[pivot['ev_bin'].isin(bin_list)]
    r1 = subset[subset['run'] == 1].groupby('subject')['accepted'].mean()
    r4 = subset[subset['run'] == 4].groupby('subject')['accepted'].mean()
    common = r1.index.intersection(r4.index)

    t, p = stats.ttest_rel(r1.loc[common], r4.loc[common])
    diff = r1.loc[common] - r4.loc[common]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0

    print(f"  {label}: Run1={r1.mean():.3f}, Run4={r4.mean():.3f}, "
          f"t={t:.2f}, p={p:.4f}, d={d:.2f}")