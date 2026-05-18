import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# 检查疲劳效应：选择行为是否随 run 变化？
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')

# ============================================================
# 全局画图设置：更大的字体，更清晰的 PDF 字体
# ============================================================

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

runs = [1, 2, 3, 4]

# ============================================================
# 按 run 计算群体平均
# ============================================================

run_stats = df.groupby('run').agg(
    mean_RT=('RT', 'mean'),
    accept_rate=('accepted', 'mean'),
).reset_index()

print("=== 按 run 的群体统计 ===")
print(run_stats.round(3))

# ============================================================
# 每个被试在每个 run 的接受率和 RT
# ============================================================

by_sub_run = df.groupby(['subject', 'run']).agg(
    accept_rate=('accepted', 'mean'),
    mean_RT=('RT', 'mean'),
    n_trials=('accepted', 'count')
).reset_index()

# ============================================================
# Choice-EV consistency
# EV = gain - loss
# 如果 EV > 0 且接受，或者 EV < 0 且拒绝，则记为 consistent
# ============================================================

df['ev_positive'] = (df['gain'] - df['loss']) > 0

df['consistent'] = (
    ((df['ev_positive'] == True) & (df['accepted'] == 1)) |
    ((df['ev_positive'] == False) & (df['accepted'] == 0))
)

run_consist = df.groupby(['subject', 'run'])['consistent'].mean().reset_index()

# 把 consistency 合并进 by_sub_run，方便画 individual trajectories
by_sub_run = by_sub_run.merge(
    run_consist,
    on=['subject', 'run'],
    how='left'
)

# ============================================================
# 画单独图的函数
# ============================================================

def plot_run_trajectory(
    data,
    value_col,
    y_label,
    title,
    color,
    output_name,
    ylim=None
):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    # 每个被试的浅灰色轨迹线：比原来更深、更清楚
    for subject, sub_df in data.groupby('subject'):
        sub_df = sub_df.sort_values('run')
        ax.plot(
            sub_df['run'],
            sub_df[value_col],
            color='0.55',
            alpha=0.35,
            linewidth=1.2,
            zorder=1
        )

    # 群体均值和 SEM
    run_agg = data.groupby('run')[value_col].agg(['mean', 'sem'])

    ax.errorbar(
        run_agg.index,
        run_agg['mean'],
        yerr=run_agg['sem'],
        marker='o',
        markersize=7,
        linewidth=3.0,
        elinewidth=2.2,
        capsize=6,
        capthick=2.2,
        color=color,
        zorder=3
    )

    ax.set_xlabel('Run')
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=14)
    ax.set_xticks(runs)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    fig.tight_layout()
    fig.savefig(output_name, format='pdf', bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 分别保存三个单独 PDF
# ============================================================

plot_run_trajectory(
    data=by_sub_run,
    value_col='mean_RT',
    y_label='Mean RT (seconds)',
    title='Reaction time across runs',
    color='#534AB7',
    output_name='fatigue_RT.pdf'
)

plot_run_trajectory(
    data=by_sub_run,
    value_col='accept_rate',
    y_label='Accept rate',
    title='Accept rate across runs',
    color='#1D9E75',
    output_name='fatigue_accept_rate.pdf',
    ylim=(0, 1)
)

plot_run_trajectory(
    data=by_sub_run,
    value_col='consistent',
    y_label='Choice-EV consistency',
    title='Decision consistency across runs',
    color='#D85A30',
    output_name='fatigue_consistency.pdf',
    ylim=(0, 1)
)

print("\n三个单独的 PDF 已保存：")
print("  fatigue_RT.pdf")
print("  fatigue_accept_rate.pdf")
print("  fatigue_consistency.pdf")

# ============================================================
# 可选：保存为一个三页 PDF
# ============================================================

with PdfPages('fatigue_check_3pages.pdf') as pdf:

    for value_col, y_label, title, color, ylim in [
        ('mean_RT', 'Mean RT (seconds)', 'Reaction time across runs', '#534AB7', None),
        ('accept_rate', 'Accept rate', 'Accept rate across runs', '#1D9E75', (0, 1)),
        ('consistent', 'Choice-EV consistency', 'Decision consistency across runs', '#D85A30', (0, 1)),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))

        for subject, sub_df in by_sub_run.groupby('subject'):
            sub_df = sub_df.sort_values('run')
            ax.plot(
                sub_df['run'],
                sub_df[value_col],
                color='0.55',
                alpha=0.35,
                linewidth=1.2,
                zorder=1
            )

        run_agg = by_sub_run.groupby('run')[value_col].agg(['mean', 'sem'])

        ax.errorbar(
            run_agg.index,
            run_agg['mean'],
            yerr=run_agg['sem'],
            marker='o',
            markersize=7,
            linewidth=3.0,
            elinewidth=2.2,
            capsize=6,
            capthick=2.2,
            color=color,
            zorder=3
        )

        ax.set_xlabel('Run')
        ax.set_ylabel(y_label)
        ax.set_title(title, pad=14)
        ax.set_xticks(runs)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print("三页 PDF 已保存为 fatigue_check_3pages.pdf")

# ============================================================
# 统计检验：run 1 vs run 4 的配对 t 检验
# ============================================================

run1 = by_sub_run[by_sub_run['run'] == 1].set_index('subject')
run4 = by_sub_run[by_sub_run['run'] == 4].set_index('subject')
common = run1.index.intersection(run4.index)

t_rt, p_rt = stats.ttest_rel(
    run1.loc[common, 'mean_RT'],
    run4.loc[common, 'mean_RT']
)

t_acc, p_acc = stats.ttest_rel(
    run1.loc[common, 'accept_rate'],
    run4.loc[common, 'accept_rate']
)

t_consist, p_consist = stats.ttest_rel(
    run1.loc[common, 'consistent'],
    run4.loc[common, 'consistent']
)

print(f"\nRun1 vs Run4 配对 t 检验：")
print(f"  RT: t={t_rt:.2f}, p={p_rt:.3f}")
print(f"    Run1 平均 RT: {run1['mean_RT'].mean():.3f}s")
print(f"    Run4 平均 RT: {run4['mean_RT'].mean():.3f}s")

print(f"  Accept rate: t={t_acc:.2f}, p={p_acc:.3f}")
print(f"    Run1 平均接受率: {run1['accept_rate'].mean():.3f}")
print(f"    Run4 平均接受率: {run4['accept_rate'].mean():.3f}")

print(f"  Choice-EV consistency: t={t_consist:.2f}, p={p_consist:.3f}")
print(f"    Run1 平均一致率: {run1['consistent'].mean():.3f}")
print(f"    Run4 平均一致率: {run4['consistent'].mean():.3f}")