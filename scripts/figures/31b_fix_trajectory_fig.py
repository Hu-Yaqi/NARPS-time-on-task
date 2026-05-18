"""
31b_fix_trajectory_fig.py
=========================
只重新画 gain vs loss vmPFC 轨迹图，图例放右上方。
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
})

COLOR_LOSS = '#C0392B'
COLOR_GAIN = '#2471A3'
COLOR_GRID = '#D5D8DC'
COLOR_RUN = '#AEB6BF'
COLOR_RUN_TEXT = '#7F8C8D'

df = pd.read_csv('gain_trajectory_results/trajectory_vmPFC.csv')
gain_mean = df['gain_mean'].values
gain_sem = df['gain_sem'].values
loss_mean = df['loss_mean'].values
loss_sem = df['loss_sem'].values

x = np.arange(1, 9)
_, _, r_loss, p_loss, _ = stats.linregress(x, loss_mean)
_, _, r_gain, p_gain, _ = stats.linregress(x, gain_mean)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.set_facecolor('white')
ax.grid(axis='y', color=COLOR_GRID, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.axhline(0, color='#BDC3C7', linewidth=0.7, zorder=1)

for b in [2.5, 4.5, 6.5]:
    ax.axvline(b, color=COLOR_RUN, linewidth=0.6, linestyle='--', alpha=0.5)

ax.errorbar(x, loss_mean, yerr=loss_sem,
            color=COLOR_LOSS, marker='o', markerfacecolor=COLOR_LOSS,
            markeredgecolor='white', markeredgewidth=1.0,
            linewidth=2.0, capsize=3, capthick=1.0, zorder=3,
            label='Loss sensitivity')

ax.errorbar(x, gain_mean, yerr=gain_sem,
            color=COLOR_GAIN, marker='s', markerfacecolor=COLOR_GAIN,
            markeredgecolor='white', markeredgewidth=1.0,
            linewidth=2.0, capsize=3, capthick=1.0, zorder=3,
            label='Gain sensitivity')

for i in range(len(x)):
    y_off_l = 8 if loss_mean[i] >= gain_mean[i] else -10
    va_l = 'bottom' if y_off_l > 0 else 'top'
    ax.annotate(f'{loss_mean[i]:.2f}',
                xy=(x[i], loss_mean[i]), fontsize=7, color=COLOR_LOSS,
                ha='center', va=va_l,
                xytext=(0, y_off_l), textcoords='offset points', alpha=0.8)

    y_off_g = 8 if gain_mean[i] > loss_mean[i] else -10
    va_g = 'bottom' if y_off_g > 0 else 'top'
    ax.annotate(f'{gain_mean[i]:.2f}',
                xy=(x[i], gain_mean[i]), fontsize=7, color=COLOR_GAIN,
                ha='center', va=va_g,
                xytext=(0, y_off_g), textcoords='offset points', alpha=0.8)

ylim = ax.get_ylim()
for i, lab in enumerate(['Run 1', 'Run 2', 'Run 3', 'Run 4']):
    ax.text(i * 2 + 1.5, ylim[1], lab,
            ha='center', va='bottom', fontsize=9, color=COLOR_RUN_TEXT,
            fontweight='semibold')

bin_labels = ['B1\n(R1 1st)', 'B2\n(R1 2nd)', 'B3\n(R2 1st)', 'B4\n(R2 2nd)',
              'B5\n(R3 1st)', 'B6\n(R3 2nd)', 'B7\n(R4 1st)', 'B8\n(R4 2nd)']
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=8, linespacing=1.3)
ax.set_xlabel('Time Bin', fontsize=12, labelpad=6)
ax.set_ylabel('vmPFC Parametric Sensitivity ($z_{\\mathrm{stat}}$)', fontsize=11, labelpad=4)

sig_l = '**' if p_loss < 0.01 else '*' if p_loss < 0.05 else 'n.s.'
sig_g = '**' if p_gain < 0.01 else '*' if p_gain < 0.05 else 'n.s.'
stats_text = (f'Loss trend: $r$ = {r_loss:.3f}, $p$ = {p_loss:.4f} {sig_l}\n'
              f'Gain trend: $r$ = {r_gain:.3f}, $p$ = {p_gain:.4f} {sig_g}')
ax.text(0.97, 0.04, stats_text, transform=ax.transAxes,
        fontsize=9, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8F9F9',
                  edgecolor='#D5D8DC', linewidth=0.6, alpha=0.95))

leg = ax.legend(loc='upper right', frameon=True, fancybox=False,
                edgecolor='#D5D8DC', framealpha=0.95)
leg.get_frame().set_linewidth(0.6)

ax.set_title('Gain vs. Loss Sensitivity Trajectory in vmPFC ($n$ = 41)',
             fontsize=13, fontweight='bold', pad=16)
ax.set_xlim(0.4, 8.6)
plt.tight_layout()

fig.savefig(os.path.join(output_dir, 'gain_vs_loss_vmPFC.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.12)
fig.savefig(os.path.join(output_dir, 'gain_vs_loss_vmPFC.png'),
            format='png', bbox_inches='tight', pad_inches=0.12, dpi=300)
print("保存: paper_figures/gain_vs_loss_vmPFC.pdf / .png")
plt.close()