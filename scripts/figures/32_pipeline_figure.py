"""
32_pipeline_figure.py
=====================
Generate a pipeline diagram showing the full analysis workflow.
Output: paper_figures/pipeline.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia', 'serif'],
    'font.size': 9,
})

fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis('off')

# ============================================================
# Color scheme
# ============================================================
C_DATA = '#E8F0FE'       # light blue - data
C_DATA_EDGE = '#4285F4'
C_BEHAV = '#FEF3E0'      # warm cream - behavioral
C_BEHAV_EDGE = '#E8912D'
C_NEURAL = '#E8F5E9'     # light green - neural
C_NEURAL_EDGE = '#34A853'
C_RESULT = '#FCE4EC'     # light pink - key findings
C_RESULT_EDGE = '#D93025'
C_METHOD = '#F3E5F5'     # light purple - methods
C_METHOD_EDGE = '#9C27B0'
C_ARROW = '#555555'
C_TITLE_BG = '#37474F'
C_TITLE_FG = 'white'

# ============================================================
# Helper functions
# ============================================================

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=8,
             fontweight='normal', text_color='#222222', alpha=1.0, style='round'):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.12",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            zorder=3, linespacing=1.4)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='->', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_section_title(x, y, w, text):
    box = FancyBboxPatch((x, y), w, 0.38,
                          boxstyle="round,pad=0.08",
                          facecolor=C_TITLE_BG, edgecolor='none',
                          linewidth=0, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + 0.19, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color=C_TITLE_FG, zorder=3)

# ============================================================
# DATA section (top)
# ============================================================

draw_section_title(0.3, 7.8, 13.4, 'NARPS Dataset (OpenNeuro ds001734)')

draw_box(0.5, 7.05, 3.8, 0.6,
         'Behavioral Data\nn = 108, 256 trials/subject\n4 runs × 64 trials',
         C_DATA, C_DATA_EDGE, fontsize=7.5)

draw_box(5.0, 7.05, 3.8, 0.6,
         'fMRI Data\nn = 41, TR = 1 s\nfMRIPrep preprocessed',
         C_DATA, C_DATA_EDGE, fontsize=7.5)

draw_box(9.5, 7.05, 4.2, 0.6,
         'Task: Mixed Gamble\nAccept/Reject 50/50 gambles\nGain: 10–40, Loss: 5–20 ILS',
         C_DATA, C_DATA_EDGE, fontsize=7.5)

# ============================================================
# BEHAVIORAL ANALYSIS (left column)
# ============================================================

draw_section_title(0.3, 6.2, 6.0, 'Behavioral Analysis')

# Part A
draw_box(0.5, 5.15, 2.7, 0.85,
         'Part A: Model Comparison\n\n4 nested PT models\nHierarchical Bayes (NUTS)\nWAIC + BIC + CV',
         C_BEHAV, C_BEHAV_EDGE, fontsize=7)

draw_box(3.5, 5.15, 2.7, 0.85,
         'Finding A\n\n$\\lambda$ essential (90.2% CV)\n$\\alpha$ adds nothing\nTriple convergence',
         C_RESULT, C_RESULT_EDGE, fontsize=7, fontweight='bold')

draw_arrow(3.2, 5.57, 3.5, 5.57)

# Part B
draw_box(0.5, 3.85, 2.7, 1.0,
         'Part B: Time-on-Task\n\nRun-wise MLE\n$\\hat{\\lambda}_{j,r}$, $\\hat{\\alpha}_{j,r}$, $\\hat{\\tau}_{j,r}$\nPaired t-tests, Cohen\'s d\nEV-bin selectivity analysis',
         C_BEHAV, C_BEHAV_EDGE, fontsize=7)

draw_box(3.5, 3.85, 2.7, 1.0,
         'Finding B\n\n$\\lambda$↑ (d=0.39, p<.0001)\n$\\alpha$→1 (d=0.40, p<.0001)\nSelective: ambiguous\ngambles most affected',
         C_RESULT, C_RESULT_EDGE, fontsize=7, fontweight='bold')

draw_arrow(3.2, 4.35, 3.5, 4.35)

# Arrow from Part A to Part B
draw_arrow(1.85, 5.15, 1.85, 4.85)

# Arrow from data to behavioral
draw_arrow(2.4, 7.05, 2.4, 6.58)

# ============================================================
# NEURAL ANALYSIS (right column)
# ============================================================

draw_section_title(7.0, 6.2, 6.7, 'Neural Analysis (n = 41)')

# Part C
draw_box(7.2, 5.15, 2.8, 0.85,
         'Part C: Gain/Loss GLM\n\nFirst-level: parametric\nmodulators (gain, loss)\nGroup: one-sample t-test\nCluster correction',
         C_NEURAL, C_NEURAL_EDGE, fontsize=7)

draw_box(10.3, 5.15, 3.2, 0.85,
         'Finding C\n\nLoss: widespread (23,318 mm³)\nGain: weak (2,064 mm³)\nLoss >> Gain neurally',
         C_RESULT, C_RESULT_EDGE, fontsize=7, fontweight='bold')

draw_arrow(10.0, 5.57, 10.3, 5.57)

# Part D
draw_box(7.2, 2.6, 2.8, 2.35,
         'Part D: Neural\nTime-on-Task\n\nD1: Early vs Late GLM\nD2: Loss×Trial interaction\nD3: Brain-behavior\n      correlation (Δλ vs\n      ΔvmPFC)\nD4: 8-bin trajectory\n      (gain + loss)',
         C_NEURAL, C_NEURAL_EDGE, fontsize=7)

draw_box(10.3, 2.6, 3.2, 2.35,
         'Finding D\n\nvmPFC loss↑ (r=.854**)\nvmPFC gain↓ (r=-.870**)\nDouble dissociation\n\nSawtooth pattern:\nwithin-run ↑\nbetween-run reset\n\nBrain-behavior:\nr=-.017, n.s.',
         C_RESULT, C_RESULT_EDGE, fontsize=7, fontweight='bold')

draw_arrow(10.0, 3.77, 10.3, 3.77)

# Arrow from Part C to Part D
draw_arrow(8.6, 5.15, 8.6, 4.95)

# Arrow from data to neural
draw_arrow(6.9, 7.05, 9.3, 6.58)

# ============================================================
# CONVERGENCE / CONCLUSION (bottom)
# ============================================================

draw_section_title(2.5, 1.6, 9.0, 'Convergence')

draw_box(2.7, 0.4, 8.6, 1.05,
         'Time-on-task selectively amplifies loss processing while attenuating gain processing.\n'
         'Behavioral: $\\lambda$↑, $\\alpha$→1, concentrated on ambiguous gambles.\n'
         'Neural: vmPFC loss sensitivity increases gradually (sawtooth with partial resets).\n'
         'The effect is channel-specific, gradual, and partially reversible.',
         '#FFF8E1', '#F9A825', fontsize=8.5, fontweight='bold')

# Arrows from findings to convergence
draw_arrow(4.85, 3.85, 4.85, 2.05, lw=1.5)
draw_arrow(11.9, 2.6, 11.9, 2.05, lw=1.5)

# ============================================================
# Method labels (small, next to arrows)
# ============================================================

ax.text(1.85, 4.98, 'identifies\nkey parameters', ha='center', va='center',
        fontsize=6, color='#777777', style='italic')

ax.text(8.6, 5.03, 'establishes\nneural baseline', ha='center', va='center',
        fontsize=6, color='#777777', style='italic')

# ============================================================
# Save
# ============================================================

fig.savefig(os.path.join(output_dir, 'pipeline.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.2)
fig.savefig(os.path.join(output_dir, 'pipeline.png'),
            format='png', bbox_inches='tight', pad_inches=0.2, dpi=250)
print("保存: paper_figures/pipeline.pdf / .png")
plt.close()