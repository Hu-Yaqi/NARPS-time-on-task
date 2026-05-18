"""
31c_fix_brainmap.py
===================
重新画综合脑图，改进：
- 标题改为白底黑字
- 脑图更大，减少空白
- 整体比例更紧凑
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting, image
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
z_loss_overall = image.load_img('group_level_results/group_loss_zmap.nii.gz')
z_loss_fatigue = image.load_img('fatigue_neural_results/group_loss_fatigue_zmap.nii.gz')
z_gain_fatigue = image.load_img('fatigue_neural_results/group_gain_fatigue_zmap.nii.gz')

cut_coords = [-14, -4, 6, 14, 22, 32, 42, 54]

# 三行配置
panels = [
    {
        'img': z_loss_overall,
        'threshold': 2.3,
        'label': '(A)  Overall Loss Effect (all runs, $z_{stat}$ > 2.3)',
    },
    {
        'img': z_loss_fatigue,
        'threshold': 2.0,
        'label': '(B)  Loss Time-on-Task Change (Late − Early, $z_{stat}$ > 2.0)',
    },
    {
        'img': z_gain_fatigue,
        'threshold': 2.0,
        'label': '(C)  Gain Time-on-Task Change (Late − Early, $z_{stat}$ > 2.0)',
    },
]

# 使用nilearn直接画，不用subplot（更好控制间距）
# 方法：用figure + 手动定义axes位置

fig = plt.figure(figsize=(15, 12))

# 每个panel占图的1/3高度，留一点空间给标题
panel_height = 0.28
panel_gap = 0.05
start_y = [0.70, 0.38, 0.06]  # 三个panel的底部y坐标

for idx, panel in enumerate(panels):
    # 标题（白底黑字）
    title_y = start_y[idx] + panel_height + 0.01
    fig.text(0.03, title_y, panel['label'],
             fontsize=13, fontweight='bold', color='black',
             fontfamily='serif', va='bottom',
             bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', linewidth=0.5))

    # 脑图axes - 占据几乎全部宽度
    ax = fig.add_axes([0.02, start_y[idx], 0.96, panel_height])

    plotting.plot_stat_map(
        panel['img'],
        threshold=panel['threshold'],
        display_mode='z',
        cut_coords=cut_coords,
        axes=ax,
        colorbar=True,
        annotate=True,
        black_bg=False,
    )

# 保存
fig.savefig(os.path.join(output_dir, 'fig3c_comprehensive.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.1)
fig.savefig(os.path.join(output_dir, 'fig3c_comprehensive.png'),
            format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
print("保存: paper_figures/fig3c_comprehensive.pdf / .png")
plt.close()