"""
23_group_level_analysis.py
==========================
Group-level (second-level) analysis: one-sample t-tests on
first-level gain and loss z-maps with cluster-level correction.

Outputs:
  - group_level_results/group_{gain,loss}_zmap.nii.gz
  - group_level_results/group_{gain,loss}_corrected.nii.gz
  - group_level_results/{gain,loss}_clusters.csv
  - group_level_results/*.png
"""

import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.reporting import get_clusters_table
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ============================================================

results_dir = 'first_level_results'
group_dir = 'group_level_results'
os.makedirs(group_dir, exist_ok=True)

# ============================================================
# ============================================================

print("=" * 60)
print("Group-Level Analysis")
print("=" * 60)

gain_maps = sorted([
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith('_gain_zmap.nii.gz')
])
loss_maps = sorted([
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith('_loss_zmap.nii.gz')
])

gain_subs = [os.path.basename(f).split('_')[0] for f in gain_maps]
loss_subs = [os.path.basename(f).split('_')[0] for f in loss_maps]
assert gain_subs == loss_subs, "Gain and loss subject lists do not match!"

n_subjects = len(gain_maps)
print(f"\nFound {n_subjects} subjects with z-maps:")
for s in gain_subs:
    print(f"  {s}")

if n_subjects < 3:
    print("\n⚠️  Fewer than 3 subjects; statistical tests unreliable. Recommend 5+.")
    print("Continuing, but interpret with caution.")

# ============================================================
# ============================================================

design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})

print(f"\nDesign matrix（{n_subjects} × 1）：testing group mean != 0")

print("\n" + "─" * 40)
print("Analyzing GAIN effect...")

second_level_gain = SecondLevelModel(smoothing_fwhm=None)

second_level_gain.fit(gain_maps, design_matrix=design_matrix)

z_map_gain_group = second_level_gain.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)
print("  Gain group z-map computed")

print("\n" + "─" * 40)
print("Analyzing LOSS effect...")

second_level_loss = SecondLevelModel(smoothing_fwhm=None)
second_level_loss.fit(loss_maps, design_matrix=design_matrix)

z_map_loss_group = second_level_loss.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score'
)
print("  Loss group z-map computed")

# ============================================================
# ============================================================
#

print("\n" + "─" * 40)
print("Multiple comparison correction（Cluster-Level）...")

cluster_threshold = 2.3

thresholded_gain, threshold_gain = threshold_stats_img(
    z_map_gain_group,
    alpha=0.05,
    height_control='fpr',
    cluster_threshold=10,
)

thresholded_loss, threshold_loss = threshold_stats_img(
    z_map_loss_group,
    alpha=0.05,
    height_control='fpr',
    cluster_threshold=10,
)

print(f"  Gain threshold: z = {threshold_gain:.2f}")
print(f"  Loss threshold: z = {threshold_loss:.2f}")

# ============================================================
# ============================================================

print("\n" + "─" * 40)
print("Cluster report")


print("\n===== GAIN effect (positive = stronger activation with larger gain)=====")
try:
    gain_table = get_clusters_table(
        z_map_gain_group,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(gain_table) > 0:
        print(gain_table.to_string())
        gain_table.to_csv(os.path.join(group_dir, 'gain_clusters.csv'), index=False)
    else:
        print("  No significant clusters found")
except Exception as e:
    print(f"  Cluster extraction error: {e}")

print("\n===== LOSS effect (positive = stronger activation with larger loss)=====")
try:
    loss_table = get_clusters_table(
        z_map_loss_group,
        stat_threshold=cluster_threshold,
        min_distance=8
    )
    if len(loss_table) > 0:
        print(loss_table.to_string())
        loss_table.to_csv(os.path.join(group_dir, 'loss_clusters.csv'), index=False)
    else:
        print("  No significant clusters found")
except Exception as e:
    print(f"  Cluster extraction error: {e}")

# ============================================================
# ============================================================

print("\n" + "─" * 40)
print("Generating figures...")

fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    z_map_gain_group,
    threshold=cluster_threshold,
    title=f'Group GAIN effect (n={n_subjects}, z>{cluster_threshold})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[0],
)

plotting.plot_stat_map(
    z_map_loss_group,
    threshold=cluster_threshold,
    title=f'Group LOSS effect (n={n_subjects}, z>{cluster_threshold})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes1[1],
)

plt.tight_layout()
fig1.savefig(os.path.join(group_dir, 'group_gain_loss_zmaps.png'), dpi=150)
print("  Saved: group_gain_loss_zmaps.png")

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

plotting.plot_stat_map(
    thresholded_gain,
    threshold=0.1,
    title=f'Group GAIN (cluster-corrected, n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes2[0],
)

plotting.plot_stat_map(
    thresholded_loss,
    threshold=0.1,
    title=f'Group LOSS (cluster-corrected, n={n_subjects})',
    display_mode='z',
    cut_coords=[-10, -2, 6, 14, 22, 30, 42, 54],
    axes=axes2[1],
)

plt.tight_layout()
fig2.savefig(os.path.join(group_dir, 'group_gain_loss_corrected.png'), dpi=150)
print("  Saved: group_gain_loss_corrected.png")

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

plotting.plot_glass_brain(
    z_map_gain_group,
    threshold=cluster_threshold,
    title=f'Gain (z>{cluster_threshold})',
    axes=axes3[0],
    display_mode='lyrz',
    colorbar=True,
)

plotting.plot_glass_brain(
    z_map_loss_group,
    threshold=cluster_threshold,
    title=f'Loss (z>{cluster_threshold})',
    axes=axes3[1],
    display_mode='lyrz',
    colorbar=True,
)

plt.tight_layout()
fig3.savefig(os.path.join(group_dir, 'group_glass_brain.png'), dpi=150)
print("  Saved: group_glass_brain.png")

# ============================================================
# ============================================================

z_map_gain_group.to_filename(os.path.join(group_dir, 'group_gain_zmap.nii.gz'))
z_map_loss_group.to_filename(os.path.join(group_dir, 'group_loss_zmap.nii.gz'))
thresholded_gain.to_filename(os.path.join(group_dir, 'group_gain_corrected.nii.gz'))
thresholded_loss.to_filename(os.path.join(group_dir, 'group_loss_corrected.nii.gz'))
print("\nGroup z-maps saved")

# ============================================================
# ============================================================

print(f"\n{'=' * 60}")
print("Group-Level Analysis Done！")
print(f"{'=' * 60}")
print(f"N subjects: {n_subjects}")
print(f"Cluster threshold: z > {cluster_threshold}")
print(f"Results saved in: {group_dir}/")
print(f"\nNext steps:")
print(f"  1. Review group effects")
print(f"  2. Review corrected clusters")
print(f"  3. Check gain_clusters.csv / loss_clusters.csv for cluster details")
print(f"  4. Run neural time-on-task contrast")