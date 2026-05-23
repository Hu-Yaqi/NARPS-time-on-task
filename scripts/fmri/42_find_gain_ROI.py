import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('other_regions_results/individual_bin_all_rois.csv')
vm = pd.read_csv('sawtooth_statistics/individual_bin_vmPFC.csv')

x = np.arange(1, 9)
subjects = df['subject'].unique()
n = len(subjects)

print('GAIN SENSITIVITY TRENDS BY REGION')
print('='*70)

# vmPFC
pivot = vm.pivot(index='subject', columns='bin', values='gain_vmPFC')
slopes = []
for subj in pivot.index:
    s, _, _, _, _ = stats.linregress(x, pivot.loc[subj].values)
    slopes.append(s)
slopes = np.array(slopes)
t, p = stats.ttest_1samp(slopes, 0)
means = [pivot[b].mean() for b in range(1, 9)]
print(f'vmPFC       : slope={slopes.mean():.4f}, t={t:.3f}, p={p:.4f}  [{means[0]:.3f} -> {means[7]:.3f}]')

# Other ROIs
rois = [c.replace('_gain','') for c in df.columns if c.endswith('_gain')]
for roi in rois:
    col = f'{roi}_gain'
    pivot = df.pivot(index='subject', columns='bin', values=col)
    slopes = []
    for subj in pivot.index:
        s, _, _, _, _ = stats.linregress(x, pivot.loc[subj].values)
        slopes.append(s)
    slopes = np.array(slopes)
    t, p = stats.ttest_1samp(slopes, 0)
    means = [pivot[b].mean() for b in range(1, 9)]
    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
    print(f'{roi:12s}: slope={slopes.mean():.4f}, t={t:.3f}, p={p:.4f} {sig}  [{means[0]:.3f} -> {means[7]:.3f}]')
