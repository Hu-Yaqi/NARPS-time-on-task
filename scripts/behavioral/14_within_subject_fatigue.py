import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 检查：同一个被试的同一种gain-loss组合是否在不同run里重复出现？
# ============================================================

df = pd.read_csv('all_subjects_behavior.csv')

# 统计每个被试的每种gain-loss组合出现了几次
combo_counts = df.groupby(['subject', 'gain', 'loss']).size().reset_index(name='n_repeats')

print("=== 同一gain-loss组合的重复次数 ===")
print(combo_counts['n_repeats'].value_counts().sort_index())
# 如果全是1，说明每种组合只出现一次，不能做配对
# 如果有2、3、4，说明有重复，可以做配对

print(f"\n平均重复次数: {combo_counts['n_repeats'].mean():.1f}")
print(f"有重复的组合比例: {(combo_counts['n_repeats'] > 1).mean():.1%}")