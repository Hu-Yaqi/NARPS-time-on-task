"""
14_within_subject_fatigue.py
============================
Check whether the same gain-loss combinations repeat across runs
within each subject (prerequisite for matched fatigue analysis).

Outputs: printed summary of repetition counts.
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('all_subjects_behavior.csv')

# Count how many times each gain-loss combination appears per subject
combo_counts = df.groupby(['subject', 'gain', 'loss']).size().reset_index(name='n_repeats')

print("=== Repetition counts for same gain-loss combinations ===")
print(combo_counts['n_repeats'].value_counts().sort_index())
print(f"\nMean repeats: {combo_counts['n_repeats'].mean():.1f}")
print(f"Proportion with repeats: {(combo_counts['n_repeats'] > 1).mean():.1%}")
