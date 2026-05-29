"""
01_prepare_behavioral_data.py
=============================
Creates all_subjects_behavior.csv from raw NARPS events files.

This is the first script in the pipeline. All behavioral and
state-space scripts depend on its output.

Input:
  - data/{sub-XXX}/func/{sub-XXX}_task-MGT_run-{01-04}_events.tsv
    (from OpenNeuro ds001734)

Output:
  - all_subjects_behavior.csv

Columns in output:
  - subject: subject ID (e.g., 'sub-001')
  - run: run number (1-4)
  - onset: trial onset time (seconds)
  - duration: trial duration (seconds)
  - gain: gain amount (10-40 ILS)
  - loss: loss amount (5-20 ILS)
  - RT: reaction time (seconds)
  - participant_response: one of strongly_reject, weakly_reject,
    weakly_accept, strongly_accept, or NoResp
  - accepted: binary (1 = accept, 0 = reject; NoResp excluded)
"""

import pandas as pd
import numpy as np
import os
import glob

data_dir = 'data'
output_file = 'all_subjects_behavior.csv'

print("=" * 60)
print("Preparing behavioral data from NARPS events files")
print("=" * 60)

# Find all subjects with events files
subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub-*')))

if not subject_dirs:
    print(f"ERROR: No subject directories found in {data_dir}/")
    print(f"Please download the NARPS dataset (ds001734) and place it in {data_dir}/")
    exit(1)

all_rows = []
subjects_found = 0
subjects_skipped = 0

for sub_dir in subject_dirs:
    subject = os.path.basename(sub_dir)
    func_dir = os.path.join(sub_dir, 'func')

    if not os.path.exists(func_dir):
        subjects_skipped += 1
        continue

    has_all_runs = True
    for run in range(1, 5):
        events_file = os.path.join(
            func_dir, f'{subject}_task-MGT_run-{run:02d}_events.tsv')

        if not os.path.exists(events_file):
            has_all_runs = False
            break

        events = pd.read_csv(events_file, sep='\t')

        for _, trial in events.iterrows():
            response = trial.get('participant_response', '')

            # Skip no-response trials
            if response == 'NoResp' or pd.isna(response):
                continue

            # Map response to binary accept/reject
            if response in ['strongly_accept', 'weakly_accept']:
                accepted = 1
            elif response in ['strongly_reject', 'weakly_reject']:
                accepted = 0
            else:
                continue

            all_rows.append({
                'subject': subject,
                'run': run,
                'onset': trial['onset'],
                'duration': trial['duration'],
                'gain': trial['gain'],
                'loss': trial['loss'],
                'RT': trial.get('RT', np.nan),
                'participant_response': response,
                'accepted': accepted,
            })

    if has_all_runs:
        subjects_found += 1
    else:
        subjects_skipped += 1

df = pd.DataFrame(all_rows)
df = df.sort_values(['subject', 'run', 'onset']).reset_index(drop=True)
df.to_csv(output_file, index=False)

print(f"\nSubjects with complete data: {subjects_found}")
print(f"Subjects skipped: {subjects_skipped}")
print(f"Total valid trials: {len(df)}")
print(f"Response distribution:")
print(df['participant_response'].value_counts().to_string())
print(f"\nSaved: {output_file}")
print("=" * 60)
