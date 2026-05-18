# Scripts

## Directory Layout

Scripts are organized by analysis stage:

- **`behavioral/`** — Prospect theory modeling, model comparison, time-on-task behavioral effects
- **`fmri/`** — First-level GLMs, group-level analysis, trajectory extraction, connectivity
- **`figures/`** — Publication-quality figure generation

## Expected Data Directory Structure

Download the NARPS dataset from [OpenNeuro ds001734](https://openneuro.org/datasets/ds001734) and organize as follows:

```
data/
├── sub-001/
│   └── func/
│       ├── sub-001_task-MGT_run-01_events.tsv
│       ├── sub-001_task-MGT_run-02_events.tsv
│       ├── sub-001_task-MGT_run-03_events.tsv
│       └── sub-001_task-MGT_run-04_events.tsv
├── sub-002/
│   └── ...
└── derivatives/
    └── fmriprep/
        ├── sub-001/
        │   └── func/
        │       ├── sub-001_task-MGT_run-01_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
        │       ├── sub-001_task-MGT_run-01_bold_confounds.tsv
        │       └── ...
        └── sub-002/
            └── ...
```

## Running Order

Scripts are numbered and should be run in order. Behavioral scripts (01–20) can run independently of fMRI scripts (21–37). Within each group, later scripts depend on outputs from earlier ones.

All scripts assume execution from the **project root directory** (one level above `scripts/`).

## Key Behavioral CSV Column Reference

- `all_subjects_behavior.csv`: onset, duration, gain, loss, RT, participant_response, subject, run, accepted, valid
- `runwise_parameters_fixed.csv`: subject, run, lambda, alpha, beta
- `individual_parameters_final.csv`: hierarchical Bayes individual-level estimates
