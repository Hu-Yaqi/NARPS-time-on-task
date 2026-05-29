# Fading Values: A Shared Latent State Links Behavioral and Neural Valuation Drift Over Time

This repository contains analysis code for a reanalysis of the NARPS mixed-gamble fMRI dataset. The project examines whether behavioral valuation drift and neural value-signal changes during repeated risky choice can be linked by a shared latent temporal process.

## Overview

Using the NARPS mixed-gamble task, this project combines:

- prospect-theory modeling of behavioral choice,
- fMRI gain/loss trajectory analysis,
- independent vmPFC ROI validation,
- stimulus-balance checks,
- dynamic state-space modeling.

Main findings:

- Loss aversion increases across the session.
- vmPFC gain and loss sensitivity converge toward zero over time.
- The vmPFC gain-loss gap convergence is also observed in an independent Tom et al. (2007) vmPFC ROI.
- Dynamic state-space models outperform a no-drift null model.
- A shared single-state model performs comparably to an independent two-state model, supporting the shared-state account on parsimony grounds.

## Repository Structure

```text
scripts/
├── behavioral/             # Behavioral modeling and stimulus-balance checks
├── fmri/                   # fMRI GLM, ROI, and trajectory analyses
├── figures/                # Figure-generation scripts only
└── state-space-model/      # State-space models and model comparison
```

Generated figures, model traces, output tables, and raw data are not stored in this repository.

## Dataset

Dataset: NARPS mixed-gamble fMRI dataset, OpenNeuro `ds001734`.

Task structure:

- 108 behavioral participants
- 41 fMRI participants analyzed here
- 4 runs × 64 trials = 256 trials per participant
- 50/50 mixed gambles with gains and losses
- Accept/reject decisions on each trial

Raw data should be downloaded from OpenNeuro and placed in the expected local data directory before running the scripts.

## Key Scripts

### Behavioral

| Script | Purpose |
|---|---|
| `45_stimulus_balance_check.py` | Checks gain, loss, EV, and gain/loss-ratio balance across runs and time bins |

### fMRI

| Script | Purpose |
|---|---|
| `38_within_runs_mutilarea.py` | Examines within-run neural trajectories across multiple ROIs |
| `40_vmpfc_behavior_link.py` | Tests links between vmPFC trajectory measures and behavioral drift |
| `42_find_gain_ROI.py` | Identifies or validates gain-sensitive ROIs |

### State-space modeling

| Script | Purpose |
|---|---|
| `41b_state_space_model_fixed.py` | Final shared latent-state model |
| `41c_state_space_with_rt.py` | State-space model including RT as an additional outcome |
| `41d_state_space_drifting_lambda.py` | Trial-level drifting-loss-aversion model |
| `43_state_space_posthoc.py` | Post-hoc state-space analyses across additional ROIs |
| `44_independent_roi_and_model_comparison.py` | Independent vmPFC ROI validation and model-comparison framework |
| `44b_model_comparison_fix.py` | WAIC model-comparison fix using per-outcome log likelihoods |

## Reproducing the Analyses

Clone the repository:

```bash
git clone https://github.com/Hu-Yaqi/NARPS-time-on-task.git
cd NARPS-time-on-task
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate narps
```

Run scripts from the project root. Some scripts depend on intermediate outputs from earlier analyses, so the full pipeline should be run in numerical order.

Example:

```bash
python scripts/behavioral/45_stimulus_balance_check.py
python scripts/fmri/40_vmpfc_behavior_link.py
python scripts/state-space-model/41b_state_space_model_fixed.py
```

For WAIC-based state-space model comparison, PyMC sampling should store log likelihoods:

```python
pm.sample(..., idata_kwargs={"log_likelihood": True})
```

## Output Policy

This repository intentionally excludes:

```text
raw fMRI data
preprocessed derivatives
generated figures
CSV result tables
model traces
posterior samples
large output files
```

Only source code and documentation are tracked.

## Citation

If you use this code or build on these analyses, please cite:

> Hu, Y. *Fading Values: A Shared Latent State Links Behavioral and Neural Valuation Drift Over Time.*

Please also cite the original NARPS dataset and the NARPS many-analysts paper:

> Botvinik-Nezer, R., Holzmeister, F., Camerer, C. F., et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. *Nature*, 582, 84–88.

## License

This project is licensed under the MIT License.
