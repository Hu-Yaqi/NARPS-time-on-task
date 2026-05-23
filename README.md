# Fading Values: A Shared Latent State Links Behavioral and Neural Valuation Drift Over Time

Reanalysis of the NARPS mixed gamble fMRI dataset ([OpenNeuro ds001734](https://openneuro.org/datasets/ds001734)) examining whether behavioral valuation drift and neural value-signal change during repeated risky choice reflect a shared temporal process.

## Overview

This project investigates how extended task performance alters both behavioral decision parameters and neural value signals. Using prospect theory modeling, fMRI trajectory analysis, and dynamic state-space modeling, we show that:

1. **Loss aversion (λ) increases** systematically across runs, while diminishing sensitivity (α) adds little out-of-sample
2. **vmPFC value discrimination attenuates** — gain and loss parametric sensitivity converge toward zero across eight time bins
3. **The vmPFC gap effect survives independent ROI validation** at the Tom et al. (2007) coordinate
4. **Behavioral and neural drift are temporally linked** by a shared latent-state model, despite null standard individual-level correlations
5. **Temporal dynamics are strongly supported** by model comparison against a no-drift null model
6. **Stimulus balance checks rule out simple composition confounds** across runs and time bins

## Repository Structure

    narps-time-on-task/
    ├── scripts/
    │   ├── behavioral/             # Prospect theory models, drift analyses, stimulus balance
    │   ├── fmri/                   # GLMs, ROI analyses, trajectory extraction
    │   ├── figures/                # Reserved for publication figure scripts
    │   └── state-space-model/      # Shared-state, independent-state, and model-comparison scripts
    ├── environment.yml
    ├── requirements.txt
    ├── LICENSE
    └── README.md

## Dataset

- **Source:** [NARPS dataset on OpenNeuro (ds001734)](https://openneuro.org/datasets/ds001734)
- **Task:** Mixed gamble task — accept/reject 50/50 gambles with gains (10–40 ILS) and losses (5–20 ILS)
- **Subjects:** 108 behavioral, 41 fMRI
- **Design:** 256 trials per subject (4 runs × 64 trials), TR = 1 s
- **Preprocessing:** fMRIPrep, MNI152NLin2009cAsym space

Raw data are not included in this repository. Download from OpenNeuro and place under `data/` following the expected local directory structure.

## Analysis Pipeline

### Behavioral Analysis

| Script | Analysis | Key Finding |
|--------|----------|-------------|
| 05 | Hierarchical Bayesian prospect theory (PyMC) | Population parameter estimation |
| 11b | Run-wise MLE estimation | Loss aversion increases across runs |
| 12b | Model comparison (WAIC, BIC) | Loss-aversion-only model favored |
| 14–15 | Matched fatigue analysis by EV bin | Effects selective to unfavorable/ambiguous gambles |
| 16 | Response strength analysis | Weak → strong rejection shift |
| 17 | Model comparison by half | Late-session model changes |
| 20 | Leave-one-run-out cross-validation | Loss-aversion model matches full model accuracy |
| 33 | Behavioral sawtooth statistics | Between-run partial recovery |
| 45 | Stimulus balance check | Gamble composition cannot explain temporal drift |

### fMRI Analysis

| Script | Analysis | Key Finding |
|--------|----------|-------------|
| 22–23 | First-level GLMs & group analysis | Loss > gain neural asymmetry |
| 24 | Early vs. late contrast | vmPFC loss sensitivity ↑, gain sensitivity ↓ |
| 26 | Brain–behavior ROI correlation | Group-level parallel, individual-level decoupled |
| 27 | Trial-level GLM interaction | Gradual loss × trial effects |
| 28, 30 | 8-bin gain/loss trajectories | vmPFC gain/loss sensitivity converges toward zero |
| 34 | Neural sawtooth statistics | Partial between-run recovery |
| 35 | Multi-ROI trajectories | Regional specificity tests |
| 36 | Trial-level logistic regression | ROI × time choice interactions |
| 37 | Per-run context-dependent correlation | Exploratory vmPFC–target coupling |
| 38 | Within-run multi-area trajectories | Multi-region within-run temporal dynamics |
| 40 | vmPFC–behavior temporal link | Behavioral and neural trajectory linkage |
| 42 | Gain ROI validation | Gain-sensitive ROI identification/validation |

### State-Space Modeling

| Script | Analysis | Key Finding |
|--------|----------|-------------|
| 41b | Shared latent-state model | One latent state links rejection rate and vmPFC gap |
| 41c | State-space model with RT | RT extension shows convergence limitations |
| 41d | Drifting-λ model | Shared process operates at valuation-parameter level |
| 43 | Post-hoc ROI state-space analysis | Multi-region sign-group structure |
| 44 | Independent ROI and model comparison | Tom ROI validation; shared vs. independent vs. null models |
| 44b | WAIC comparison fix | Per-outcome log-likelihood model comparison |

## Key Model-Comparison Logic

The model-comparison analyses test whether temporal dynamics are necessary and whether a single shared latent process is sufficient.

- **Shared vs. null:** tests whether temporal dynamics improve fit
- **Independent vs. null:** tests whether separate behavioral/neural dynamics improve fit
- **Shared vs. independent:** tests whether one latent state performs comparably to two separate states
- **Reset vs. no-reset:** tests whether run-boundary recovery improves the temporal model

For WAIC-based comparison, PyMC sampling should store log likelihoods:

```python
pm.sample(..., idata_kwargs={"log_likelihood": True})
```

## Notation

| Symbol | Meaning |
|--------|---------|
| λ | Loss aversion |
| α | Diminishing sensitivity / curvature |
| τ | Inverse temperature |
| λ̂*ⱼ,ᵣ* | Run-wise MLE estimate for subject *j*, run *r* |
| β^G | fMRI gain parametric coefficient |
| β^L | fMRI loss parametric coefficient |
| SV | Subjective value of a gamble |
| vmPFC gap | Gain-minus-loss value-discrimination measure |
| *z*ⱼ,ᵦ | Latent state for subject *j*, time bin *b* |
| β_B | Behavioral loading on the latent state |
| β_N | Neural loading on the latent state |
| ρ | Run-boundary reset parameter |
| elpd_WAIC | Expected log predictive density from WAIC; higher is better |

## Requirements

- Python ≥ 3.9
- Key packages: `nilearn`, `pymc`, `arviz`, `nibabel`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`
- See `environment.yml` or `requirements.txt` for full specification

### Known Compatibility Notes

- `threshold_stats_img` is in `nilearn.glm`, not `nilearn.image`
- `SecondLevelModel.compute_contrast()` has no `stat_type` parameter
- `NiftiSpheresMasker.fit_transform()` may return a 1D array; use `vals.flat[0]`
- Binned regressors across runs require all bins to be present in every run design matrix

## Reproducing the Analyses

1. Download the NARPS dataset from [OpenNeuro ds001734](https://openneuro.org/datasets/ds001734)
2. Run fMRIPrep preprocessing, or use available fMRIPrep derivatives
3. Place data following the expected local directory structure
4. Create the environment: `conda env create -f environment.yml`
5. Activate the environment: `conda activate narps`
6. Run scripts in numerical order from the project root directory

Example:

```bash
python scripts/behavioral/45_stimulus_balance_check.py
python scripts/fmri/40_vmpfc_behavior_link.py
python scripts/state-space-model/41b_state_space_model_fixed.py
```

All scripts assume execution from the project root (`narps-time-on-task/`) and use relative paths where possible.

## Data Availability

This repository contains **analysis code only**. Raw fMRI data, preprocessed derivatives, generated figures, output tables, model traces, and posterior samples are not included. The raw fMRI data are available on [OpenNeuro ds001734](https://openneuro.org/datasets/ds001734).

## Citation

If you use this code or build on these analyses, please cite:

> Hu, Y. (2026). Fading Values: A Shared Latent State Links Behavioral and Neural Valuation Drift Over Time.

And the original NARPS dataset:

> Botvinik-Nezer, R., Holzmeister, F., Camerer, C.F. et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. *Nature*, 582, 84–88. https://doi.org/10.1038/s41586-020-2314-9

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
