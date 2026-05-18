# Losing More, Gaining Less: Behavioral and Neural Signatures of Shifting Valuation During Risky Choice

Reanalysis of the NARPS mixed gamble fMRI dataset ([OpenNeuro ds001734](https://openneuro.org/datasets/ds001734)) examining time-on-task effects on value computation during risky decision-making.

**Target venue:** CCN (Cognitive Computational Neuroscience) 2026, NYU, August 3–6

## Overview

This project investigates how extended task performance alters both behavioral decision parameters and neural value signals. Using prospect theory modeling and fMRI analysis of 41 subjects performing 256 mixed gamble trials, we show that:

1. **Loss aversion (λ) increases** systematically across runs while diminishing sensitivity (α) adds nothing out-of-sample
2. **vmPFC value discrimination collapses** — both gain and loss parametric sensitivity converge toward zero over time
3. **Upstream regions remain stable** — insula, dACC, and IFG maintain consistent loss coding while vmPFC degrades
4. **Behavioral consequences are selective** — unfavorable (ambiguous) gambles are most affected; favorable gambles are not

## Repository Structure

```
narps-time-on-task/
├── scripts/
│   ├── behavioral/          # Prospect theory models, model comparison, fatigue analysis
│   │   ├── 01_data_exploration.py
│   │   ├── ...
│   │   └── 20_cross_validation.py
│   ├── fmri/                # GLMs, group analysis, trajectory extraction
│   │   ├── 21_single_subject_glm.py
│   │   ├── ...
│   │   └── 37_per_run_connectivity.py
│   └── figures/             # Publication figure generation
│       ├── 29_behavioral_fatigue_figure.py
│       ├── 31_publication_figures.py
│       └── 32_pipeline_figure.py
├── environment.yml          # Conda environment specification
├── requirements.txt         # pip dependencies
├── LICENSE
└── README.md
```

## Dataset

- **Source:** [NARPS dataset on OpenNeuro (ds001734)](https://openneuro.org/datasets/ds001734)
- **Task:** Mixed gamble task — accept/reject 50/50 gambles with gains (10–40 ILS) and losses (5–20 ILS)
- **Subjects:** 108 behavioral, 41 fMRI
- **Design:** 256 trials per subject (4 runs × 64 trials), TR = 1 s
- **Preprocessing:** fMRIPrep, MNI152NLin2009cAsym space

Raw data are not included in this repository. Download from OpenNeuro and place under `data/` following the directory structure in `scripts/README.md`.

## Analysis Pipeline

### Behavioral Analysis

| Script(s) | Analysis | Key Finding |
|-----------|----------|-------------|
| 01–05 | Data exploration & cleaning | 108 usable subjects |
| 09–13 | Hierarchical Bayesian prospect theory (PyMC) | 4-model comparison via WAIC, BIC, CV |
| 14–18 | Run-wise MLE & time-on-task effects | λ: 1.168 → 1.299, *p* < .0001, *d* = 0.39 |
| 19–20 | Response strength & cross-validation | Selective to unfavorable gambles (*d* = 0.67) |

### fMRI Analysis

| Script(s) | Analysis | Key Finding |
|-----------|----------|-------------|
| 22–23 | First-level GLMs & group analysis | Loss > Gain (R IFG: 23,318 mm³) |
| 24 | Early vs. late contrast | vmPFC loss sensitivity ↑, gain sensitivity ↓ |
| 27 | Trial-level GLM (loss × trial interaction) | Gradual change in dmPFC/dACC |
| 26 | Brain–behavior ROI correlation | Group-level parallel, individual-level decoupled |
| 28, 30 | 8-bin gain/loss trajectories | vmPFC: both channels → zero (*p* < .01) |
| 34 | Sawtooth analysis | Loss between-run partial recovery (*p* = .029) |
| 35 | Multi-ROI trajectories | Insula, dACC, IFG stable (all *p* > .16) |
| 36 | Trial-level logistic regression | vmPFC vs. insula/dACC predictive shift |
| 37 | Per-run context-dependent correlation | vmPFC–target connectivity over time |

## Notation

| Symbol | Meaning |
|--------|---------|
| λ | Loss aversion (population hierarchical Bayes) |
| α | Diminishing sensitivity / curvature |
| τ | Inverse temperature (not β, to avoid GLM confusion) |
| λ̂*ⱼ,ᵣ* | Run-wise MLE estimate (subject *j*, run *r*) |
| β^G*ₖ* | fMRI GLM regression coefficient for regressor *k* |
| *z*_stat | Voxel-level test statistic (β^G*ₖ* / SE) |
| SV | Subjective value of a gamble |

## Requirements

- Python ≥ 3.9
- Key packages: `nilearn` (≥ 0.13.1), `pymc` (≥ 5.0), `nibabel`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`
- See `environment.yml` or `requirements.txt` for full specification

### Known Compatibility Notes (nilearn 0.13.1)

- `threshold_stats_img` is in `nilearn.glm`, not `nilearn.image`
- `SecondLevelModel.compute_contrast()` has no `stat_type` parameter
- `NiftiSpheresMasker.fit_transform()` returns a 1D array; use `vals.flat[0]`
- Binned regressors across runs require all bins present in every run's design matrix (use `modulation=0` for inactive bins)

## Reproducing the Analyses

1. Download the NARPS dataset from [OpenNeuro ds001734](https://openneuro.org/datasets/ds001734)
2. Run fMRIPrep preprocessing (or use the preprocessed derivatives if available)
3. Place data following the expected directory structure (see `scripts/README.md`)
4. Create the environment: `conda env create -f environment.yml`
5. Run scripts in numerical order from the project root directory

All scripts assume execution from the project root (`narps-time-on-task/`) and use relative paths.

## Data Availability

This repository currently contains **analysis code only**. Result files, figures, and the paper will be added upon publication. The raw fMRI data are available on [OpenNeuro ds001734](https://openneuro.org/datasets/ds001734).

## Citation

If you use this code or build on these analyses, please cite:

> [Author]. (2026). Losing More, Gaining Less: Behavioral and Neural Signatures of Shifting Valuation During Risky Choice. *Proceedings of the Cognitive Computational Neuroscience Conference*.

And the original NARPS dataset:

> Botvinik-Nezer, R., Holzmeister, F., Camerer, C.F. et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. *Nature*, 582, 84–88. https://doi.org/10.1038/s41586-020-2314-9

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
