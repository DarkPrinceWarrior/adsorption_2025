# Project Overview

This repository implements an inverse-design framework for porous metal-organic frameworks (MOFs). The main goal is to find synthesis conditions that produce target structural/energetic properties: `E0`, `x0`, and `Sme`.

The production workflow described in `README.md` uses:
- Forward model: CatBoost with MAPIE conformal intervals, fold-local feature selection, OOF and production artifacts.
- Production inverse design: native BoFire strategy loop via `scripts/run_bofire_opt.py`.
- Research/challenger branches: TabPFN, BoTorch, BayBE, direct inverse baseline, comparative Wave 2 benchmark/report.

Primary dataset files live under `data/`, with enriched input usually at `data/SEC_SYN_with_features_enriched.csv`. Generated models, metrics, plots, and optimizer outputs live under `artifacts/`.

Core package path: `src/adsorb_synthesis/`.
Important package modules:
- `config.py`: dataclass-based CatBoost/forward model configuration and tuned per-target hyperparameters.
- `constants.py`: random seed, targets, feature lists, physical constants, solvent/stoichiometry lookup data.
- `data_processing.py`: dataset loading and feature engineering, including salt masses, thermodynamic features, temperatures, descriptors, and interaction features.
- `data_validation.py`: validation for physical synthesis constraints.
- `feature_selection.py`: domain curation, multicollinearity filtering, and feature selection helpers.
- `forward_modeling.py`: stratification, quality weights, CatBoost wrapper with selected features, TabPFN frame preparation.
- `holdout_evaluation.py`: chemistry holdout split/evaluation helpers.
- `inverse_optimization.py`: BoFire/legacy optimizer contexts, constraints, candidate scoring, shortlist generation.
- `physics_losses.py`: physical constraints and penalties.
- `molar_masses.py`: molar mass lookup and fallback behavior.

Main scripts:
- `scripts/enrich_descriptors.py`
- `scripts/tune_hyperparams.py`
- `scripts/train_forward_model.py`
- `scripts/validate_uncertainty.py`
- `scripts/evaluate_forward_holdout.py`
- `scripts/run_bofire_opt.py`
- `scripts/run_bofire_optuna_legacy.py`
- `scripts/run_botorch_mobo.py`
- `scripts/run_baybe_campaign.py`
- `scripts/train_inverse_direct.py`
- `scripts/benchmark_wave2.py`
- `scripts/generate_wave2_report.py`
- `scripts/run_wave2_suite.py`
- `scripts/run_bayes_opt.py`
- `scripts/generate_paper_figures.py`

Dependencies are managed with `requirements.txt`; no `pyproject.toml`, Makefile, tox config, Ruff/Black/isort/mypy config was found during onboarding.