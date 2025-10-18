# Adsorb Synthesis Inverse-Design Pipeline

End-to-end system for inverse design of MOF adsorbents. The project ingests experimental synthesis data, engineers physico-chemical descriptors, trains a staged prediction pipeline, enforces physics constraints at inference time, and exposes reproducible CLI tooling for retraining, inference, and hyperparameter tuning.

---

## Contents

1. [Getting Started](#getting-started)
2. [Repository Layout](#repository-layout)
3. [Data Model](#data-model)
   * [Raw inputs](#raw-inputs)
   * [Engineered features](#engineered-features)
   * [Physics-inferred fields](#physics-inferred-fields)
4. [Model Pipeline](#model-pipeline)
   * [Stage overview](#stage-overview)
   * [Algorithms](#algorithms)
   * [Physics enforcement](#physics-enforcement)
5. [Command Line Workflows](#command-line-workflows)
   * [Training](#training)
   * [Inference](#inference)
   * [Hyperparameter tuning](#hyperparameter-tuning)
6. [Testing](#testing)
7. [Extending the System](#extending-the-system)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

```bash
# 1. Install dependencies (Python ≥ 3.10)
python -m pip install -r requirements.txt

# 2. Train the full pipeline on supplied data
python scripts/train_inverse_design.py \
    --data data/SEC_SYN_with_features_DMFA_only_no_Y.csv \
    --output artifacts

# 3. Generate predictions for new adsorption descriptors
python scripts/predict_inverse_design.py \
    --model artifacts \
    --input tmp_inference_input.csv \
    --output tmp_predictions.csv \
    --targets-only
```

Prerequisites: CUDA is optional; the ensemble falls back to CPU. Optuna-based tuning additionally requires `optuna` (already listed in `requirements.txt`).

---

## Repository Layout

```
adsorb_synthesis/
├── data/                         # Raw and engineered datasets
├── docs/                         # Supplementary analysis (if any)
├── scripts/                      # CLI entry points
│   ├── train_inverse_design.py
│   ├── predict_inverse_design.py
│   └── tune_inverse_design.py
├── src/adsorb_synthesis/
│   ├── __init__.py
│   ├── constants.py              # Feature lists, physics bounds, seeds
│   ├── data_processing.py        # Loading, feature engineering, lookups
│   ├── modern_models.py          # ModernTabular ensemble implementations
│   ├── physics_losses.py         # Physics loss/regularisers & projectors
│   └── pipeline.py               # Stage configs, training & inference
├── tests/                        # Pytest-based sanity checks
├── requirements.txt
└── README.md
```

Artifacts produced by `train_inverse_design.py` live under the configured output directory (`artifacts/` by default) and include stage pipelines, lookup tables, and metadata for reload.

---

## Data Model

### Raw inputs

* `data/SEC_SYN_new.csv` – raw synthesis experiments with process metadata.
* `data/SEC_SYN_with_features_DMFA_only_no_Y.csv` – curated training/inference dataset (DMFA solvent only, Y-metal excluded) with engineered descriptors.

### Engineered features

`src/adsorb_synthesis/data_processing.py` is responsible for

1. `load_dataset`: reading CSV, cloning the frame, and orchestrating feature builders.
2. `_ensure_adsorption_features`: deriving adsorption descriptors if absent (surface/pore metrics, energy ratios, etc.).
3. `add_temperature_categories`: binning numeric temperatures into three process-specific categories (`Tsyn`, `Tdry`, `Treg`).
4. `add_salt_mass_features`: log transforms, categorical composites (`Metal_Ligand_Combo`), and metal flags.
5. `build_lookup_tables`: generating descriptor lookups for metal, ligand, solvent to fill missing categorical information at inference time.

Feature group constants are centralised in `src/adsorb_synthesis/constants.py`:

| Group                        | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| `ADSORPTION_FEATURES`        | Raw adsorption measurements and engineered ratios (`W0`, `E0`, `S_BET_E`, etc.) |
| `METAL_DESCRIPTOR_FEATURES`  | Descriptor columns supplied via lookup (molar weights, ionic radius, etc.)     |
| `LIGAND_DESCRIPTOR_FEATURES` | Structural descriptors and molar quantities for ligands                        |
| `SOLVENT_DESCRIPTOR_FEATURES`| Basic solvent phys-chem descriptors                                            |
| `PROCESS_CONTEXT_FEATURES`   | Currently `Mix_solv_ratio`, can be extended with process context                |

### Physics-inferred fields

During loading and inference the pipeline derives thermodynamic helpers:

* `Delta_G_equilibrium` – Gibbs free energy inferred from measured `K_equilibrium` and actual synthesis temperature.
* `K_equilibrium_from_delta_G` – Theoretical `K_eq` computed from dataset-provided `Delta_G`.
* `Delta_G_residual` / `K_equilibrium_ratio` – Diagnostics showing deviation between supplied and equilibrium-consistent values.
* `n_соли`, `n_кислоты`, `n_ratio`, `n_ratio_residual` – Stoichiometric molar quantities and deviation from the admissible range `[0.45, 2.3]`.

These fields enable thermodynamic and stoichiometric enforcement during inference.

---

## Model Pipeline

### Stage overview

`default_stage_configs()` defines the sequential inverse-design workflow:

1. **`metal`** *(classification)* – Predicts the metal species from adsorption descriptors.
2. **`ligand`** *(classification)* – Predicts the ligand conditioned on adsorption profile and predicted metal.
3. **`salt_mass`** *(regression)* – Estimates log-transformed salt mass (Huber/IsolationForest for outliers).
4. **`acid_mass`** *(regression)* – Predicts acid mass given salt and stoichiometry.
5. **`solvent_volume`** *(regression)* – Recommends solvent volume.
6. **`tsyn_category`**, **`tdry_category`**, **`treg_category`** *(classification)* – Classify temperature regimes for synthesis, drying, regeneration.

Each stage records dependencies via `StageConfig.depends_on`, ensuring features predicted upstream are accessible downstream.

### Algorithms

The ensemble models (`modern_models.py`) combine TabNet, CatBoost, and XGBoost:

* **TabNet** – 3 steps, `n_d=n_a=32`, AdamW (`lr=8e-3`), CosineAnnealingLR, sparsemax masks for classification.
* **CatBoost** – depth 8, 1600 iterations, MAE loss for regressions, explicit class weights (no auto balancing).
* **XGBoost** – histogram tree method, depth 5, 1600 estimators, pseudo-Huber loss for regressions.
* Ensemble weights are optimised via SciPy SLSQP to minimise Brier score / MAE with L2 regularisation.

Sampling and class imbalance handling:

* Optional ADASYN/SMOTE for classification stages (skipped if sample weights are supplied by physics regularisation).
* Focal-style class weighting to emphasise minority classes.
* `IsolationForest` trimming for regression outliers when configured.

### Physics enforcement

Physics integration is woven throughout the pipeline:

| Enforcement Layer      | Mechanism                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------|
| **Loss regularisation**| `combined_physics_loss` penalises deviations from Gibbs equilibrium and adsorption energy bounds.   |
| **Sample weighting**   | Physics penalty per-sample increases training weight for physically inconsistent observations.      |
| **Thermodynamic projection** | `project_thermodynamics` overwrites `K_equilibrium` and `Delta_G_equilibrium` to satisfy `K = exp(-ΔG/RT)`. |
| **Stoichiometry clamp**| `_project_stoichiometry` rescales acid mass to keep `n_ratio` in `[0.45, 2.3]`.                      |
| **Temperature order**  | `_enforce_temperature_order` ensures `Tdry ≥ Tsyn` and `Treg ≥ Tdry`.                               |

Physics loss defaults (from Optuna tuning):

```
physics_weight = 0.017
w_thermo        = 0.175
w_energy        = 0.061
```

These can be retuned with `tune_inverse_design.py` (see below).

---

## Command Line Workflows

### Training

```
python scripts/train_inverse_design.py \
    --data data/SEC_SYN_with_features_DMFA_only_no_Y.csv \
    --output artifacts
```

The command:

1. Loads the dataset and derives missing descriptors.
2. Builds lookup tables for categorical augmentation.
3. Trains every stage (with cross-validation reporting).
4. Saves pipelines, lookup tables, and metadata into the output directory.

Artifacts include `<stage>_pipeline.joblib`, lookup `joblib`s, and `pipeline_metadata.joblib` for reproducible reloads via `InverseDesignPipeline.load()`.

### Inference

```
python scripts/predict_inverse_design.py \
    --model artifacts \
    --input tmp_inference_input.csv \
    --output tmp_predictions.csv \
    --targets-only
```

Options:

* `--targets-only` – return only stage outputs (metal, ligand, reagent masses, temps).
* Without `--targets-only`, the tool returns a superset including engineered features, physics diagnostics (`Delta_G_residual`, `n_ratio_residual`), and input columns.
* By default the CLI enforces physics corrections (`thermo → stoichiometry → temperature order`). Use `--disable-physics` (flag exposed in the script) to inspect raw model predictions.

### Hyperparameter tuning

`scripts/tune_inverse_design.py` wraps Optuna multi-objective optimisation (maximise metal balanced accuracy, minimise physics penalty).

```
python scripts/tune_inverse_design.py \
    --data data/SEC_SYN_with_features_DMFA_only_no_Y.csv \
    --trials 20 \
    --subset 200 \
    --n-jobs 1
```

Results include the Pareto front with associated `physics_weight`, `w_thermo`, `w_energy`, and diagnostic MAE values (`ΔG_residual`, `n_ratio_residual`). Chosen parameters can be applied by editing `default_stage_configs()`.

---

## Testing

Lightweight sanity checks live under `tests/`. To run them (requires `pytest`):

```bash
python -m pip install pytest
python -m pytest tests
```

`tests/test_physics_constraints.py` currently verifies:

* Thermodynamic projection aligns `K_equilibrium` with `Delta_G_equilibrium`.
* Stoichiometry post-processing clamps `n_ratio` and zeroes residuals.
* Temperature category ordering never violates process monotonicity.

Add further tests as the pipeline evolves (e.g. CLI smoke tests, stage-specific metric assertions).

---

## Extending the System

1. **New stages** – Add to `default_stage_configs()`, supply feature lists, dependencies, and update CLI output formatting if necessary.
2. **Alternate models** – Modify `_default_classifier` / `_default_regressor` to inject custom estimators or adjust ensemble parameters.
3. **Additional physics** – Implement new loss components in `physics_losses.py` and include them via `PhysicsConstraintEvaluator`.
4. **Feature engineering** – Extend `data_processing.py` with new descriptor builders or ingestion routines.
5. **Packaging** – Adopt `pyproject.toml` or `poetry` for reproducible environments when distributing beyond a notebooks/CLI workflow.

---

## Troubleshooting

| Symptom | Possible cause & resolution |
|---------|-----------------------------|
| `ValueError: Missing required features …` | Input CSV lacks expected columns; run `load_dataset` on the file to auto-generate derived columns or update the dataset schema. |
| CatBoost training skipped with class-weight warnings | Ensure `catboost` ≥ 1.2 (installed via requirements) – the project disables `auto_class_weights` when manual weights are provided. |
| Physics penalty remains high | Inspect `Delta_G_residual` and `n_ratio_residual` in predictions; retune loss weights via Optuna or investigate data inconsistencies. |
| Optuna run fails with CUDA messages | TabNet detects CUDA; set `CUDA_VISIBLE_DEVICES=` to disable GPU if the environment lacks a compatible GPU. |
| Tests fail complaining about missing `pytest` | Install testing dependencies (`python -m pip install pytest`) before running `python -m pytest`. |

For deeper debugging, inspect individual stage pickles (`joblib.load`) or enable verbose logging inside `ModernTabularEnsembleClassifier/Regressor`.

---

## Changelog Snapshot

* Thermodynamic projections ensure `K_equilibrium = exp(-ΔG/RT)` at inference time.
* Stoichiometric post-processing clamps `n_ratio` and readjusts acid mass accordingly.
* Temperature classification outputs are monotonically ordered (`Tsyn ≤ Tdry ≤ Treg`).
* Physics loss weights initialised from Optuna search (`physics_weight=0.017`, `w_thermo=0.175`, `w_energy=0.061`).
* Optuna tuning CLI prints ΔG/n_ratio residual MAE to evaluate physical fidelity alongside accuracy.

---

© 2025 — Adsorb Synthesis Inverse-Design. All rights reserved.
