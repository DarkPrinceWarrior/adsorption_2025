# Core

Inverse-design framework for porous metal-organic frameworks (MOFs): find synthesis
conditions that yield target structural-energetic properties.
Targets: `E0` (kJ/mol, characteristic adsorption energy), `x0` (nm, characteristic
pore half-width), `Sme` (m²/g, mesopore specific surface area).

Pipeline: CSV synthesis dataset -> descriptor enrichment -> forward model
(CatBoost + MAPIE) -> UQ + external chemistry holdout -> target-oriented inverse
design (native BoFire).

## Source map
- `src/adsorb_synthesis/` — importable package; every script imports from it:
  - `constants.py` — targets, feature lists, physics bounds, `RANDOM_SEED`; single source of truth for config.
  - `config.py` — per-target tuned CatBoost hyperparameters (frozen dataclasses).
  - `data_processing.py` — descriptor generation + feature engineering; DataFrame mutators take an `inplace` flag.
  - `data_validation.py` — `validate_synthesis_data`, `warn`/`strict` (temperature order, boiling point, stoichiometry).
  - `feature_selection.py` — domain + statistical selection (|r|>0.85, VIF>10, permutation importance); MUST stay fold-local (no leakage).
  - `forward_modeling.py` — CatBoost ensemble + MAPIE cross-conformal intervals; TabPFN frame prep.
  - `holdout_evaluation.py` — deterministic `Metal|Ligand` chemistry split + metrics.
  - `inverse_optimization.py` — BoFire/legacy optimizer contexts, constraints, candidate scoring, shortlist generation.
  - `physics_losses.py` — physics-informed constraints and penalties.
  - `molar_masses.py` — reagent molar-mass lookup + fallback.
- `scripts/` — runnable workflow/benchmark/ingestion/report entrypoints (each `main()` + argparse).
- `tests/` — pytest (unit + `test_wave3_regressions.py`).
- `data/` — CSV/XLS datasets; enriched input usually `data/SEC_SYN_with_features_enriched.csv`.
- `artifacts/` — generated models/metrics/plots/results; gitignored, never commit.

## Invariants
- Package is under `src/`; run scripts AND tests with `PYTHONPATH=src` (note: `tests/conftest.py` also adds project root to `sys.path`).
- Production stack: forward = `catboost` + MAPIE; inverse = `run_bofire_opt.py` (native BoFire). TabPFN / BoTorch / BayBE / direct-inverse are research challengers only.
- Feature selection runs inside each outer CV fold — preserve the no-leakage boundary.
- RDKit ligand descriptors intentionally excluded from the production model (~4 unique ligands -> degenerate lookup).
- `validate_synthesis_data` defaults to `warn`; the dataset has constraint-violating rows.
- Local-only project: NOT synced to / run on the a100 server (unlike `alma_servie`); runs on the laptop.

References: `mem:tech_stack` for languages/deps/version pins; `mem:suggested_commands`
for the exact workflow/test commands; `mem:conventions` for code style and design
patterns; `mem:task_completion` for the done-checklist.
