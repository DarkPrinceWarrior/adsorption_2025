# Repository Guidelines

## Project Structure & Module Organization
- Core logic lives in `src/adsorb_synthesis`: `data_processing.py` prepares datasets and lookup tables, `pipeline.py` defines the staged inverse-design workflow, `modern_models.py` wraps ensemble estimators, and `constants.py` centralises feature lists and seeds.
- CLI entry points sit in `scripts/`: use `train_inverse_design.py` to fit models and `predict_inverse_design.py` for inference.
- Raw and engineered datasets stay in `data/`; trained pipelines and lookups are emitted to `artifacts/`. Keep temporary inputs (e.g. `tmp_inference_input.csv`) out of version control unless intentionally shared.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` — install the Python 3.10 toolchain (scikit-learn, ModernTabular ensemble stack, Optuna).
- `python scripts/train_inverse_design.py --data data/SEC_SYN_with_features.csv --output artifacts` — retrain the full multi-stage pipeline and persist artefacts.
- `python scripts/predict_inverse_design.py --model artifacts --input <csv> --output predictions.csv --targets-only` — run inference on adsorption descriptors; drop intermediate columns when reviewing only targets.
- `python -m pip install -e .` is not configured; keep edits within the source tree or add a `pyproject.toml` if packaging becomes necessary.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents, trailing commas in multiline structures, and type hints as in `pipeline.py`.
- Use `snake_case` for functions/variables, `CapWords` for classes, and `UPPER_CASE` for constants mirroring `constants.py`.
- Prefer f-strings and explicit return types; document non-obvious helpers with short docstrings.
- When extending stages or features, reuse shared constants to avoid desynchronised column names.

## Testing Guidelines
- Add `pytest` suites under a new `tests/` directory; target data cleaning (e.g. `load_dataset`) and stage dependencies (`default_stage_configs`).
- Craft fixtures from slimmed CSV slices in `data/` or synthetic frames to keep tests fast and deterministic.
- Run `pytest` plus a smoke invocation of the training script before pushing; capture metrics deltas if model behaviour changes.

## Commit & Pull Request Guidelines
- Recent history (`Normalize line endings to LF`, `Update README…`) shows short, imperative commits; keep the format `<scope>: <action>` when possible and mention affected stages or scripts.
- Include PR summaries covering: motivation, datasets touched, expected metric shifts, and how to reproduce (`train_inverse_design.py` parameters, test commands).
- Attach diffs for regenerated artefacts, and note any large files excluded via `.gitignore`; provide before/after metrics when retraining models.
