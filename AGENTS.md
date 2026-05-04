# AGENTS.md instructions for /home/ruslan_safaev/adsorb_synth/adsorb_synthesis

## Repository Tools

For any file search or grep in this git repository, use `fff` first.

Do not use shell `find`, `grep`, or `rg` when `fff` is available, unless `fff` cannot express the query.

Use `fff` for:

- finding files by name or pattern
- searching code across the repository
- discovering entry points, scripts, services, modules, and tests
- narrowing down where to inspect before using Serena

Use Serena after `fff` has identified the relevant area.

Prefer Serena for:

- `get_symbols_overview`
- `find_symbol`
- `find_referencing_symbols`
- `replace_symbol_body`
- `insert_before_symbol`
- `insert_after_symbol`
- `rename_symbol`
- `safe_delete_symbol`

Prefer Serena tools over reading full source files when symbolic tools are sufficient.

Serena runs with project-from-cwd. Do not ask to manually activate the repository when Serena is already available.

## Skills And Docs

Use a skill when the task matches its description.

When working on FastAPI routes, dependencies, request/response schemas, Pydantic models, startup/lifespan logic, or API design, use the `fastapi` skill.

Use Context7 before relying on memory for framework or library behavior that may be version-sensitive or uncertain.

Use Tavily MCP tools for web research and fresh external information.

## Project Commands

Set up the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Common workflow entry points:

```bash
PYTHONPATH=src python scripts/enrich_descriptors.py \
    --input data/SEC_SYN_with_features.csv \
    --output data/SEC_SYN_with_features_enriched.csv

PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --backend catboost \
    --validation-mode warn

PYTHONPATH=src python scripts/validate_uncertainty.py

PYTHONPATH=src python scripts/evaluate_forward_holdout.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --output-dir artifacts/forward_holdout \
    --backend all

PYTHONPATH=src python scripts/run_bofire_opt.py \
    --E0 15.0 \
    --x0 0.5 \
    --Sme 100.0 \
    --trials 300 \
    --shortlist-size 12 \
    --output artifacts/predictions_bofire.csv
```

## Working Style

Before changing code, briefly state:

- what was found
- what will change
- why the change is needed

Prefer minimal, localized edits over broad refactors.

Do not rename files, move modules, change public interfaces, install dependencies, or edit secrets unless explicitly required.

Separate confirmed facts from inference.

When finishing a code task:

- run the focused relevant tests when possible
- run the full test suite for broad/shared changes
- check `git status --short`
- report only task-related changes

Do not revert unrelated dirty work.

## Project Notes

This repository implements an inverse-design framework for porous metal-organic frameworks (MOFs).

Core package: `src/adsorb_synthesis/`

Important areas:

- `src/adsorb_synthesis/config.py`: model configuration
- `src/adsorb_synthesis/constants.py`: targets, features, constants, lookup data
- `src/adsorb_synthesis/data_processing.py`: feature engineering and dataset preparation
- `src/adsorb_synthesis/data_validation.py`: synthesis constraint validation
- `src/adsorb_synthesis/feature_selection.py`: domain and statistical feature selection
- `src/adsorb_synthesis/forward_modeling.py`: forward model helpers
- `src/adsorb_synthesis/inverse_optimization.py`: inverse-design optimizer logic
- `scripts/`: runnable workflow and benchmark scripts
- `tests/`: pytest suite

The production path described in `README.md` is CatBoost + MAPIE for forward modeling and native BoFire for inverse optimization.
