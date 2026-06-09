# Suggested Commands

System: Linux (WSL). Package under `src/` -> prefix scripts/tests with `PYTHONPATH=src`.

Setup:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Workflow (ordered):
```bash
# Step 0: enrich descriptors
PYTHONPATH=src python scripts/enrich_descriptors.py \
    --input data/SEC_SYN_with_features.csv \
    --output data/SEC_SYN_with_features_enriched.csv

# Step 1 (optional): Optuna HP tuning (fold-local feature selection)
PYTHONPATH=src python scripts/tune_hyperparams.py \
    --data data/SEC_SYN_with_features_enriched.csv --trials 80

# Step 2: train production forward model (CatBoost + MAPIE)
PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --backend catboost --validation-mode warn

# Step 3: internal UQ diagnostic (rejection plots + coverage)
PYTHONPATH=src python scripts/validate_uncertainty.py

# Step 3b: external chemistry holdout (Metal|Ligand split)
PYTHONPATH=src python scripts/evaluate_forward_holdout.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --output-dir artifacts/forward_holdout --backend all

# Step 4: production inverse design (native BoFire)
PYTHONPATH=src python scripts/run_bofire_opt.py \
    --E0 15.0 --x0 0.5 --Sme 100.0 \
    --trials 300 --shortlist-size 12 \
    --output artifacts/predictions_bofire.csv

# Step 5 (optional): wave 2 comparative suite
PYTHONPATH=src python scripts/run_wave2_suite.py --mode full \
    --data data/SEC_SYN_with_features_enriched.csv \
    --catboost-models artifacts/forward_models \
    --E0 15 --x0 0.5 --Sme 100 \
    --run-dir artifacts/wave2_runs/selection_run
```

Navigation: use `fff` for file search / repo grep; `codegraph` for structural
questions; Serena symbolic tools for symbol reads/edits. Avoid shell
`find`/`grep`/`rg` while `fff` is available.
