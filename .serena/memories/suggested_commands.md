# Suggested Commands

Environment setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Common workflow commands from `README.md`:

Enrich descriptors:
```bash
PYTHONPATH=src python scripts/enrich_descriptors.py \
    --input data/SEC_SYN_with_features.csv \
    --output data/SEC_SYN_with_features_enriched.csv
```

Tune CatBoost hyperparameters:
```bash
PYTHONPATH=src python scripts/tune_hyperparams.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --trials 80
```

Train production forward model:
```bash
PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --backend catboost \
    --validation-mode warn
```

Validate uncertainty intervals:
```bash
PYTHONPATH=src python scripts/validate_uncertainty.py
```

Run external chemistry holdout:
```bash
PYTHONPATH=src python scripts/evaluate_forward_holdout.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --output-dir artifacts/forward_holdout \
    --backend all
```

Run production inverse design with BoFire:
```bash
PYTHONPATH=src python scripts/run_bofire_opt.py \
    --E0 15.0 \
    --x0 0.5 \
    --Sme 100.0 \
    --trials 300 \
    --shortlist-size 12 \
    --output artifacts/predictions_bofire.csv
```

Run Wave 2 comparative workflow:
```bash
PYTHONPATH=src python scripts/run_wave2_suite.py \
    --mode full \
    --tabpfn-only \
    --catboost-models artifacts/forward_models \
    --bofire-shortlist artifacts/predictions_bofire_shortlist.csv \
    --bofire-pool artifacts/predictions_bofire_all.csv \
    --botorch-shortlist artifacts/predictions_botorch.csv \
    --botorch-pool artifacts/predictions_botorch_all.csv \
    --data data/SEC_SYN_with_features_enriched.csv \
    --E0 15 \
    --x0 0.5 \
    --Sme 100 \
    --run-dir artifacts/wave2_runs/selection_run
```

Repository utility commands:
```bash
git status --short
git diff -- <path>
```

Project-specific search/navigation rules:
- Use `fff` first for file search and repo grep.
- Use Serena symbolic tools after `fff` identifies the relevant area.
- Avoid shell `find`, `grep`, or `rg` for repo search while `fff` is available.