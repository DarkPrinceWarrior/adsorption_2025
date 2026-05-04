# Repository Structure

Top-level areas:
- `README.md`: project description, recommended production/research stack, workflow commands, structure, testing.
- `requirements.txt`: Python dependencies.
- `src/adsorb_synthesis/`: importable package.
- `scripts/`: runnable workflow and benchmark scripts.
- `tests/`: pytest unit and regression tests.
- `data/`: CSV datasets, including `SEC_SYN_with_features.csv` and enriched variants.
- `artifacts/`: generated models, metrics, predictions, plots, and benchmark outputs.
- `analysis_results/`: correlation matrices/plots and related analysis outputs.
- `papers/`: reference PDFs.
- `wave2_decision.md`: documented production/research stack decision.

Important tests found:
- `tests/test_wave3_regressions.py`
- `tests/test_forward_modeling.py`
- `tests/test_feature_engineering.py`
- `tests/test_data_validation.py`
- `tests/test_molar_masses.py`
- `tests/conftest.py`

`tests/conftest.py` adds the project root to `sys.path`; README still recommends running commands with `PYTHONPATH=src`.