# Tech Stack

- Language: Python 3.10+ (uses `X | Y` unions, builtin `tuple[...]`/`dict[...]`). Single-language project.
- Package manager: pip + `requirements.txt` in a venv. No `pyproject.toml`, Poetry, or uv lock.
- No build system; no linter/formatter/type-checker config (no ruff/black/isort/mypy/tox/Makefile).

Key dependencies (minimum pins in `requirements.txt`):
- Forward / ML: `catboost`, `scikit-learn>=1.4`, `xgboost`, `tabpfn>=8.0.0` (TabPFN-3 challenger).
- Uncertainty: `mapie>=1.3` (cross-conformal intervals).
- Inverse / optimization: `bofire>=0.3.1` (production inverse), `botorch`, `baybe`, `pymoo`, `optuna`, `cvxpy>=1.6.5,<1.8`.
- Chemistry descriptors: `rdkit`, `mordred`.
- Data / scientific: `pandas>=2`, `numpy`, `scipy`, `statsmodels`, `imbalanced-learn`, `joblib`.
- Explain / plots: `shap<0.51`, `seaborn`.
- Tests: `pytest`.

I/O formats: datasets and predictions are CSV; metrics/config/tuning outputs are JSON;
fitted models and conformal calibrators are joblib.

TabPFN-3 access (non-obvious): the forward TabPFN backend pins `ModelVersion.V3` via
`create_default_for_version`. Downloading v3 weights requires a one-time Prior Labs license
acceptance (`tabpfn-3-license-v1.0`) and a `TABPFN_TOKEN` API key from https://ux.priorlabs.ai
(cached at `~/.cache/tabpfn/auth_token`); an HuggingFace token alone is NOT sufficient. In a
headless run without the token, `.fit()` raises `TabPFNLicenseError`. Benchmark vs CatBoost is
in `wave2_decision.md` ("Обновление Wave 4"): TabPFN-3 ≈ CatBoost on internal CV, both negative
on the external chemistry holdout.
