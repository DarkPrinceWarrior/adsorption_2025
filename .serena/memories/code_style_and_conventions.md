# Code Style And Conventions

Observed conventions:
- Python code generally starts with `from __future__ import annotations`.
- Type hints are used broadly, commonly with `Dict`, `List`, `Optional`, `Tuple`, `Sequence`, `Iterable`, and dataclasses.
- Configuration and lightweight data containers use `@dataclass`, often `frozen=True` for immutable config (`config.py`, `data_processing.py`, `holdout_evaluation.py`).
- Scripts usually expose a `main() -> None` entrypoint and an `if __name__ == "__main__": main()` guard.
- CLI scripts use `argparse.ArgumentParser`.
- Package code is under `src/adsorb_synthesis`; scripts are top-level under `scripts/` and are run with `PYTHONPATH=src`.
- Tests are under `tests/` and use pytest.
- Existing tests include direct assertions, `pytest.raises`, `pytest.warns`, and small DataFrame fixtures.
- Docstrings are present where useful, including short module or test-support docstrings.
- README and user-facing workflow docs are primarily in Russian; code identifiers and comments are mostly English.

Domain conventions:
- Forward targets are `E0`, `x0`, and `Sme`.
- Physical validation supports `warn` and `strict` modes; README recommends `warn` for the current dataset because some rows violate constraints.
- DataFrame feature engineering helpers may support `inplace`; use non-mutating mode when avoiding side effects matters.
- Feature selection should avoid leakage by remaining fold-local in model validation/training code.
- RDKit ligand descriptors are intentionally excluded from the production model in README rationale because only four unique ligands make them lookup-like.

Formatting/linting:
- No dedicated formatter/linter config was found during onboarding (`pyproject.toml`, Makefile, tox, Ruff/Black/isort/mypy config not present).
- Preserve the local style of the edited file and keep changes localized.