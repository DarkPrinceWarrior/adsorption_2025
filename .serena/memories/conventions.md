# Conventions

- `from __future__ import annotations` at top of package modules.
- Type hints throughout; `X | Y` unions (3.10+); builtin generics; `typing` `Optional/Sequence/Iterable` where convenient.
- `@dataclass` for data-holding structs; `frozen=True` for immutable config (`config.py`, `data_processing.py`, `holdout_evaluation.py`).
- Scripts: `def main() -> None` + `if __name__ == "__main__": main()`; CLI via `argparse.ArgumentParser`.
- Concise module-level docstrings where useful; no per-function docstrings by default; names self-documenting.
- README and user-facing workflow docs in Russian; code identifiers and comments in English.
- DataFrame feature-engineering helpers expose `inplace` (default `True`); pass `inplace=False` when avoiding side effects.
- No formatter/linter to fall back on — preserve the local style of the edited file and keep changes localized.

Domain conventions:
- Forward targets are exactly `E0`, `x0`, `Sme`.
- Feature selection stays fold-local (no leakage) in training/validation code.
- Physics validation has `warn`/`strict` modes; current dataset needs `warn`.
