#!/usr/bin/env python3
"""Evaluate forward backends on an external chemistry holdout split."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adsorb_synthesis.constants import RANDOM_SEED
from adsorb_synthesis.data_processing import load_dataset
from adsorb_synthesis.holdout_evaluation import (
    DEFAULT_HOLDOUT_GROUP_COLUMNS,
    build_chemistry_holdout_split,
    evaluate_catboost_holdout,
    evaluate_tabpfn_holdout,
    load_split_manifest,
    safe_target_name,
    write_split_manifest,
)


def _run(command: List[str], *, cwd: Path) -> None:
    print(f"\n$ {' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, cwd=cwd, check=True)


def _write_predictions(output_dir: Path, predictions: Dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, df in predictions.items():
        df.to_csv(output_dir / f"predictions_{safe_target_name(target)}.csv", index=False)


def _write_metrics(output_dir: Path, metrics: Dict[str, Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "holdout_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def evaluate_holdout(
    *,
    data_path: str,
    output_dir: str,
    backend: str,
    holdout_fraction: float,
    validation_mode: str,
    split_manifest: str | None,
    force_split: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python = str(repo_root / ".venv" / "bin" / "python")
    output_path = (repo_root / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(data_path, validation_mode=validation_mode).reset_index(drop=True)
    manifest_path = Path(split_manifest).resolve() if split_manifest else output_path / "split_manifest.json"
    if manifest_path.exists() and not force_split:
        split = load_split_manifest(manifest_path)
    else:
        split = build_chemistry_holdout_split(
            df_raw,
            group_columns=DEFAULT_HOLDOUT_GROUP_COLUMNS,
            holdout_fraction=holdout_fraction,
            seed=RANDOM_SEED,
        )
        write_split_manifest(split, manifest_path)

    df_train = df_raw.iloc[split.train_indices].reset_index(drop=True)
    df_holdout = df_raw.iloc[split.holdout_indices].reset_index(drop=True)
    train_csv_path = output_path / "train_partition.csv"
    df_train.to_csv(train_csv_path, index=False)

    backends: Iterable[str] = ("catboost", "tabpfn") if backend == "all" else (backend,)
    trained_root = output_path / "trained"
    trained_root.mkdir(parents=True, exist_ok=True)

    for current_backend in backends:
        backend_train_dir = trained_root / current_backend
        if not (backend_train_dir / "metrics.json").exists():
            _run(
                [
                    python,
                    "scripts/train_forward_model.py",
                    "--data",
                    str(train_csv_path),
                    "--output",
                    str(backend_train_dir),
                    "--backend",
                    current_backend,
                    "--validation-mode",
                    validation_mode,
                ],
                cwd=repo_root,
            )

        holdout_backend_dir = output_path / current_backend
        if current_backend == "catboost":
            metrics, predictions = evaluate_catboost_holdout(
                models_dir=backend_train_dir,
                df_train_raw=df_train,
                df_holdout_raw=df_holdout,
            )
        elif current_backend == "tabpfn":
            metrics, predictions = evaluate_tabpfn_holdout(
                models_dir=backend_train_dir,
                df_train_raw=df_train,
                df_holdout_raw=df_holdout,
            )
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"Unsupported backend: {current_backend}")
        _write_metrics(holdout_backend_dir, metrics)
        _write_predictions(holdout_backend_dir, predictions)
        print(f"Saved holdout evaluation for {current_backend} to {holdout_backend_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run external chemistry-split holdout evaluation for forward backends.",
    )
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/forward_holdout")
    parser.add_argument("--backend", choices=["catboost", "tabpfn", "all"], default="all")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=str)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--validation-mode", choices=["warn", "strict"], default="warn")
    args = parser.parse_args()
    evaluate_holdout(
        data_path=args.data,
        output_dir=args.output_dir,
        backend=args.backend,
        holdout_fraction=args.holdout_fraction,
        validation_mode=args.validation_mode,
        split_manifest=args.split_manifest,
        force_split=args.force_split,
    )


if __name__ == "__main__":
    main()
