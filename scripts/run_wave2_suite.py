#!/usr/bin/env python3
"""Run the full runnable wave 2 comparison suite."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(command: List[str], *, cwd: Path) -> None:
    print(f"\n$ {' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, cwd=cwd, check=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _artifact_exists(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the runnable wave 2 comparison suite.")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--catboost-models", type=str, default="artifacts/forward_models")
    parser.add_argument("--tabpfn-models", type=str, help="Optional precomputed TabPFN models dir.")
    parser.add_argument("--forward-holdout-dir", type=str, help="Optional precomputed forward holdout artifact root.")
    parser.add_argument("--bofire-shortlist", type=str, default="artifacts/predictions_bofire_shortlist.csv")
    parser.add_argument("--bofire-pool", type=str, default="artifacts/predictions_bofire_all.csv")
    parser.add_argument("--botorch-shortlist", type=str, default="artifacts/predictions_botorch.csv")
    parser.add_argument("--botorch-pool", type=str, default="artifacts/predictions_botorch_all.csv")
    parser.add_argument("--baybe-shortlist", type=str, help="Optional precomputed BayBE shortlist CSV.")
    parser.add_argument("--baybe-pool", type=str, help="Optional precomputed BayBE history CSV.")
    parser.add_argument("--inverse-direct-dir", type=str, help="Optional precomputed inverse direct output dir.")
    parser.add_argument("--run-dir", type=str, default="artifacts/wave2_runs/default")
    parser.add_argument("--mode", choices=["full", "forward-only", "inverse-only", "report-only"], default="full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--E0", type=float, default=15.0)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--Sme", type=float, default=100.0)
    parser.add_argument("--tabpfn-only", action="store_true", help="Only train TabPFN challenger, reuse CatBoost baseline.")
    parser.add_argument("--baybe-trials", type=int, default=36)
    parser.add_argument("--baybe-batch-size", type=int, default=6)
    parser.add_argument("--inverse-direct-cv-folds", type=int, default=5)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    python = str(cwd / ".venv" / "bin" / "python")
    run_dir = (cwd / args.run_dir).resolve()
    forward_dir = run_dir / "forward"
    holdout_dir = (cwd / args.forward_holdout_dir).resolve() if args.forward_holdout_dir else (run_dir / "holdout")
    inverse_dir = run_dir / "inverse"
    benchmark_dir = run_dir / "benchmark"
    report_dir = run_dir / "report"
    manifest_path = run_dir / "suite_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    catboost_models = (cwd / args.catboost_models).resolve()
    tabpfn_models = (cwd / args.tabpfn_models).resolve() if args.tabpfn_models else (forward_dir / "tabpfn")
    holdout_catboost_dir = holdout_dir / "catboost"
    holdout_tabpfn_dir = holdout_dir / "tabpfn"
    baybe_shortlist = (cwd / args.baybe_shortlist).resolve() if args.baybe_shortlist else (inverse_dir / "predictions_baybe.csv")
    baybe_pool = (cwd / args.baybe_pool).resolve() if args.baybe_pool else (inverse_dir / "predictions_baybe_history.csv")
    inverse_direct_dir = (cwd / args.inverse_direct_dir).resolve() if args.inverse_direct_dir else (inverse_dir / "inverse_direct")
    inverse_direct_predictions = inverse_direct_dir / "predictions.csv"
    bofire_shortlist = (cwd / args.bofire_shortlist).resolve()
    bofire_pool = (cwd / args.bofire_pool).resolve()
    botorch_shortlist = (cwd / args.botorch_shortlist).resolve()
    botorch_pool = (cwd / args.botorch_pool).resolve()

    do_forward = args.mode in {"full", "forward-only"}
    do_inverse = args.mode in {"full", "inverse-only"}
    do_report = args.mode in {"full", "report-only"}

    if do_forward:
        if not _artifact_exists(tabpfn_models / "metrics.json") or args.force:
            forward_dir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    python,
                    "scripts/train_forward_model.py",
                    "--data",
                    args.data,
                    "--output",
                    str(forward_dir),
                    "--backend",
                    "tabpfn" if args.tabpfn_only else "all",
                    "--validation-mode",
                    "warn",
                ],
                cwd=cwd,
            )
            if args.tabpfn_only:
                tabpfn_models = (forward_dir if (forward_dir / "metrics.json").exists() else forward_dir / "tabpfn").resolve()
        elif args.tabpfn_models:
            print(f"Reusing existing TabPFN models: {tabpfn_models}")
        if not _artifact_exists(holdout_catboost_dir / "holdout_metrics.json") or not _artifact_exists(holdout_tabpfn_dir / "holdout_metrics.json") or args.force:
            holdout_command = [
                python,
                "scripts/evaluate_forward_holdout.py",
                "--data",
                args.data,
                "--output-dir",
                str(holdout_dir),
                "--backend",
                "all",
                "--holdout-fraction",
                str(args.holdout_fraction),
                "--validation-mode",
                "warn",
            ]
            if args.force:
                holdout_command.append("--force-split")
            _run(holdout_command, cwd=cwd)

    if do_inverse:
        if (not _artifact_exists(baybe_shortlist) or not _artifact_exists(baybe_pool) or args.force) and not args.baybe_shortlist:
            _ensure_parent(baybe_shortlist)
            _run(
                [
                    python,
                    "scripts/run_baybe_campaign.py",
                    "--models-dir",
                    str(catboost_models),
                    "--data",
                    args.data,
                    "--E0",
                    str(args.E0),
                    "--x0",
                    str(args.x0),
                    "--Sme",
                    str(args.Sme),
                    "--trials",
                    str(args.baybe_trials),
                    "--batch-size",
                    str(args.baybe_batch_size),
                    "--output",
                    str(baybe_shortlist),
                    "--history-output",
                    str(baybe_pool),
                    "--campaign-state",
                    str(inverse_dir / "predictions_baybe_campaign.json"),
                ],
                cwd=cwd,
            )
        elif args.baybe_shortlist:
            print(f"Reusing existing BayBE artifacts: {baybe_shortlist}")

        if not _artifact_exists(inverse_direct_predictions) or args.force:
            inverse_direct_dir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    python,
                    "scripts/train_inverse_direct.py",
                    "--models-dir",
                    str(catboost_models),
                    "--data",
                    args.data,
                    "--output-dir",
                    str(inverse_direct_dir),
                    "--cv-folds",
                    str(args.inverse_direct_cv_folds),
                ],
                cwd=cwd,
            )
        elif args.inverse_direct_dir:
            print(f"Reusing existing inverse direct artifacts: {inverse_direct_dir}")

    if do_report:
        if not _artifact_exists(tabpfn_models / "metrics.json"):
            raise FileNotFoundError(f"TabPFN metrics not found: {tabpfn_models / 'metrics.json'}")
        _run(
            [
                python,
                "scripts/benchmark_wave2.py",
                "--forward-models",
                str(catboost_models),
                str(tabpfn_models),
                "--forward-holdouts",
                str(holdout_catboost_dir),
                str(holdout_tabpfn_dir),
                "--inverse-shortlists",
                str(bofire_shortlist),
                str(botorch_shortlist),
                str(baybe_shortlist),
                str(inverse_direct_predictions),
                "--inverse-pools",
                str(bofire_pool),
                str(botorch_pool),
                str(baybe_pool),
                str(inverse_direct_predictions),
                "--output-dir",
                str(benchmark_dir),
            ],
            cwd=cwd,
        )
        _run(
            [
                python,
                "scripts/generate_wave2_report.py",
                "--benchmark-dir",
                str(benchmark_dir),
                "--output-dir",
                str(report_dir),
            ],
            cwd=cwd,
        )

    manifest = {
        "catboost_models": str(catboost_models),
        "tabpfn_models": str(tabpfn_models),
        "forward_holdout_dir": str(holdout_dir),
        "bofire_shortlist": str(bofire_shortlist),
        "bofire_pool": str(bofire_pool),
        "botorch_shortlist": str(botorch_shortlist),
        "botorch_pool": str(botorch_pool),
        "baybe_shortlist": str(baybe_shortlist),
        "baybe_pool": str(baybe_pool),
        "inverse_direct_predictions": str(inverse_direct_predictions),
        "benchmark_dir": str(benchmark_dir),
        "report_dir": str(report_dir),
        "targets": {"E0": args.E0, "x0": args.x0, "Sme": args.Sme},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved suite manifest to {manifest_path}")


if __name__ == "__main__":
    main()
