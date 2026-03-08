#!/usr/bin/env python3
"""Aggregate wave 2 forward and inverse benchmark artifacts into comparable tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


FORWARD_TARGETS = ["E0, кДж/моль", "х0, нм", "Sme, м2/г"]
DEFAULT_FORWARD_PATHS = [
    "artifacts/forward_models",
    "artifacts/tabpfn_smoke_v641",
]
DEFAULT_INVERSE_SHORTLISTS = [
    "artifacts/predictions_bofire.csv",
    "artifacts/predictions_botorch.csv",
    "artifacts/predictions_baybe.csv",
    "artifacts/inverse_direct/predictions.csv",
]
DEFAULT_INVERSE_POOLS = [
    "artifacts/predictions_bofire_all.csv",
    "artifacts/predictions_botorch_all.csv",
    "artifacts/predictions_baybe_history.csv",
    "artifacts/inverse_direct/predictions.csv",
]


def _metric_as_float(payload: dict, key: str) -> float:
    value = payload.get(key, np.nan)
    return np.nan if value is None else float(value)


def infer_forward_backend(models_dir: Path, metrics: dict) -> str:
    payloads = list(metrics.values())
    backend = payloads[0].get("backend") if payloads else None
    if backend:
        return str(backend)
    if any(path.name.startswith("catboost_") and path.suffix == ".cbm" for path in models_dir.iterdir()):
        return "catboost"
    return models_dir.name or "forward_model"


def infer_inverse_backend(path: Path, df: pd.DataFrame) -> str:
    if "backend" in df.columns and not df["backend"].dropna().empty:
        return str(df["backend"].dropna().iloc[0])
    name = path.stem.lower()
    if "inverse_direct" in name or "direct" in name:
        return "inverse_direct"
    if "bofire" in name:
        return "bofire"
    if "botorch" in name:
        return "botorch"
    return path.stem


def load_forward_runs(paths: Iterable[str]) -> List[dict]:
    runs: List[dict] = []
    seen: set[str] = set()
    for raw_path in paths:
        candidate = Path(raw_path).resolve()
        if not candidate.exists():
            continue
        metric_file = candidate / "metrics.json"
        search_dirs: List[Path]
        if metric_file.exists():
            search_dirs = [candidate]
        else:
            search_dirs = [subdir for subdir in candidate.iterdir() if subdir.is_dir() and (subdir / "metrics.json").exists()]
        for models_dir in search_dirs:
            models_dir_key = str(models_dir)
            if models_dir_key in seen:
                continue
            metrics = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))
            runs.append({
                "backend": infer_forward_backend(models_dir, metrics),
                "models_dir": models_dir,
                "metrics": metrics,
            })
            seen.add(models_dir_key)
    return runs


def build_forward_benchmark(runs: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for run in runs:
        backend = run["backend"]
        metrics = run["metrics"]
        for target in FORWARD_TARGETS:
            if target not in metrics:
                continue
            payload = metrics[target]
            rows.append({
                "backend": backend,
                "target": target,
                "R2_oof": _metric_as_float(payload, "R2_oof") if payload.get("R2_oof") is not None else _metric_as_float(payload, "R2"),
                "RMSE_oof": _metric_as_float(payload, "RMSE_oof") if payload.get("RMSE_oof") is not None else _metric_as_float(payload, "RMSE"),
                "MAE_oof": _metric_as_float(payload, "MAE_oof") if payload.get("MAE_oof") is not None else _metric_as_float(payload, "MAE"),
                "R2_production": _metric_as_float(payload, "R2_production"),
                "RMSE_production": _metric_as_float(payload, "RMSE_production"),
                "MAE_production": _metric_as_float(payload, "MAE_production"),
                "interval_coverage": _metric_as_float(payload, "interval_coverage"),
                "interval_width_mean": _metric_as_float(payload, "interval_width_mean"),
                "n_selected_features": len(payload.get("selected_features", [])),
                "ensemble_members": payload.get("ensemble_members"),
                "cv_folds": payload.get("cv_folds"),
                "models_dir": str(run["models_dir"]),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "backend"]).reset_index(drop=True)


def _ensure_inverse_keys(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "chemistry_key" not in frame.columns:
        if {"Металл", "Лиганд", "Растворитель"}.issubset(frame.columns):
            frame["chemistry_key"] = frame[["Металл", "Лиганд", "Растворитель"]].astype(str).agg("|".join, axis=1)
        else:
            frame["chemistry_key"] = np.nan
    if "process_key" not in frame.columns:
        process_cols = {"Т.син., °С", "Т суш., °С", "Tрег, ᵒС"}
        if process_cols.issubset(frame.columns):
            frame["process_key"] = frame[["Т.син., °С", "Т суш., °С", "Tрег, ᵒС"]].astype(str).agg("|".join, axis=1)
        else:
            frame["process_key"] = np.nan
    if "selection_mode" not in frame.columns:
        frame["selection_mode"] = "shortlist"
    if "search_rank" not in frame.columns:
        frame["search_rank"] = np.arange(1, len(frame) + 1)
    return frame


def _read_existing_csvs(paths: Iterable[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        tables[str(path)] = _ensure_inverse_keys(df)
    return tables


def build_inverse_benchmark(shortlists: Dict[str, pd.DataFrame], pools: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    shortlist_by_backend = {
        infer_inverse_backend(Path(path_str), df): (Path(path_str), df)
        for path_str, df in shortlists.items()
    }
    pool_by_backend = {
        infer_inverse_backend(Path(path_str), df): (Path(path_str), df)
        for path_str, df in pools.items()
    }
    all_backends = sorted(set(shortlist_by_backend.keys()) | set(pool_by_backend.keys()))

    for backend in all_backends:
        shortlist_entry = shortlist_by_backend.get(backend)
        pool_entry = pool_by_backend.get(backend)
        shortlist = shortlist_entry[1] if shortlist_entry is not None else None
        pool = pool_entry[1] if pool_entry is not None else None
        reference = shortlist if shortlist is not None else pool
        if reference is None or reference.empty:
            continue

        shortlist_df = shortlist if shortlist is not None else reference
        pool_df = pool if pool is not None else reference

        shortlist_feasible = shortlist_df["feasible"].fillna(False).astype(bool) if "feasible" in shortlist_df.columns else pd.Series(dtype=bool)
        pool_feasible = pool_df["feasible"].fillna(False).astype(bool) if "feasible" in pool_df.columns else pd.Series(dtype=bool)
        shortlist_feasible_df = shortlist_df.loc[shortlist_feasible] if not shortlist_feasible.empty else shortlist_df.iloc[0:0]
        pool_feasible_df = pool_df.loc[pool_feasible] if not pool_feasible.empty else pool_df.iloc[0:0]

        rows.append({
            "backend": backend,
            "shortlist_path": str(shortlist_entry[0]) if shortlist_entry is not None else "",
            "pool_path": str(pool_entry[0]) if pool_entry is not None else "",
            "shortlist_size": int(len(shortlist_df)),
            "pool_size": int(len(pool_df)),
            "shortlist_feasibility_rate": float(shortlist_feasible.mean()) if len(shortlist_df) else np.nan,
            "pool_feasibility_rate": float(pool_feasible.mean()) if len(pool_df) else np.nan,
            "best_score_shortlist": float(pd.to_numeric(shortlist_df.get("score"), errors="coerce").min()) if "score" in shortlist_df.columns and len(shortlist_df) else np.nan,
            "best_score_pool": float(pd.to_numeric(pool_df.get("score"), errors="coerce").min()) if "score" in pool_df.columns and len(pool_df) else np.nan,
            "mean_score_shortlist": float(pd.to_numeric(shortlist_df.get("score"), errors="coerce").mean()) if "score" in shortlist_df.columns and len(shortlist_df) else np.nan,
            "mean_score_pool": float(pd.to_numeric(pool_df.get("score"), errors="coerce").mean()) if "score" in pool_df.columns and len(pool_df) else np.nan,
            "unique_chemistries_shortlist": int(shortlist_feasible_df["chemistry_key"].nunique(dropna=True)) if "chemistry_key" in shortlist_feasible_df.columns else 0,
            "unique_processes_shortlist": int(shortlist_feasible_df["process_key"].nunique(dropna=True)) if "process_key" in shortlist_feasible_df.columns else 0,
            "interval_width_total_mean_shortlist": float(pd.to_numeric(shortlist_df.get("interval_width_total"), errors="coerce").mean()) if "interval_width_total" in shortlist_df.columns and len(shortlist_df) else np.nan,
            "interval_width_total_mean_pool": float(pd.to_numeric(pool_df.get("interval_width_total"), errors="coerce").mean()) if "interval_width_total" in pool_df.columns and len(pool_df) else np.nan,
            "selection_mode": str(shortlist_df.get("selection_mode", pd.Series(["shortlist"])).iloc[0]) if len(shortlist_df) else "shortlist",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("backend").reset_index(drop=True)


def print_forward_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Forward benchmark: no runs found")
        return
    print("Forward benchmark:")
    for target in FORWARD_TARGETS:
        target_df = df.loc[df["target"] == target]
        if target_df.empty:
            continue
        print(f"  {target}")
        for _, row in target_df.sort_values("R2_oof", ascending=False).iterrows():
            coverage = row["interval_coverage"]
            coverage_str = "NA" if pd.isna(coverage) else f"{coverage:.3f}"
            print(
                f"    {row['backend']}: "
                f"R2_oof={row['R2_oof']:.4f} "
                f"RMSE_oof={row['RMSE_oof']:.4f} "
                f"R2_prod={row['R2_production']:.4f} "
                f"coverage={coverage_str}"
            )


def print_inverse_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Inverse benchmark: no runs found")
        return
    print("Inverse benchmark:")
    for _, row in df.sort_values("best_score_pool").iterrows():
        print(
            f"  {row['backend']}: "
            f"pool={int(row['pool_size'])} "
            f"shortlist={int(row['shortlist_size'])} "
            f"feasible_pool={row['pool_feasibility_rate']:.3f} "
            f"best_score={row['best_score_pool']:.4f} "
            f"chemistries={int(row['unique_chemistries_shortlist'])} "
            f"processes={int(row['unique_processes_shortlist'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate wave 2 forward and inverse benchmark artifacts.")
    parser.add_argument("--forward-models", nargs="+", default=DEFAULT_FORWARD_PATHS)
    parser.add_argument("--inverse-shortlists", nargs="+", default=DEFAULT_INVERSE_SHORTLISTS)
    parser.add_argument("--inverse-pools", nargs="+", default=DEFAULT_INVERSE_POOLS)
    parser.add_argument("--output-dir", type=str, default="artifacts/wave2_benchmark")
    args = parser.parse_args()

    forward_runs = load_forward_runs(args.forward_models)
    forward_benchmark = build_forward_benchmark(forward_runs)

    inverse_shortlists = _read_existing_csvs(args.inverse_shortlists)
    inverse_pools = _read_existing_csvs(args.inverse_pools)
    inverse_benchmark = build_inverse_benchmark(inverse_shortlists, inverse_pools)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forward_path = output_dir / "forward_benchmark.csv"
    inverse_path = output_dir / "inverse_benchmark.csv"
    summary_path = output_dir / "benchmark_summary.json"

    forward_benchmark.to_csv(forward_path, index=False)
    inverse_benchmark.to_csv(inverse_path, index=False)
    summary = {
        "forward_rows": int(len(forward_benchmark)),
        "inverse_rows": int(len(inverse_benchmark)),
        "forward_backends": sorted(forward_benchmark["backend"].dropna().astype(str).unique().tolist()) if not forward_benchmark.empty else [],
        "inverse_backends": sorted(inverse_benchmark["backend"].dropna().astype(str).unique().tolist()) if not inverse_benchmark.empty else [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_forward_summary(forward_benchmark)
    print_inverse_summary(inverse_benchmark)
    print(f"Saved forward benchmark to {forward_path}")
    print(f"Saved inverse benchmark to {inverse_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
