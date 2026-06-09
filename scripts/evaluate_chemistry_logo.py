#!/usr/bin/env python3
"""Leave-One-Chemistry-Out (LOGO) cross-validation for the forward model.

Robust out-of-chemistry generalization estimate that replaces the single, noisy
hash-based holdout split (audit item #3). Every `Металл|Лиганд` group is held out
exactly once; the model is trained on all other groups (lookup tables built from
train only — no leakage) and predicts the held-out group. We report pooled and
per-group metrics, plus a y-scrambling null baseline to gauge significance.

Usage:
    PYTHONPATH=src python scripts/evaluate_chemistry_logo.py \
        --data data/SEC_SYN_with_features_enriched.csv \
        --output-dir artifacts/forward_logo --permutations 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adsorb_synthesis.config import get_catboost_config
from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS, RANDOM_SEED
from adsorb_synthesis.data_processing import (
    build_lookup_tables,
    load_dataset,
    prepare_forward_dataset,
)
from adsorb_synthesis.holdout_evaluation import (
    DEFAULT_HOLDOUT_GROUP_COLUMNS,
    build_chemistry_group_keys,
    safe_target_name,
)


def _prep_for_catboost(frame: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if col in cat_cols:
            out[col] = out[col].astype("string").fillna("nan").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _logo_oof(
    df: pd.DataFrame,
    target: str,
    *,
    params: Dict,
    group_columns,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """Return (y_true, y_oof, per_group_metrics) for one target via LOGO-CV."""
    groups = build_chemistry_group_keys(df, group_columns=group_columns)
    n = len(df)
    y_oof = np.full(n, np.nan, dtype=float)
    y_true_full = np.full(n, np.nan, dtype=float)
    per_group: Dict[str, Dict[str, float]] = {}

    for group in sorted(groups.unique()):
        test_mask = (groups == group).to_numpy()
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        df_train = df.loc[train_mask].reset_index(drop=True)
        df_test = df.loc[test_mask].reset_index(drop=True)

        lookup_tables = build_lookup_tables(df_train)
        X_train, y_train = prepare_forward_dataset(df_train, lookup_tables=lookup_tables)
        X_test, y_test = prepare_forward_dataset(df_test, lookup_tables=lookup_tables)

        feats = [c for c in X_train.columns if c in X_test.columns]
        cat_cols = [
            c for c in feats
            if (X_train[c].dtype == object) or str(X_train[c].dtype) == "category"
        ]

        model = CatBoostRegressor(**params, cat_features=cat_cols)
        model.fit(_prep_for_catboost(X_train[feats], cat_cols), y_train[target])
        pred = np.asarray(
            model.predict(_prep_for_catboost(X_test[feats], cat_cols)), dtype=float
        )

        pos = np.where(test_mask)[0]
        y_oof[pos] = pred
        y_true_full[pos] = y_test[target].to_numpy(dtype=float)

        yt = y_test[target].to_numpy(dtype=float)
        per_group[group] = {
            "n": int(test_mask.sum()),
            # R2 needs >=2 points and variance; report NaN otherwise
            "R2": float(r2_score(yt, pred)) if len(yt) >= 2 and np.ptp(yt) > 0 else None,
            "RMSE": float(np.sqrt(mean_squared_error(yt, pred))),
            "MAE": float(mean_absolute_error(yt, pred)),
        }

    valid = ~np.isnan(y_oof)
    return y_true_full[valid], y_oof[valid], per_group


def _pooled(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2_pooled": float(r2_score(y_true, y_pred)),
        "RMSE_pooled": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_pooled": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Chemistry-Out CV for the forward CatBoost model."
    )
    parser.add_argument("--data", default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--output-dir", default="artifacts/forward_logo")
    parser.add_argument("--permutations", type=int, default=3,
                        help="Number of y-scrambling permutations for the null baseline.")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override CatBoost iterations (lower = faster LOGO eval).")
    parser.add_argument("--validation-mode", choices=["warn", "strict"], default="warn")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, validation_mode=args.validation_mode).reset_index(drop=True)
    group_columns = DEFAULT_HOLDOUT_GROUP_COLUMNS
    groups = build_chemistry_group_keys(df, group_columns=group_columns)
    rng = np.random.default_rng(RANDOM_SEED)

    summary: Dict[str, Dict] = {
        "method": "leave-one-chemistry-out",
        "group_columns": list(group_columns),
        "n_groups": int(groups.nunique()),
        "n_rows": int(len(df)),
        "targets": {},
    }

    for target in FORWARD_MODEL_TARGETS:
        if target not in df.columns:
            continue
        print(f"\n=== LOGO-CV target: {target} ===")
        params = get_catboost_config(target).to_params(random_state=RANDOM_SEED)
        params.pop("random_seed", None)
        params["verbose"] = False
        params["allow_writing_files"] = False
        if args.iterations is not None:
            params["iterations"] = args.iterations

        y_true, y_oof, per_group = _logo_oof(
            df, target, params=params, group_columns=group_columns
        )
        pooled = _pooled(y_true, y_oof)
        print(f"  pooled R2={pooled['R2_pooled']:.4f} RMSE={pooled['RMSE_pooled']:.4f} "
              f"MAE={pooled['MAE_pooled']:.4f} (n={pooled['n']})")

        # y-scrambling null baseline (target permuted globally, full LOGO re-run)
        null_r2: List[float] = []
        for p in range(max(0, args.permutations)):
            df_perm = df.copy()
            df_perm[target] = rng.permutation(df[target].to_numpy())
            yt_p, yp_p, _ = _logo_oof(
                df_perm, target, params=params, group_columns=group_columns
            )
            null_r2.append(float(r2_score(yt_p, yp_p)))
        if null_r2:
            print(f"  y-scrambling null pooled R2: mean={np.mean(null_r2):.4f} "
                  f"(min={min(null_r2):.4f} max={max(null_r2):.4f})")

        pred_df = pd.DataFrame({"y_actual": y_true, "y_oof": y_oof})
        pred_df.to_csv(out / f"logo_predictions_{safe_target_name(target)}.csv", index=False)

        summary["targets"][target] = {
            **pooled,
            "per_group": per_group,
            "y_scramble_null_R2_mean": float(np.mean(null_r2)) if null_r2 else None,
            "y_scramble_null_R2": null_r2,
        }

    with open(out / "logo_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nLOGO-CV complete. Results saved to {out}")


if __name__ == "__main__":
    main()
