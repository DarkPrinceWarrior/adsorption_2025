#!/usr/bin/env python3
"""Tune CatBoost forward-model hyperparameters with fold-local feature selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.config import FORWARD_MODEL_CONFIG
from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS, RANDOM_SEED
from adsorb_synthesis.data_processing import (
    build_lookup_tables,
    load_dataset,
    prepare_forward_dataset,
)
from adsorb_synthesis.forward_modeling import (
    build_stratification_key,
    compute_quality_weights,
    select_curated_features,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_for_target(
    X: pd.DataFrame,
    y_target: pd.Series,
    cat_features: list[str],
    sample_weights: np.ndarray,
    strat_key: pd.Series,
    *,
    n_trials: int = 80,
    n_splits: int = 5,
) -> Dict:
    """Run Optuna tuning with per-fold feature selection."""

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = list(outer_cv.split(X, strat_key))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 400, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "loss_function": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
        }
        oof_predictions = np.full(len(X), np.nan, dtype=float)

        for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y_target.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y_target.iloc[valid_idx]
            selected_features, _ = select_curated_features(
                X_train,
                y_train,
                cat_features,
                corr_threshold=FORWARD_MODEL_CONFIG.feature_selection_corr_threshold,
                vif_threshold=FORWARD_MODEL_CONFIG.feature_selection_vif_threshold,
                max_features=FORWARD_MODEL_CONFIG.feature_selection_max_features,
                verbose=False,
            )
            selected_cat = [feature for feature in selected_features if feature in cat_features]
            model = CatBoostRegressor(
                **params,
                random_seed=RANDOM_SEED + fold_idx,
                cat_features=selected_cat,
            )
            model.fit(
                X_train[selected_features],
                y_train,
                sample_weight=sample_weights[train_idx],
                eval_set=(X_valid[selected_features], y_valid),
                early_stopping_rounds=FORWARD_MODEL_CONFIG.early_stopping_rounds,
                use_best_model=True,
            )
            oof_predictions[valid_idx] = model.predict(X_valid[selected_features])

        return float(np.sqrt(mean_squared_error(y_target, oof_predictions)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best RMSE: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CatBoost hyperparameters for forward models.")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--output", type=str, default="artifacts/best_hyperparams.json")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    df_raw = load_dataset(args.data)
    lookup_tables = build_lookup_tables(df_raw)
    X, y = prepare_forward_dataset(df_raw, lookup_tables=lookup_tables)
    cat_features = [col for col in X.columns if X[col].dtype.name in ["object", "category"]]
    sample_weights = compute_quality_weights(
        df_raw.reindex(X.index),
        penalty_weight=FORWARD_MODEL_CONFIG.physics_penalty_weight,
        minimum_weight=FORWARD_MODEL_CONFIG.minimum_sample_weight,
    ).to_numpy(dtype=float)

    targets_to_tune = [args.target] if args.target else FORWARD_MODEL_TARGETS
    all_best: Dict[str, Dict] = {}

    for target in targets_to_tune:
        if target not in y.columns:
            print(f"Skipping {target}: target missing in dataset.")
            continue
        print(f"\n=== Tuning {target} ({args.trials} trials) ===")
        best = tune_for_target(
            X,
            y[target],
            cat_features,
            sample_weights,
            build_stratification_key(X, y[target]),
            n_trials=args.trials,
            n_splits=FORWARD_MODEL_CONFIG.n_ensemble_splits,
        )
        all_best[target] = best

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(all_best, handle, indent=2, ensure_ascii=False)
    print(f"\nBest hyperparameters saved to {args.output}")


if __name__ == "__main__":
    main()
