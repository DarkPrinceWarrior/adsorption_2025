#!/usr/bin/env python3
"""
Hyperparameter tuning for CatBoost Forward Models using Optuna.

Tunes CatBoost hyperparameters via 5-fold CV, then prints the best
configuration for updating config.py.

Usage:
    python scripts/tune_hyperparams.py --data data/SEC_SYN_with_features_enriched.csv
    python scripts/tune_hyperparams.py --trials 100 --target "E0, кДж/моль"
"""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from adsorb_synthesis.data_processing import (
    load_dataset, build_lookup_tables, prepare_forward_dataset,
)
from adsorb_synthesis.constants import (
    RANDOM_SEED, FORWARD_MODEL_TARGETS, RARE_METALS_THRESHOLD,
)
from adsorb_synthesis.config import FORWARD_MODEL_CONFIG
from adsorb_synthesis.feature_selection import (
    select_features_advanced, get_curated_features,
)
from adsorb_synthesis.physics_losses import compute_physics_penalty

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_strat_key(X: pd.DataFrame, y_target: pd.Series) -> pd.Series:
    if 'Металл' in X.columns:
        metal_counts = X['Металл'].value_counts()
        rare = metal_counts[metal_counts < RARE_METALS_THRESHOLD].index.tolist()
        metal_group = X['Металл'].apply(lambda m: 'Other' if m in rare else m)
    else:
        metal_group = pd.Series(['Unknown'] * len(X), index=X.index)
    try:
        bins = pd.qcut(y_target, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                        duplicates='drop')
    except ValueError:
        bins = pd.Series(['All'] * len(y_target), index=y_target.index)
    return metal_group.astype(str) + '_' + bins.astype(str)


def tune_for_target(
    X: pd.DataFrame,
    y_target: pd.Series,
    cat_features: list,
    selected_features: list,
    sample_weights: np.ndarray,
    strat_key: pd.Series,
    n_trials: int = 80,
    n_splits: int = 5,
) -> Dict:
    """Run Optuna HP search for a single target, return best params."""

    cat_features_sel = [c for c in selected_features if c in cat_features]

    def objective(trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 400, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15,
                                                  log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0,
                                                log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',
                                                      0.5, 1.0),
            'loss_function': 'RMSE',
            'random_seed': RANDOM_SEED,
            'verbose': False,
            'allow_writing_files': False,
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_SEED)
        oof = np.full(len(X), np.nan)

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key)):
            model = CatBoostRegressor(**params, cat_features=cat_features_sel)
            model.fit(
                X.iloc[tr_idx][selected_features],
                y_target.iloc[tr_idx],
                sample_weight=sample_weights[tr_idx],
                eval_set=(X.iloc[va_idx][selected_features],
                          y_target.iloc[va_idx]),
                early_stopping_rounds=80,
                use_best_model=True,
            )
            oof[va_idx] = model.predict(X.iloc[va_idx][selected_features])

        rmse = np.sqrt(mean_squared_error(y_target, oof))
        return rmse

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best RMSE: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def main():
    parser = argparse.ArgumentParser(
        description="Tune CatBoost hyperparameters for Forward Models")
    parser.add_argument("--data", type=str,
                        default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--trials", type=int, default=80,
                        help="Optuna trials per target")
    parser.add_argument("--target", type=str, default=None,
                        help="Tune only this target (default: all)")
    parser.add_argument("--output", type=str,
                        default="artifacts/best_hyperparams.json",
                        help="Save best params to JSON")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    df_raw = load_dataset(args.data)
    lookup_tables = build_lookup_tables(df_raw)
    X, y = prepare_forward_dataset(df_raw, lookup_tables=lookup_tables)

    cat_features = [c for c in X.columns
                    if X[c].dtype.name in ['object', 'category']]

    physics_penalty = compute_physics_penalty(df_raw).reindex(X.index).fillna(0.0)
    sample_weights = 1.0 + FORWARD_MODEL_CONFIG.physics_penalty_weight * physics_penalty.values

    targets_to_tune = FORWARD_MODEL_TARGETS
    if args.target:
        targets_to_tune = [args.target]

    all_best = {}

    for target in targets_to_tune:
        if target not in y.columns:
            print(f"Skipping {target}: not in dataset")
            continue

        print(f"\n=== Tuning: {target} ({args.trials} trials) ===")
        y_target = y[target]
        strat_key = _build_strat_key(X, y_target)

        # Feature selection (on fold-0 train — no leakage)
        skf_fs = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=RANDOM_SEED)
        fs_train_idx = list(skf_fs.split(X, strat_key))[0][0]

        keep_features, drop_features = get_curated_features()
        available_drop = [f for f in drop_features if f in X.columns]
        numeric_cols = [c for c in X.columns if c not in cat_features]
        curated_numeric = [c for c in numeric_cols if c not in available_drop]

        selected_features, _ = select_features_advanced(
            X.iloc[fs_train_idx][cat_features + curated_numeric],
            y_target.iloc[fs_train_idx],
            categorical_cols=cat_features,
            corr_threshold=FORWARD_MODEL_CONFIG.feature_selection_corr_threshold,
            vif_threshold=FORWARD_MODEL_CONFIG.feature_selection_vif_threshold,
            max_features=FORWARD_MODEL_CONFIG.feature_selection_max_features,
            verbose=False,
        )

        best = tune_for_target(
            X, y_target, cat_features, selected_features,
            sample_weights, strat_key,
            n_trials=args.trials,
        )
        all_best[target] = best

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)
    print(f"\nBest hyperparameters saved to {args.output}")

    # Print config.py suggestion
    print("\n=== Suggested config.py update ===")
    if all_best:
        # Average across targets for a single config
        keys = list(next(iter(all_best.values())).keys())
        avg = {}
        for k in keys:
            vals = [all_best[t][k] for t in all_best if k in all_best[t]]
            if isinstance(vals[0], (int, float)):
                avg[k] = round(sum(vals) / len(vals), 4)
        print("CatBoostConfig(")
        for k, v in avg.items():
            print(f"    {k}={v},")
        print(")")


if __name__ == "__main__":
    main()
