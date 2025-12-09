#!/usr/bin/env python3
"""
Train the Forward Model (Simulator) for Bayesian Optimization.

This script implements 'Stage 2: Forward Model Creation' from the BO plan.
It trains separate CatBoost regressors for each target property:
Recipe (Inputs) -> Physical Properties (Outputs).
"""

import argparse
import json
import os
import sys
from typing import Dict
import numpy as np

# Add src to path to import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from adsorb_synthesis.data_processing import load_dataset, build_lookup_tables, prepare_forward_dataset
from adsorb_synthesis.constants import RANDOM_SEED, FORWARD_MODEL_TARGETS, RARE_METALS_THRESHOLD
from adsorb_synthesis.feature_selection import (
    select_features_advanced,
    get_curated_features
)
from adsorb_synthesis.physics_losses import compute_physics_penalty


def fit_uncertainty_calibrator(sigmas: np.ndarray, abs_errors: np.ndarray) -> Dict:
    """Fit sigma->abs_error calibrator using isotonic regression or scale fallback."""
    sigmas = np.asarray(sigmas, dtype=float)
    abs_errors = np.asarray(abs_errors, dtype=float)
    mask = np.isfinite(sigmas) & np.isfinite(abs_errors)
    sigmas = sigmas[mask]
    abs_errors = abs_errors[mask]

    if len(sigmas) >= 20 and np.unique(sigmas).size >= 5:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sigmas, abs_errors)
        return {"type": "isotonic", "model": iso}

    # Fallback: linear scaling of sigma to median absolute error
    median_sigma = np.median(sigmas) if len(sigmas) else 1e-8
    scale = np.median(abs_errors) / max(median_sigma, 1e-8)
    return {"type": "scale", "scale": float(scale)}

def train_forward_models(
    data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    iterations: int = 1000
):
    print(f"Loading dataset from {data_path}...")
    # Load raw data with standard enrichment
    df_raw = load_dataset(data_path)
    
    # Build lookups for descriptors (Metal, Ligand, Solvent)
    lookup_tables = build_lookup_tables(df_raw)
    
    # Prepare X (Recipe) and y (Properties) specifically for the Forward Model
    print("Preparing Forward Model dataset (Data Flip)...")
    X, y = prepare_forward_dataset(df_raw, lookup_tables=lookup_tables)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features in X: {list(X.columns)}")
    
    # Identify categorical features for CatBoost
    cat_features = [col for col in X.columns if X[col].dtype.name in ['object', 'category']]
    print(f"Categorical features found: {cat_features}")
    
    metrics = {}
    models = {} # target -> list of model paths
    os.makedirs(output_dir, exist_ok=True)
    calibrators = {}
    
    # CV ensemble parameters
    n_splits = 5
    PHYSICS_PENALTY_WEIGHT = 1.0
    
    for target in FORWARD_MODEL_TARGETS:
        print(f"\n=== Training ENSEMBLE for target: {target} ===")
        
        if target not in y.columns:
            print(f"Skipping {target}: not found in targets.")
            continue

        y_target = y[target]

        # Stratification key (Metal + target bins)
        if 'Металл' in X.columns:
            metal_counts = X['Металл'].value_counts()
            rare_metals = metal_counts[metal_counts < RARE_METALS_THRESHOLD].index.tolist()
            metal_group = X['Металл'].apply(lambda m: 'Other' if m in rare_metals else m)
        else:
            metal_group = pd.Series(['Unknown'] * len(X), index=X.index)
        try:
            target_bins = pd.qcut(y_target, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        except ValueError:
            target_bins = pd.Series(['All'] * len(y_target), index=y_target.index)
        strat_key = metal_group.astype(str) + '_' + target_bins.astype(str)

        # Physics penalty as sample weight proxy
        physics_penalty = compute_physics_penalty(df_raw)
        physics_penalty = physics_penalty.reindex(X.index).fillna(0.0)
        sample_weights_full = 1.0 + PHYSICS_PENALTY_WEIGHT * physics_penalty.values
        print(f"  Mean physics penalty weight: {np.mean(sample_weights_full):.3f}")
        
        # Feature Selection (once per target on full data)
        print(f"  Advanced Feature Selection for {target}...")
        keep_features, drop_features = get_curated_features()
        available_drop = [f for f in drop_features if f in X.columns]
        numeric_cols = [c for c in X.columns if c not in cat_features]
        curated_numeric = [c for c in numeric_cols if c not in available_drop]
        selected_features, selection_report = select_features_advanced(
            X[cat_features + curated_numeric],
            y_target,
            categorical_cols=cat_features,
            corr_threshold=0.85,
            vif_threshold=10.0,
            max_features=15,
            verbose=False
        )
        n_removed = len(selection_report['removed_correlation']) + len(selection_report['removed_vif'])
        print(f"    Removed {n_removed} multicollinear features")
        print(f"    Selected {len(selected_features)} features: {selected_features[:5]}...")
        cat_features_sel = [c for c in selected_features if c in cat_features]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        model_paths = []
        fold_models = []
        oof_preds = np.full(len(X), np.nan, dtype=float)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
            seed = RANDOM_SEED + fold_idx
            print(f"  Fold {fold_idx+1}/{n_splits} (seed={seed})...")
            X_train_sel = X.iloc[train_idx][selected_features]
            X_val_sel = X.iloc[val_idx][selected_features]
            y_train_target = y_target.iloc[train_idx]
            y_val_target = y_target.iloc[val_idx]
            w_train = sample_weights_full[train_idx]
            w_val = sample_weights_full[val_idx]

            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=1.0,
                min_data_in_leaf=1,
                subsample=0.8,
                colsample_bylevel=0.8,
                loss_function='RMSE',
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                cat_features=cat_features_sel
            )

            model.fit(
                X_train_sel, y_train_target,
                sample_weight=w_train,
                eval_set=(X_val_sel, y_val_target),
                early_stopping_rounds=100,
                use_best_model=True
            )

            safe_target = target.replace('/', '_').replace(' ', '_')
            model_path = os.path.join(output_dir, f"catboost_{safe_target}_ens{fold_idx}.cbm")
            model.save_model(model_path)
            model_paths.append(model_path)
            fold_models.append(model)

            oof_preds[val_idx] = model.predict(X_val_sel)

        # Ensemble predictions on full data
        all_preds = np.stack([m.predict(X[selected_features]) for m in fold_models], axis=1)
        ensemble_mean = np.mean(all_preds, axis=1)
        ensemble_std = np.std(all_preds, axis=1)

        r2_all = r2_score(y_target, ensemble_mean)
        rmse_all = np.sqrt(mean_squared_error(y_target, ensemble_mean))
        mae_all = mean_absolute_error(y_target, ensemble_mean)
        print(f"  CV Ensemble R2: {r2_all:.4f}, RMSE: {rmse_all:.4f}, MAE: {mae_all:.4f}")
        print(f"  Avg Uncertainty (StdDev across folds): {np.mean(ensemble_std):.4f}")

        models[target] = model_paths

        safe_target = target.replace('/', '_').replace(' ', '_')
        predictions_df = pd.DataFrame({
            'y_actual': y_target.values,
            'y_pred': ensemble_mean,
            'y_std': ensemble_std,
            'y_oof': oof_preds
        })
        predictions_path = os.path.join(output_dir, f"predictions_{safe_target}.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  Saved predictions: {predictions_path}")

        physics_features = [f for f in selected_features if any(x in f for x in 
            ['metal_coord', 'ligand_3d', 'ligand_2d', 'Size_Ratio', 'Electronegativity_Diff', 'Jahn_Teller'])]

        abs_errors = np.abs(y_target.values - ensemble_mean)
        calibrator = fit_uncertainty_calibrator(ensemble_std, abs_errors)
        if calibrator["type"] == "isotonic":
            calibrator_meta = {"type": "isotonic", "n_samples": len(ensemble_std)}
        else:
            calibrator_meta = {"type": "scale", "n_samples": len(ensemble_std), "scale": calibrator.get("scale", None)}
        calibrators[target] = calibrator

        metrics[target] = {
            "R2": r2_all,
            "RMSE": rmse_all,
            "MAE": mae_all,
            "selected_features": selected_features,
            "physics_features": physics_features,
            "n_removed_multicollinear": n_removed,
            "Uncertainty_Mean": float(np.mean(ensemble_std)),
            "Uncertainty_Calibrator": calibrator_meta,
            "cv_folds": n_splits
        }

    # Save metrics summary
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    # Save uncertainty calibrators (if any)
    if 'calibrators' in locals():
        joblib.dump(calibrators, os.path.join(output_dir, "uncertainty_calibrators.joblib"))
        
    print(f"\nTraining complete. Models saved to {output_dir}")
    
    # Save feature names to ensure consistent inference later
    feature_meta = {
        "feature_names": list(X.columns),
        "cat_features": cat_features
    }
    joblib.dump(feature_meta, os.path.join(output_dir, "feature_meta.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forward Models for Adsorbent Synthesis")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="artifacts/forward_models", help="Directory to save models")
    parser.add_argument("--iterations", type=int, default=1000, help="CatBoost iterations")
    parser.add_argument("--no-feature-selection", action="store_true", help="Disable feature selection, use all features")
    
    args = parser.parse_args()
    
    train_forward_models(args.data, args.output, iterations=args.iterations)
