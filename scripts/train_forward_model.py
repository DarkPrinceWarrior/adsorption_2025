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

from adsorb_synthesis.data_processing import load_dataset, build_lookup_tables, prepare_forward_dataset
from adsorb_synthesis.constants import RANDOM_SEED, FORWARD_MODEL_TARGETS, RARE_METALS_THRESHOLD
from adsorb_synthesis.config import CATBOOST_CONFIG, FORWARD_MODEL_CONFIG
from adsorb_synthesis.feature_selection import (
    select_features_advanced,
    get_curated_features
)
from adsorb_synthesis.physics_losses import compute_physics_penalty


def train_forward_models(
    data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    iterations: int = 1000,
    validation_mode: str = "warn"
):
    print(f"Loading dataset from {data_path}...")
    # Load raw data with standard enrichment
    df_raw = load_dataset(data_path, validation_mode=validation_mode)
    
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
    
    # CV ensemble parameters (from config)
    n_splits = FORWARD_MODEL_CONFIG.n_ensemble_splits
    PHYSICS_PENALTY_WEIGHT = FORWARD_MODEL_CONFIG.physics_penalty_weight
    
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
        
        # Create CV splits BEFORE feature selection to avoid data leakage.
        # Feature selection will only see fold-0's training data.
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_splits = list(skf.split(X, strat_key))
        fs_train_idx = cv_splits[0][0]  # fold-0 train indices for feature selection

        # Feature Selection (on fold-0 train data only — no leakage)
        print(f"  Advanced Feature Selection for {target} (on fold-0 train, n={len(fs_train_idx)})...")
        keep_features, drop_features = get_curated_features()
        available_drop = [f for f in drop_features if f in X.columns]
        numeric_cols = [c for c in X.columns if c not in cat_features]
        curated_numeric = [c for c in numeric_cols if c not in available_drop]
        selected_features, selection_report = select_features_advanced(
            X.iloc[fs_train_idx][cat_features + curated_numeric],
            y_target.iloc[fs_train_idx],
            categorical_cols=cat_features,
            corr_threshold=FORWARD_MODEL_CONFIG.feature_selection_corr_threshold,
            vif_threshold=FORWARD_MODEL_CONFIG.feature_selection_vif_threshold,
            max_features=FORWARD_MODEL_CONFIG.feature_selection_max_features,
            verbose=False
        )
        n_removed = len(selection_report['removed_correlation']) + len(selection_report['removed_vif'])
        print(f"    Removed {n_removed} multicollinear features")
        print(f"    Selected {len(selected_features)} features: {selected_features[:5]}...")
        cat_features_sel = [c for c in selected_features if c in cat_features]

        # =====================================================================
        # Phase 1: CV loop — honest OOF metrics (fold models are temporary)
        # =====================================================================
        oof_preds = np.full(len(X), np.nan, dtype=float)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            seed = RANDOM_SEED + fold_idx
            print(f"  CV Fold {fold_idx+1}/{n_splits} (seed={seed})...")
            X_train_sel = X.iloc[train_idx][selected_features]
            X_val_sel = X.iloc[val_idx][selected_features]
            y_train_target = y_target.iloc[train_idx]
            y_val_target = y_target.iloc[val_idx]
            w_train = sample_weights_full[train_idx]

            fold_model = CatBoostRegressor(
                **CATBOOST_CONFIG.to_params(random_state=seed),
                cat_features=cat_features_sel
            )
            fold_model.fit(
                X_train_sel, y_train_target,
                sample_weight=w_train,
                eval_set=(X_val_sel, y_val_target),
                early_stopping_rounds=FORWARD_MODEL_CONFIG.early_stopping_rounds,
                use_best_model=True
            )
            oof_preds[val_idx] = fold_model.predict(X_val_sel)

        r2_oof = r2_score(y_target, oof_preds)
        rmse_oof = np.sqrt(mean_squared_error(y_target, oof_preds))
        mae_oof = mean_absolute_error(y_target, oof_preds)
        print(f"  CV OOF R2: {r2_oof:.4f}, RMSE: {rmse_oof:.4f}, MAE: {mae_oof:.4f}")

        # =====================================================================
        # Phase 2: Production Deep Ensemble — N models on 100% data
        # =====================================================================
        n_members = FORWARD_MODEL_CONFIG.n_ensemble_members
        seed_step = FORWARD_MODEL_CONFIG.ensemble_seed_step
        print(f"  Training production ensemble ({n_members} members on full data)...")
        safe_target = target.replace('/', '_').replace(' ', '_')
        model_paths = []
        prod_models = []

        for m_idx in range(n_members):
            seed = RANDOM_SEED + (m_idx + 1) * seed_step
            model = CatBoostRegressor(
                **CATBOOST_CONFIG.to_params(random_state=seed),
                cat_features=cat_features_sel
            )
            model.fit(
                X[selected_features], y_target,
                sample_weight=sample_weights_full,
            )
            model_path = os.path.join(output_dir, f"catboost_{safe_target}_ens{m_idx}.cbm")
            model.save_model(model_path)
            model_paths.append(model_path)
            prod_models.append(model)

        # Production ensemble predictions (on training data — for diagnostics)
        prod_preds = np.stack([m.predict(X[selected_features]) for m in prod_models], axis=1)
        prod_mean = np.mean(prod_preds, axis=1)
        prod_sigma = np.std(prod_preds, axis=1)

        r2_prod = r2_score(y_target, prod_mean)
        rmse_prod = np.sqrt(mean_squared_error(y_target, prod_mean))
        mae_prod = mean_absolute_error(y_target, prod_mean)
        print(f"  Production Ensemble R2: {r2_prod:.4f}, RMSE: {rmse_prod:.4f}, MAE: {mae_prod:.4f}")
        print(f"  Avg Ensemble σ: {np.mean(prod_sigma):.4f}")

        models[target] = model_paths

        # =====================================================================
        # Phase 3: Conformal calibration from OOF residuals
        # =====================================================================
        alpha = FORWARD_MODEL_CONFIG.conformal_alpha
        oof_residuals = np.abs(y_target.values - oof_preds)
        # Normalized scores: honest residual / production sigma
        eps = 1e-8
        norm_scores = oof_residuals / (prod_sigma + eps)
        # Finite-sample corrected quantile (Vovk et al.)
        n_cal = len(norm_scores)
        q_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)
        conformal_q = float(np.quantile(norm_scores, q_level))

        # Marginal (non-normalized) quantile as fallback
        marginal_q = float(np.quantile(oof_residuals, q_level))

        print(f"  Conformal quantile (α={alpha}): q={conformal_q:.4f} "
              f"(marginal={marginal_q:.4f})")

        calibrators[target] = {
            "type": "conformal",
            "conformal_q": conformal_q,
            "marginal_q": marginal_q,
            "alpha": alpha,
            "n_calibration": n_cal,
        }

        # Save predictions (OOF + production)
        predictions_df = pd.DataFrame({
            'y_actual': y_target.values,
            'y_oof': oof_preds,
            'y_prod_mean': prod_mean,
            'y_prod_sigma': prod_sigma,
            'oof_residual': oof_residuals,
            'norm_score': norm_scores,
        })
        predictions_path = os.path.join(output_dir, f"predictions_{safe_target}.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  Saved predictions: {predictions_path}")

        physics_features = [f for f in selected_features if any(x in f for x in 
            ['metal_coord', 'ligand_3d', 'ligand_2d', 'Size_Ratio', 'Electronegativity_Diff', 'Jahn_Teller'])]

        metrics[target] = {
            "R2": r2_oof,
            "R2_oof": r2_oof,
            "R2_production": r2_prod,
            "RMSE": rmse_oof,
            "MAE": mae_oof,
            "RMSE_oof": rmse_oof,
            "MAE_oof": mae_oof,
            "RMSE_production": rmse_prod,
            "MAE_production": mae_prod,
            "selected_features": selected_features,
            "physics_features": physics_features,
            "n_removed_multicollinear": n_removed,
            "Ensemble_Sigma_Mean": float(np.mean(prod_sigma)),
            "Conformal_q": conformal_q,
            "Conformal_alpha": alpha,
            "cv_folds": n_splits,
            "ensemble_members": n_members,
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
    parser.add_argument("--validation-mode", type=str, default="warn", choices=["warn", "strict"], help="Validation mode for dataset loading")
    
    args = parser.parse_args()
    
    train_forward_models(args.data, args.output, iterations=args.iterations, validation_mode=args.validation_mode)
