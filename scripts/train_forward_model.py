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
from sklearn.model_selection import train_test_split

from adsorb_synthesis.data_processing import load_dataset, build_lookup_tables, prepare_forward_dataset
from adsorb_synthesis.constants import RANDOM_SEED, FORWARD_MODEL_TARGETS

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
    # These are columns with object/category dtype in X
    cat_features = [col for col in X.columns if X[col].dtype.name in ['object', 'category']]
    print(f"Categorical features found: {cat_features}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    
    models = {}
    metrics = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train a separate ENSEMBLE of models for each target property
    n_ensemble = 5
    
    for target in FORWARD_MODEL_TARGETS:
        print(f"\n=== Training ENSEMBLE for target: {target} ===")
        
        if target not in y_train.columns:
            print(f"Skipping {target}: not found in targets.")
            continue
            
        y_train_target = y_train[target]
        y_test_target = y_test[target]
        
        # --- Feature Selection Step ---
        print(f"  Performing Feature Selection for {target}...")
        
        # Train a temporary model to find best features
        fs_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_features
        )
        
        fs_model.fit(X_train, y_train_target, verbose=False)
        
        # Get feature importance
        importance = fs_model.get_feature_importance(type='PredictionValuesChange')
        feature_imp = pd.DataFrame({'feature': X_train.columns, 'importance': importance})
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        # Select top 15 features
        selected_features = feature_imp['feature'].head(15).tolist()
        print(f"  Selected {len(selected_features)} features: {selected_features}")
        
        # Filter datasets to selected features only
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        # Update cat_features for the selected subset
        cat_features_sel = [c for c in selected_features if c in cat_features]

        model_paths = []
        ensemble_preds_test = []
        ensemble_preds_train = []
        
        for i in range(n_ensemble):
            seed = RANDOM_SEED + i
            print(f"  Training member {i+1}/{n_ensemble} (seed={seed})...")
            
            # Initialize CatBoost with balanced regularization
            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=0.05, # Increased from 0.03
                depth=6,            # Increased from 4
                l2_leaf_reg=1.0,    # Decreased from 3.0 (less regularization)
                min_data_in_leaf=1, # Allow smaller leaves
                subsample=0.8,
                colsample_bylevel=0.8,
                loss_function='RMSE',
                random_seed=seed,
                verbose=False, # Less verbose
                allow_writing_files=False,
                cat_features=cat_features_sel # Use updated cat_features
            )
            
            # Fit model on SELECTED features
            model.fit(
                X_train_sel, y_train_target,
                eval_set=(X_test_sel, y_test_target),
                early_stopping_rounds=100,
                use_best_model=True
            )
            
            # Save individual model
            safe_target = target.replace('/', '_').replace(' ', '_')
            model_path = os.path.join(output_dir, f"catboost_{safe_target}_ens{i}.cbm")
            model.save_model(model_path)
            model_paths.append(model_path)
            
            # Collect predictions
            ensemble_preds_test.append(model.predict(X_test_sel))
            ensemble_preds_train.append(model.predict(X_train_sel))

        # Aggregate predictions (Mean of Ensemble)
        mean_preds_test = np.mean(ensemble_preds_test, axis=0)
        std_preds_test = np.std(ensemble_preds_test, axis=0)
        mean_preds_train = np.mean(ensemble_preds_train, axis=0)
        
        r2_test = r2_score(y_test_target, mean_preds_test)
        r2_train = r2_score(y_train_target, mean_preds_train)
        rmse_test = np.sqrt(mean_squared_error(y_test_target, mean_preds_test))
        mae_test = mean_absolute_error(y_test_target, mean_preds_test)
        
        print(f"Ensemble Results for {target}:")
        print(f"  R2 (Test):  {r2_test:.4f}")
        print(f"  R2 (Train): {r2_train:.4f}")
        print(f"  RMSE:       {rmse_test:.4f}")
        print(f"  Avg Uncertainty (StdDev): {np.mean(std_preds_test):.4f}")
        
        models[target] = model_paths # Store list of paths
        
        metrics[target] = {
            "R2_test": r2_test,
            "R2_train": r2_train,
            "RMSE": rmse_test,
            "MAE": mae_test,
            "Uncertainty_Mean": float(np.mean(std_preds_test))
        }

    # Save metrics summary
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    print(f"\nTraining complete. Models saved to {output_dir}")
    
    # Save feature names to ensure consistent inference later
    feature_meta = {
        "feature_names": list(X.columns),
        "cat_features": cat_features
    }
    joblib.dump(feature_meta, os.path.join(output_dir, "feature_meta.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forward Models for Adsorbent Synthesis")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="artifacts/forward_models", help="Directory to save models")
    parser.add_argument("--iterations", type=int, default=1000, help="CatBoost iterations")
    
    args = parser.parse_args()
    
    train_forward_models(args.data, args.output, iterations=args.iterations)
