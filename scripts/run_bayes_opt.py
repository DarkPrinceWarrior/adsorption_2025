#!/usr/bin/env python3
"""
Bayesian Optimization Engine for Adsorbent Inverse Design.

Stage 3 & 4: "Navigator" + "Inference"
User Constraints -> Optuna (Search) -> Forward Model (Simulator) -> Top Recipes
"""

import argparse
import os
import sys
import json
import joblib
import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from catboost import CatBoostRegressor

# Add src to path to import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.data_processing import load_dataset, build_lookup_tables
from adsorb_synthesis.constants import (
    FORWARD_MODEL_INPUTS,
    FORWARD_MODEL_TARGETS,
    FORWARD_MODEL_AUGMENTED_FEATURES
)

# Suppress Optuna logging to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AdsorbentOptimizer:
    def __init__(self, 
                 models_dir: str, 
                 data_path: str, 
                 n_trials: int = 200):
        
        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
        
        # Load Models
        self.models = self._load_models()
        
        # Load Reference Data & Lookups
        print(f"Loading reference data from {data_path}...")
        self.df_ref = load_dataset(data_path)
        self.lookup_tables = build_lookup_tables(self.df_ref)
        
        # Define Search Space based on available data
        self.search_space = self._define_search_space()
        
    def _load_models(self) -> Dict[str, CatBoostRegressor]:
        models = {}
        for target in FORWARD_MODEL_TARGETS:
            # Filename convention from training script
            filename = f"catboost_{target.replace('/', '_').replace(' ', '_')}.cbm"
            path = os.path.join(self.models_dir, filename)
            
            if os.path.exists(path):
                model = CatBoostRegressor()
                model.load_model(path)
                models[target] = model
            else:
                print(f"Warning: Model for {target} not found at {path}")
        
        if not models:
            raise RuntimeError("No models found! Please run train_forward_model.py first.")
            
        return models

    def _define_search_space(self) -> Dict:
        """Extract unique categories and ranges from reference data."""
        return {
            "metals": self.df_ref['Металл'].unique().tolist(),
            "ligands": self.df_ref['Лиганд'].unique().tolist(),
            "solvents": self.df_ref['Растворитель'].unique().tolist(),
            
            # Ranges (min, max) for continuous vars
            "m_salt_range": (self.df_ref['m (соли), г'].min(), self.df_ref['m (соли), г'].max()),
            "m_acid_range": (self.df_ref['m(кис-ты), г'].min(), self.df_ref['m(кис-ты), г'].max()),
            "v_solv_range": (self.df_ref['Vсин. (р-ля), мл'].min(), self.df_ref['Vсин. (р-ля), мл'].max()),
            "t_syn_range": (80, 220), # Reasonable synthesis bounds
            "t_dry_range": (25, 150),
            "t_act_range": (100, 400),
        }

    def _objective(self, trial: optuna.Trial, targets: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        The core function optimized by Optuna.
        1. Suggest a recipe.
        2. Predict properties using Forward Models.
        3. Calculate Loss (difference between predicted and desired properties).
        """
        
        # --- 1. Sample Recipe (X) ---
        metal = trial.suggest_categorical("Металл", self.search_space["metals"])
        ligand = trial.suggest_categorical("Лиганд", self.search_space["ligands"])
        solvent = trial.suggest_categorical("Растворитель", self.search_space["solvents"])
        
        m_salt = trial.suggest_float("m (соли), г", *self.search_space["m_salt_range"], log=True)
        m_acid = trial.suggest_float("m(кис-ты), г", *self.search_space["m_acid_range"], log=True)
        v_solv = trial.suggest_float("Vсин. (р-ля), мл", *self.search_space["v_solv_range"], step=5.0)
        
        t_syn = trial.suggest_int("Т.син., °С", *self.search_space["t_syn_range"], step=5)
        t_dry = trial.suggest_int("Т суш., °С", *self.search_space["t_dry_range"], step=5)
        t_act = trial.suggest_int("Tрег, ᵒС", *self.search_space["t_act_range"], step=5)
        
        # --- 2. Prepare Input Vector for Model ---
        # We need to construct a DataFrame row similar to what the model was trained on
        
        input_data = {
            "Металл": metal,
            "Лиганд": ligand,
            "Растворитель": solvent,
            "m (соли), г": m_salt,
            "m(кис-ты), г": m_acid,
            "Vсин. (р-ля), мл": v_solv,
            "Т.син., °С": t_syn,
            "Т суш., °С": t_dry,
            "Tрег, ᵒС": t_act,
            
            # Log transforms (Feature Engineering mirror)
            "log_m (соли), г": np.log1p(m_salt),
            "log_m(кис-ты), г": np.log1p(m_acid),
            "log_Vсин. (р-ля), мл": np.log1p(v_solv)
        }
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Enrich with Descriptors (Join Tables)
        # We manually simulate the merge behavior
        metal_desc = self.lookup_tables.metal.loc[metal]
        ligand_desc = self.lookup_tables.ligand.loc[ligand]
        solvent_desc = self.lookup_tables.solvent.loc[solvent]
        
        for idx, val in metal_desc.items():
            df_input[idx] = val
        for idx, val in ligand_desc.items():
            df_input[idx] = val
        for idx, val in solvent_desc.items():
            df_input[idx] = val

        # Ensure column order matches training (if possible, CatBoost is usually robust by name)
        # But let's be safe. We assume the model handles column order by name.
        
        # --- 3. Predict Properties ---
        loss = 0.0
        predictions = {}
        
        for target_name, target_val in targets.items():
            if target_name in self.models:
                pred = self.models[target_name].predict(df_input)[0]
                predictions[target_name] = pred
                
                # Normalized MSE Loss component
                # (pred - target)^2 / target^2  <- to make it relative percentage error
                if target_val != 0:
                    term = weights.get(target_name, 1.0) * ((pred - target_val) / target_val) ** 2
                else:
                    term = weights.get(target_name, 1.0) * (pred - target_val) ** 2
                    
                loss += term
        
        # Store predictions in trial user attrs for later retrieval
        for k, v in predictions.items():
            trial.set_user_attr(k, float(v))
            
        return loss

    def optimize(self, targets: Dict[str, float], weights: Dict[str, float] = None) -> pd.DataFrame:
        if weights is None:
            weights = {k: 1.0 for k in targets.keys()}
            
        print(f"\nStarting optimization for targets: {targets}")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: self._objective(t, targets, weights), n_trials=self.n_trials)
        
        print(f"Optimization finished. Best Loss: {study.best_value:.6f}")
        
        # Retrieve Top-N results
        trials = sorted(study.trials, key=lambda t: t.value)[:10] # Top 10
        
        results = []
        for t in trials:
            row = t.params.copy()
            row['Loss'] = t.value
            # Add predicted properties
            for k, v in t.user_attrs.items():
                row[f"Pred_{k}"] = v
            results.append(row)
            
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Inverse Design Inference via Bayesian Optimization")
    
    # Target arguments
    parser.add_argument("--W0", type=float, help="Target W0 (cm3/g)")
    parser.add_argument("--E0", type=float, help="Target E0 (kJ/mol)")
    parser.add_argument("--SBET", type=float, help="Target S_BET (m2/g)")
    parser.add_argument("--x0", type=float, help="Target x0 (nm)")
    
    parser.add_argument("--trials", type=int, default=200, help="Number of optimization trials")
    parser.add_argument("--output", type=str, default="predictions_bo.csv", help="Output CSV file")
    parser.add_argument("--models", type=str, default="artifacts/forward_models", help="Path to trained models")
    
    args = parser.parse_args()
    
    # Construct target dict
    targets = {}
    if args.W0: targets['W0, см3/г'] = args.W0
    if args.E0: targets['E0, кДж/моль'] = args.E0
    if args.SBET: targets['SБЭТ, м2/г'] = args.SBET
    if args.x0: targets['х0, нм'] = args.x0
    
    if not targets:
        print("Error: No targets specified! Use --W0, --E0, --SBET, or --x0.")
        return

    optimizer = AdsorbentOptimizer(
        models_dir=args.models,
        data_path="data/SEC_SYN_with_features.csv",
        n_trials=args.trials
    )
    
    df_results = optimizer.optimize(targets)
    
    print("\nTop 5 Recipes Found:")
    # Select columns to display cleanly
    display_cols = ['Loss'] + [c for c in df_results.columns if 'Pred_' in c] + ['Металл', 'Лиганд', 'Т.син., °С']
    print(df_results[display_cols].head(5).to_string(index=False))
    
    df_results.to_csv(args.output, index=False)
    print(f"\nFull results saved to {args.output}")

if __name__ == "__main__":
    main()
