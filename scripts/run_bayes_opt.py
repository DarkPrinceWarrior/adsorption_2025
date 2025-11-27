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
    FORWARD_MODEL_AUGMENTED_FEATURES,
    FORWARD_MODEL_ENGINEERED_FEATURES,
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
        self._feature_order = None  # Will be set from first loaded model
        
        for target in FORWARD_MODEL_TARGETS:
            # Filename convention from training script
            filename = f"catboost_{target.replace('/', '_').replace(' ', '_')}.cbm"
            path = os.path.join(self.models_dir, filename)
            
            if os.path.exists(path):
                model = CatBoostRegressor()
                model.load_model(path)
                models[target] = model
                
                # Store feature order from first model (all models have same features)
                if self._feature_order is None:
                    self._feature_order = model.feature_names_
                    print(f"Feature order loaded: {len(self._feature_order)} features")
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
        2. Apply hard physical constraints.
        3. Calculate engineered features (must match training!).
        4. Predict properties using Forward Models.
        5. Calculate Loss (difference between predicted and desired properties).
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
        
        # --- 2. HARD CONSTRAINTS (Physical Feasibility) ---
        # Reject physically impossible recipes before expensive model inference
        
        # 2.1. Temperature constraints
        # Drying should not be hotter than synthesis (usually)
        if t_dry > t_syn + 20:
            return 1e9
        
        # Activation must be higher than drying
        if t_act < t_dry:
            return 1e9
        
        # 2.2. Stoichiometry constraints
        # Get molar masses from lookup tables
        metal_desc = self.lookup_tables.metal.loc[metal]
        ligand_desc = self.lookup_tables.ligand.loc[ligand]
        solvent_desc = self.lookup_tables.solvent.loc[solvent]
        
        mw_salt = metal_desc.get('Молярка_соли', np.nan)
        mw_acid = ligand_desc.get('Молярка_кислоты', np.nan)
        
        if pd.isna(mw_salt) or pd.isna(mw_acid) or mw_salt == 0 or mw_acid == 0:
            return 1e9  # Missing data, cannot proceed
        
        # Calculate moles
        n_salt = m_salt / mw_salt
        n_acid = m_acid / mw_acid
        
        # Avoid division by zero
        if n_acid == 0:
            return 1e9
        
        n_ratio = n_salt / n_acid
        
        # Reject extreme stoichiometry (e.g., 1:100 ratio is unrealistic)
        if n_ratio < 0.1 or n_ratio > 10.0:
            return 1e9
        
        # 2.3. Concentration constraints
        if v_solv <= 0:
            return 1e9
        
        c_metal = m_salt / v_solv
        c_ligand = m_acid / v_solv
        
        # --- 3. FEATURE ENGINEERING (Must match training!) ---
        # Calculate all 45 features that the model expects
        
        # Temperature features
        t_range = t_act - t_syn
        t_activation = t_act - 100.0
        t_range_denom = t_range if t_range != 0 else 1e-9
        t_dry_norm = (t_dry - t_syn) / t_range_denom
        
        input_data = {
            # Base categorical
            "Металл": metal,
            "Лиганд": ligand,
            "Растворитель": solvent,
            
            # Base numeric
            "m (соли), г": m_salt,
            "m(кис-ты), г": m_acid,
            "Vсин. (р-ля), мл": v_solv,
            "Т.син., °С": t_syn,
            "Т суш., °С": t_dry,
            "Tрег, ᵒС": t_act,
            
            # Log transforms
            "log_m (соли), г": np.log1p(m_salt),
            "log_m(кис-ты), г": np.log1p(m_acid),
            "log_Vсин. (р-ля), мл": np.log1p(v_solv),
            
            # Stoichiometry (from FORWARD_MODEL_ENGINEERED_FEATURES)
            "R_molar": n_ratio,
            "R_mass": m_salt / m_acid if m_acid != 0 else 0,
            
            # Concentrations
            "C_metal": c_metal,
            "C_ligand": c_ligand,
            "log_C_metal": np.log1p(c_metal),
            "log_C_ligand": np.log1p(c_ligand),
            
            # From CSV (pre-computed in training data)
            "n_соли": n_salt,
            "n_кислоты": n_acid,
            "Vsyn_m": v_solv / m_salt if m_salt != 0 else 0,
            
            # Temperature features
            "T_range": t_range,
            "T_activation": t_activation,
            "T_dry_norm": t_dry_norm,
            
            # Interaction feature
            "Metal_Ligand_Combo": f"{metal}_{ligand}",
        }
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Enrich with Descriptors from lookup tables
        for idx, val in metal_desc.items():
            if idx not in df_input.columns:
                df_input[idx] = val
        for idx, val in ligand_desc.items():
            if idx not in df_input.columns:
                df_input[idx] = val
        for idx, val in solvent_desc.items():
            if idx not in df_input.columns:
                df_input[idx] = val
        
        # Ensure categorical features have correct dtype for CatBoost
        cat_features = ['Металл', 'Лиганд', 'Растворитель', 'Metal_Ligand_Combo']
        for col in cat_features:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
        
        # Reorder columns to match training order (CatBoost is sensitive to this)
        # Get expected column order from first model
        if hasattr(self, '_feature_order') and self._feature_order is not None:
            # Reorder df_input to match training feature order
            missing_cols = set(self._feature_order) - set(df_input.columns)
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                return 1e9
            df_input = df_input[self._feature_order]
        
        # --- 4. Predict Properties ---
        loss = 0.0
        predictions = {}
        
        for target_name, target_val in targets.items():
            if target_name in self.models:
                try:
                    pred = self.models[target_name].predict(df_input)[0]
                except Exception as e:
                    # Feature mismatch or other error
                    print(f"Prediction error for {target_name}: {e}")
                    return 1e9
                
                predictions[target_name] = pred
                
                # Normalized MSE Loss component
                # (pred - target)^2 / target^2  <- relative percentage error
                if target_val != 0:
                    term = weights.get(target_name, 1.0) * ((pred - target_val) / target_val) ** 2
                else:
                    term = weights.get(target_name, 1.0) * (pred - target_val) ** 2
                    
                loss += term
        
        # --- 5. Store predictions for later retrieval ---
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
    parser.add_argument("--W0", type=float, help="Target W0 micropore volume (cm3/g)")
    parser.add_argument("--E0", type=float, help="Target E0 characteristic energy (kJ/mol)")
    parser.add_argument("--SBET", type=float, help="Target S_BET surface area (m2/g)")
    parser.add_argument("--x0", type=float, help="Target x0 pore half-width (nm)")
    parser.add_argument("--Sme", type=float, help="Target Sme mesopore surface area (m2/g)")
    
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
    if args.Sme: targets['Sme, м2/г'] = args.Sme
    
    if not targets:
        print("Error: No targets specified! Use --W0, --E0, --SBET, --x0, or --Sme.")
        return

    optimizer = AdsorbentOptimizer(
        models_dir=args.models,
        data_path="data/SEC_SYN_with_features_DMFA_only.csv",
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
