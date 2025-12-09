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
from catboost import CatBoostRegressor, Pool

# Add src to path to import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.data_processing import (
    load_dataset,
    build_lookup_tables,
    add_salt_mass_features,
    add_physicochemical_descriptors,
)
from adsorb_synthesis.constants import (
    FORWARD_MODEL_INPUTS,
    FORWARD_MODEL_TARGETS,
    FORWARD_MODEL_AUGMENTED_FEATURES,
    FORWARD_MODEL_ENGINEERED_FEATURES,
    SOLVENT_BOILING_POINTS_C,
    STOICHIOMETRY_TARGETS,
    DEFAULT_STOICHIOMETRY_BOUNDS,
    E0_BOUNDS_KJ_MOL,
)
from adsorb_synthesis.physics_losses import compute_physics_penalty

# Suppress Optuna logging to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Risk aversion for BO (weight on uncertainty term)
LAMBDA_UNCERTAINTY = 0.5
PHYSICS_PENALTY_WEIGHT = 1.0

class AdsorbentOptimizer:
    def __init__(self, 
                 models_dir: str, 
                 data_path: str, 
                 n_trials: int = 200):
        
        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
        self.strict_validation = False
        
        # Load Models
        self.models = self._load_models()
        self.calibrators = self._load_calibrators()
        
        # Load Reference Data & Lookups
        print(f"Loading reference data from {data_path}...")
        validation_mode = "strict" if getattr(self, "strict_validation", False) else "warn"
        self.df_ref = load_dataset(data_path, validation_mode=validation_mode)
        self.lookup_tables = build_lookup_tables(self.df_ref)
        
        # Define Search Space based on available data
        self.search_space = self._define_search_space()
        
    def _load_models(self) -> Dict[str, List[CatBoostRegressor]]:
        models = {}
        self._feature_order = None  # Will be set from first loaded model
        
        for target in FORWARD_MODEL_TARGETS:
            target_models = []
            # Try to load ensemble members
            safe_target = target.replace('/', '_').replace(' ', '_')
            
            # Check for ensemble first
            ensemble_found = False
            for i in range(5): # Assume max 5 members
                path = os.path.join(self.models_dir, f"catboost_{safe_target}_ens{i}.cbm")
                if os.path.exists(path):
                    ensemble_found = True
                    model = CatBoostRegressor()
                    model.load_model(path)
                    target_models.append(model)
                    
                    # Store feature order from first model
                    if self._feature_order is None:
                        self._feature_order = model.feature_names_
                        print(f"Feature order loaded: {len(self._feature_order)} features")
            
            # Fallback to single model if no ensemble found
            if not ensemble_found:
                path = os.path.join(self.models_dir, f"catboost_{safe_target}.cbm")
                if os.path.exists(path):
                    print(f"Warning: Loading single model for {target} (No ensemble found)")
                    model = CatBoostRegressor()
                    model.load_model(path)
                    target_models.append(model)
                    
                    if self._feature_order is None:
                        self._feature_order = model.feature_names_
                else:
                    print(f"Warning: Model for {target} not found at {path}")
            else:
                print(f"Loaded ensemble of {len(target_models)} models for {target}")

            if target_models:
                models[target] = target_models
        
        if not models:
            raise RuntimeError("No models found! Please run train_forward_model.py first.")
            
        return models

    def _load_calibrators(self) -> Dict[str, object]:
        """Load uncertainty calibrators if available."""
        path = os.path.join(self.models_dir, "uncertainty_calibrators.joblib")
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: failed to load calibrators at {path}: {e}")
        return {}

    def _calibrate_sigma(self, target_name: str, raw_sigma: float) -> float:
        """Apply sigma calibration if calibrator is available."""
        calibrator = self.calibrators.get(target_name)
        if calibrator is None:
            return float(raw_sigma)
        try:
            if isinstance(calibrator, dict) and calibrator.get("type") == "scale":
                scale = calibrator.get("scale", 1.0)
                return float(raw_sigma * scale)
            if hasattr(calibrator, "predict"):
                return float(calibrator.predict([raw_sigma])[0])
        except Exception:
            pass
        return float(raw_sigma)

    def _define_search_space(self) -> Dict:
        """Extract unique categories and ranges from reference data."""
        return {
            "metals": self.df_ref['Металл'].unique().tolist(),
            "ligands": self.df_ref['Лиганд'].unique().tolist(),
            "solvents": self.df_ref['Растворитель'].unique().tolist(),
            
            # Ranges (min, max) for continuous vars
            "m_salt_range": (self.df_ref['m (соли), г'].min(), self.df_ref['m (соли), г'].max()),
            "m_acid_range": (self.df_ref['m(кис-ты), г'].min(), self.df_ref['m(кис-ты), г'].max()),
            # Round to nearest 5 to avoid Optuna step warnings
            "v_solv_range": (10.0, 180.0),  # Rounded from actual data range
            "t_syn_range": (80, 220), # Reasonable synthesis bounds
            "t_dry_range": (25, 150),
            "t_act_range": (100, 400),
        }

    @staticmethod
    def _get_boiling_point(solvent: str) -> Optional[float]:
        if solvent is None:
            return None
        key = str(solvent).strip()
        return SOLVENT_BOILING_POINTS_C.get(key) or SOLVENT_BOILING_POINTS_C.get(key.capitalize()) or SOLVENT_BOILING_POINTS_C.get(key.lower())

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

        # --- 2. SOFT CONSTRAINTS (Physical Feasibility) ---
        # Instead of flat 1e9 penalties, accumulate a smooth penalty term.
        constraint_penalty = 0.0
        penalty_reasons: List[Tuple[str, float]] = []

        def add_penalty(amount: float, reason: str) -> None:
            nonlocal constraint_penalty
            penalty = float(max(amount, 0.0))
            if penalty <= 0:
                return
            constraint_penalty += penalty
            if len(penalty_reasons) < 5:  # keep a short trace for debugging
                penalty_reasons.append((reason, penalty))

        def relative_violation(value: float, lower: Optional[float], upper: Optional[float], eps: float = 1e-8) -> float:
            if lower is not None and value < lower:
                return (lower - value) / max(abs(lower), eps)
            if upper is not None and value > upper:
                return (value - upper) / max(abs(upper), eps)
            return 0.0

        # 2.1. Temperature constraints (soft)
        dry_excess = max(0.0, t_dry - (t_syn + 20))
        add_penalty(dry_excess * 2.0, "dry_above_synthesis")

        act_shortfall = max(0.0, t_dry - t_act)
        add_penalty(act_shortfall * 3.0, "activation_below_dry")

        # 2.1b. Solvent boiling point constraint (soft)
        bp = self._get_boiling_point(solvent)
        if bp is not None:
            boil_over = max(0.0, t_syn - bp)
            add_penalty(boil_over * 5.0, "syn_above_boiling")
        
        # 2.2. Stoichiometry constraints
        # Get molar masses from lookup tables
        try:
            metal_desc = self.lookup_tables.metal.loc[metal]
            ligand_desc = self.lookup_tables.ligand.loc[ligand]
            solvent_desc = self.lookup_tables.solvent.loc[solvent]
        except KeyError:
            add_penalty(5_000.0, "lookup_missing")
            trial.report(constraint_penalty, step=0)
            raise optuna.TrialPruned("Missing lookup entry for metal/ligand/solvent")
        
        # Handle case where lookup returns DataFrame (multiple rows) vs Series (single row)
        if isinstance(metal_desc, pd.DataFrame):
            metal_desc = metal_desc.iloc[0]
        if isinstance(ligand_desc, pd.DataFrame):
            ligand_desc = ligand_desc.iloc[0]
        if isinstance(solvent_desc, pd.DataFrame):
            solvent_desc = solvent_desc.iloc[0]
        
        mw_salt = metal_desc.get('Молярка_соли', np.nan)
        mw_acid = ligand_desc.get('Молярка_кислоты', np.nan)
        
        # Ensure we have scalar values
        if hasattr(mw_salt, 'item'):
            mw_salt = mw_salt.item()
        if hasattr(mw_acid, 'item'):
            mw_acid = mw_acid.item()
        
        if pd.isna(mw_salt) or pd.isna(mw_acid) or mw_salt == 0 or mw_acid == 0:
            add_penalty(5_000.0, "missing_molar_mass")
            trial.report(constraint_penalty, step=0)
            raise optuna.TrialPruned("Missing molar mass data")
        
        # Calculate moles
        n_salt = m_salt / mw_salt
        n_acid = m_acid / mw_acid
        
        # Avoid division by zero
        if n_acid == 0:
            add_penalty(2_000.0, "zero_acid_moles")
            n_acid = 1e-6
        
        n_ratio = n_salt / n_acid
        # Target stoichiometry by metal-ligand pair if available
        stoich_spec = STOICHIOMETRY_TARGETS.get((metal, ligand))
        if stoich_spec:
            target_ratio = stoich_spec["ratio"]
            tol = stoich_spec.get("tolerance", 0.1)
            lower = target_ratio * (1 - tol)
            upper = target_ratio * (1 + tol)
            violation = relative_violation(n_ratio, lower, upper)
            add_penalty(500.0 * violation * violation, "stoichiometry")
        else:
            # Fallback broad bounds
            lo, hi = DEFAULT_STOICHIOMETRY_BOUNDS
            violation = relative_violation(n_ratio, lo, hi)
            add_penalty(250.0 * violation * violation, "stoichiometry_fallback")
        
        # 2.3. Concentration constraints
        if v_solv <= 0:
            add_penalty(1_000.0 + abs(v_solv) * 100.0, "non_positive_solvent_volume")
            v_solv = max(v_solv, 1e-3)  # guard log/ratio calculations
        
        # --- 3. FEATURE ENGINEERING (Must match training!) ---
        # Calculate all features that the model expects
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

            # From CSV (pre-computed in training data)
            "n_соли": n_salt,
            "n_кислоты": n_acid,
            "Vsyn_m": v_solv / m_salt if m_salt != 0 else 0,
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

        # Reuse training feature engineering to avoid train/inference drift
        add_salt_mass_features(df_input, inplace=True)
        add_physicochemical_descriptors(df_input, inplace=True)

        # Interaction feature (categorical, used by CatBoost)
        df_input["Metal_Ligand_Combo"] = df_input["Металл"].astype(str) + "_" + df_input["Лиганд"].astype(str)

        # Temperature-derived features not handled by helper
        t_range = t_act - t_syn
        t_activation = t_act - 100.0
        t_range_denom = t_range if t_range != 0 else 1e-9
        t_dry_norm = (t_dry - t_syn) / t_range_denom
        df_input["T_range"] = t_range
        df_input["T_activation"] = t_activation
        df_input["T_dry_norm"] = t_dry_norm
        
        # Ensure categorical features have correct dtype for CatBoost
        cat_features = ['Металл', 'Лиганд', 'Растворитель', 'Metal_Ligand_Combo']
        for col in cat_features:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
        
        # --- 4. Predict Properties with Uncertainty ---
        loss = 0.0
        predictions = {}
        uncertainties = {}
        ei_terms = []
        pi_terms = []
        eps = 1e-8
        
        for target_name, target_val in targets.items():
            if target_name in self.models:
                try:
                    ensemble = self.models[target_name]
                    # Get expected features from the first model in the ensemble
                    model_features = ensemble[0].feature_names_
                    
                    # Check if we have all needed features
                    missing = set(model_features) - set(df_input.columns)
                    if missing:
                        if trial.number == 0:  # Only print once
                            print(f"Missing features for {target_name}: {missing}")
                        add_penalty(10_000.0 * len(missing), "missing_features")
                        trial.report(loss + constraint_penalty, step=1)
                        raise optuna.TrialPruned(f"Missing features for {target_name}")
                        
                    # Prepare input slice for this specific model
                    df_slice = df_input[model_features]
                    
                    # Prepare Pool just for this target
                    # Identify present categorical features
                    known_cats = ['Металл', 'Лиганд', 'Растворитель', 'Metal_Ligand_Combo']
                    present_cats = [c for c in df_slice.columns if c in known_cats]
                    
                    predict_pool = Pool(df_slice, cat_features=present_cats)
                    
                    # Predict using Pool
                    preds = [model.predict(predict_pool)[0] for model in ensemble]
                    
                    mean_pred = np.mean(preds)
                    std_pred = np.std(preds)
                    
                except Exception as e:
                    # Feature mismatch or other error
                    print(f"Prediction error for {target_name}: {e}")
                    add_penalty(20_000.0, f"prediction_error_{target_name}")
                    trial.report(loss + constraint_penalty, step=1)
                    raise optuna.TrialPruned(f"Prediction failed for {target_name}")
                
                predictions[target_name] = mean_pred
                calibrated_sigma = self._calibrate_sigma(target_name, std_pred)
                uncertainties[target_name] = calibrated_sigma
                
                # Scale by target magnitude (to balance units) if provided
                scale = abs(target_val) if target_val != 0 else max(abs(mean_pred), 1.0)
                err_term = ((mean_pred - target_val) / scale) ** 2
                sigma_term = (calibrated_sigma / scale) ** 2

                # EI / PI components (maximize improvement -> minimize negative EI/PI)
                z = (abs(target_val - mean_pred) + eps) / (calibrated_sigma + eps)
                from math import erf, sqrt, exp, pi
                cdf = 0.5 * (1 + erf(-z / sqrt(2)))  # P(improvement)
                pdf = (1 / sqrt(2 * pi)) * exp(-0.5 * z * z)
                ei = (abs(target_val - mean_pred)) * cdf + calibrated_sigma * pdf
                pi = cdf
                ei_terms.append(ei)
                pi_terms.append(pi)

                term = weights.get(target_name, 1.0) * (err_term + LAMBDA_UNCERTAINTY * sigma_term)
                loss += term

        # Combine EI/PI into acquisition penalty (encourage improvement & low sigma)
        if ei_terms:
            # Use mean negative EI and PI as additional terms
            loss += LAMBDA_UNCERTAINTY * (np.mean(ei_terms) + np.mean(pi_terms))

        # Physics penalty using available predictions
        if predictions:
            phys_row = {}
            for col in ["W0, см3/г", "E0, кДж/моль", "Ws, см3/г", "E, кДж/моль"]:
                if col in predictions:
                    phys_row[col] = predictions[col]
            phys_df = pd.DataFrame([phys_row]) if phys_row else None
            if phys_df is not None:
                phys_penalty = compute_physics_penalty(phys_df).iloc[0]
                loss += PHYSICS_PENALTY_WEIGHT * phys_penalty
        # Additional physics-based checks on predictions (where available)
        # E0 bounds
        e0_pred = predictions.get("E0, кДж/моль")
        if e0_pred is not None:
            lo_e0, hi_e0 = E0_BOUNDS_KJ_MOL
            violation = relative_violation(e0_pred, lo_e0, hi_e0)
            add_penalty(300.0 * violation, "E0_bounds")
        
        # Physics consistency checks on predictions (hard rejection)
        w0_pred = predictions.get("W0, см3/г")
        ws_pred = predictions.get("Ws, см3/г")
        if w0_pred is not None and ws_pred is not None and ws_pred < w0_pred:
            add_penalty(
                200.0 * (w0_pred - ws_pred) / max(abs(w0_pred), 1e-6),
                "Ws_less_than_W0"
            )

        e_pred = predictions.get("E, кДж/моль")
        if e_pred is not None and e0_pred is not None and e0_pred != 0:
            ratio = e_pred / e0_pred
            ratio_violation = relative_violation(ratio, 0.2, 1.0)
            add_penalty(200.0 * ratio_violation, "E_over_E0_ratio")

        # --- 5. Store predictions for later retrieval ---
        for k, v in predictions.items():
            trial.set_user_attr(k, float(v))
        for k, v in uncertainties.items():
            trial.set_user_attr(f"Uncertainty_{k}", float(v))

        trial.set_user_attr("ConstraintPenalty", float(constraint_penalty))
        if penalty_reasons:
            trial.set_user_attr(
                "PenaltyReasons",
                [f"{reason}:{value:.3f}" for reason, value in penalty_reasons]
            )

        total_loss = loss + constraint_penalty
        trial.report(total_loss, step=2)
        return total_loss

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
    parser.add_argument("--strict-validation", action="store_true", help="Use strict validation (errors on invalid rows)")
    
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
        data_path="data/SEC_SYN_with_features_enriched.csv",
        n_trials=args.trials
    )
    optimizer.strict_validation = args.strict_validation
    
    df_results = optimizer.optimize(targets)
    
    print("\nTop 5 Recipes Found:")
    # Select columns to display cleanly
    pred_cols = [c for c in df_results.columns if 'Pred_' in c]
    unc_cols = [c for c in df_results.columns if 'Uncertainty_' in c]
    
    # Interleave Pred and Uncertainty for readability
    val_cols = []
    for p in pred_cols:
        val_cols.append(p)
        # Find matching uncertainty column
        base_name = p.replace('Pred_', '')
        u_name = f"Uncertainty_{base_name}"
        if u_name in df_results.columns:
            val_cols.append(u_name)
            
    display_cols = ['Loss'] + val_cols + ['Металл', 'Лиганд', 'Т.син., °С']
    print(df_results[display_cols].head(5).to_string(index=False))
    
    df_results.to_csv(args.output, index=False)
    print(f"\nFull results saved to {args.output}")

if __name__ == "__main__":
    main()
