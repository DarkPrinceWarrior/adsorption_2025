#!/usr/bin/env python3
"""
Multi-Objective Bayesian Optimization for Adsorbent Inverse Design.

Uses Optuna NSGA-II to optimise multiple adsorption targets simultaneously,
producing a Pareto front of non-dominated synthesis recipes.

Stage 3 & 4: "Navigator" + "Inference"
User Constraints -> Optuna NSGA-II -> Forward Model -> Pareto Recipes
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


def _relative_violation(value: float, lower: Optional[float],
                        upper: Optional[float], eps: float = 1e-8) -> float:
    if lower is not None and value < lower:
        return (lower - value) / max(abs(lower), eps)
    if upper is not None and value > upper:
        return (value - upper) / max(abs(upper), eps)
    return 0.0


class AdsorbentOptimizer:
    def __init__(self,
                 models_dir: str,
                 data_path: str,
                 n_trials: int = 200,
                 strict_validation: bool = False):

        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
        self.strict_validation = strict_validation

        # Load Models
        self.models = self._load_models()
        self.calibrators = self._load_calibrators()

        # Load Reference Data & Lookups
        print(f"Loading reference data from {data_path}...")
        validation_mode = "strict" if self.strict_validation else "warn"
        self.df_ref = load_dataset(data_path, validation_mode=validation_mode)
        self.lookup_tables = build_lookup_tables(self.df_ref)

        # Define Search Space based on available data
        self.search_space = self._define_search_space()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self) -> Dict[str, List[CatBoostRegressor]]:
        models = {}
        self._feature_order = None

        for target in FORWARD_MODEL_TARGETS:
            target_models = []
            safe_target = target.replace('/', '_').replace(' ', '_')

            ensemble_found = False
            for i in range(10):  # support up to 10 members
                path = os.path.join(self.models_dir,
                                    f"catboost_{safe_target}_ens{i}.cbm")
                if os.path.exists(path):
                    ensemble_found = True
                    model = CatBoostRegressor()
                    model.load_model(path)
                    target_models.append(model)
                    if self._feature_order is None:
                        self._feature_order = model.feature_names_
                        print(f"Feature order loaded: "
                              f"{len(self._feature_order)} features")

            if not ensemble_found:
                path = os.path.join(self.models_dir,
                                    f"catboost_{safe_target}.cbm")
                if os.path.exists(path):
                    print(f"Warning: single model for {target}")
                    model = CatBoostRegressor()
                    model.load_model(path)
                    target_models.append(model)
                    if self._feature_order is None:
                        self._feature_order = model.feature_names_
                else:
                    print(f"Warning: no model for {target}")
            else:
                print(f"Loaded {len(target_models)} ensemble members "
                      f"for {target}")

            if target_models:
                models[target] = target_models

        if not models:
            raise RuntimeError(
                "No models found! Run train_forward_model.py first.")
        return models

    def _load_calibrators(self) -> Dict[str, object]:
        path = os.path.join(self.models_dir,
                            "uncertainty_calibrators.joblib")
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: failed to load calibrators: {e}")
        return {}

    def _calibrate_sigma(self, target_name: str,
                         raw_sigma: float) -> float:
        calibrator = self.calibrators.get(target_name)
        if calibrator is None:
            return float(raw_sigma)
        try:
            if isinstance(calibrator, dict):
                cal_type = calibrator.get("type")
                if cal_type == "conformal":
                    q = calibrator.get("conformal_q", 1.0)
                    return float(raw_sigma * q)
                if cal_type == "scale":
                    return float(raw_sigma * calibrator.get("scale", 1.0))
            if hasattr(calibrator, "predict"):
                return float(calibrator.predict([raw_sigma])[0])
        except Exception:
            pass
        return float(raw_sigma)

    # ------------------------------------------------------------------
    # Search space
    # ------------------------------------------------------------------
    def _define_search_space(self) -> Dict:
        return {
            "metals": self.df_ref['Металл'].unique().tolist(),
            "ligands": self.df_ref['Лиганд'].unique().tolist(),
            "solvents": self.df_ref['Растворитель'].unique().tolist(),
            "m_salt_range": (self.df_ref['m (соли), г'].min(),
                             self.df_ref['m (соли), г'].max()),
            "m_acid_range": (self.df_ref['m(кис-ты), г'].min(),
                             self.df_ref['m(кис-ты), г'].max()),
            "v_solv_range": (10.0, 180.0),
            "t_syn_range": (80, 220),
            "t_dry_range": (25, 150),
            "t_act_range": (100, 400),
        }

    @staticmethod
    def _get_boiling_point(solvent: str) -> Optional[float]:
        if solvent is None:
            return None
        key = str(solvent).strip()
        return (SOLVENT_BOILING_POINTS_C.get(key)
                or SOLVENT_BOILING_POINTS_C.get(key.capitalize())
                or SOLVENT_BOILING_POINTS_C.get(key.lower()))

    # ------------------------------------------------------------------
    # Feature engineering (shared between objectives)
    # ------------------------------------------------------------------
    def _build_features(self, trial: optuna.Trial) -> Tuple[
            pd.DataFrame, float, List[Tuple[str, float]]]:
        """Sample recipe, apply constraints, compute features.

        Returns:
            df_input: Single-row DataFrame ready for prediction.
            constraint_penalty: Total constraint violation (0 = feasible).
            penalty_reasons: Short trace of violated constraints.
        """
        # --- 1. Sample Recipe ---
        metal = trial.suggest_categorical(
            "Металл", self.search_space["metals"])
        ligand = trial.suggest_categorical(
            "Лиганд", self.search_space["ligands"])
        solvent = trial.suggest_categorical(
            "Растворитель", self.search_space["solvents"])

        m_salt = trial.suggest_float(
            "m (соли), г", *self.search_space["m_salt_range"], log=True)
        m_acid = trial.suggest_float(
            "m(кис-ты), г", *self.search_space["m_acid_range"], log=True)
        v_solv = trial.suggest_float(
            "Vсин. (р-ля), мл", *self.search_space["v_solv_range"], step=5.0)

        t_syn = trial.suggest_int(
            "Т.син., °С", *self.search_space["t_syn_range"], step=5)
        t_dry = trial.suggest_int(
            "Т суш., °С", *self.search_space["t_dry_range"], step=5)
        t_act = trial.suggest_int(
            "Tрег, ᵒС", *self.search_space["t_act_range"], step=5)

        # --- 2. Soft Constraints ---
        constraint_penalty = 0.0
        penalty_reasons: List[Tuple[str, float]] = []

        def add_penalty(amount: float, reason: str) -> None:
            nonlocal constraint_penalty
            p = float(max(amount, 0.0))
            if p > 0:
                constraint_penalty += p
                if len(penalty_reasons) < 5:
                    penalty_reasons.append((reason, p))

        # Temperature
        add_penalty(max(0.0, t_dry - (t_syn + 20)) * 2.0,
                    "dry_above_synthesis")
        add_penalty(max(0.0, t_dry - t_act) * 3.0,
                    "activation_below_dry")
        bp = self._get_boiling_point(solvent)
        if bp is not None:
            add_penalty(max(0.0, t_syn - bp) * 5.0, "syn_above_boiling")

        # Lookups
        try:
            metal_desc = self.lookup_tables.metal.loc[metal]
            ligand_desc = self.lookup_tables.ligand.loc[ligand]
            solvent_desc = self.lookup_tables.solvent.loc[solvent]
        except KeyError:
            add_penalty(5_000.0, "lookup_missing")
            raise optuna.TrialPruned("Missing lookup")

        if isinstance(metal_desc, pd.DataFrame):
            metal_desc = metal_desc.iloc[0]
        if isinstance(ligand_desc, pd.DataFrame):
            ligand_desc = ligand_desc.iloc[0]
        if isinstance(solvent_desc, pd.DataFrame):
            solvent_desc = solvent_desc.iloc[0]

        mw_salt = metal_desc.get('Молярка_соли', np.nan)
        mw_acid = ligand_desc.get('Молярка_кислоты', np.nan)
        if hasattr(mw_salt, 'item'):
            mw_salt = mw_salt.item()
        if hasattr(mw_acid, 'item'):
            mw_acid = mw_acid.item()

        if pd.isna(mw_salt) or pd.isna(mw_acid) or mw_salt == 0 or mw_acid == 0:
            add_penalty(5_000.0, "missing_molar_mass")
            raise optuna.TrialPruned("Missing molar mass")

        n_salt = m_salt / mw_salt
        n_acid = m_acid / mw_acid
        if n_acid == 0:
            add_penalty(2_000.0, "zero_acid_moles")
            n_acid = 1e-6

        # Stoichiometry
        n_ratio = n_salt / n_acid
        stoich_spec = STOICHIOMETRY_TARGETS.get((metal, ligand))
        if stoich_spec:
            lo = stoich_spec["ratio"] * (1 - stoich_spec.get("tolerance", 0.1))
            hi = stoich_spec["ratio"] * (1 + stoich_spec.get("tolerance", 0.1))
            v = _relative_violation(n_ratio, lo, hi)
            add_penalty(500.0 * v * v, "stoichiometry")
        else:
            lo, hi = DEFAULT_STOICHIOMETRY_BOUNDS
            v = _relative_violation(n_ratio, lo, hi)
            add_penalty(250.0 * v * v, "stoichiometry_fallback")

        if v_solv <= 0:
            add_penalty(1_000.0 + abs(v_solv) * 100.0,
                        "non_positive_solvent_volume")
            v_solv = max(v_solv, 1e-3)

        # --- 3. Feature Engineering ---
        input_data = {
            "Металл": metal, "Лиганд": ligand, "Растворитель": solvent,
            "m (соли), г": m_salt, "m(кис-ты), г": m_acid,
            "Vсин. (р-ля), мл": v_solv,
            "Т.син., °С": t_syn, "Т суш., °С": t_dry, "Tрег, ᵒС": t_act,
            "log_m (соли), г": np.log1p(m_salt),
            "log_m(кис-ты), г": np.log1p(m_acid),
            "log_Vсин. (р-ля), мл": np.log1p(v_solv),
            "n_соли": n_salt, "n_кислоты": n_acid,
            "Vsyn_m": v_solv / m_salt if m_salt != 0 else 0,
        }

        df_input = pd.DataFrame([input_data])

        for idx, val in metal_desc.items():
            if idx not in df_input.columns:
                df_input[idx] = val
        for idx, val in ligand_desc.items():
            if idx not in df_input.columns:
                df_input[idx] = val
        for idx, val in solvent_desc.items():
            if idx not in df_input.columns:
                df_input[idx] = val

        add_salt_mass_features(df_input, inplace=True)
        add_physicochemical_descriptors(df_input, inplace=True)

        df_input["Metal_Ligand_Combo"] = (
            df_input["Металл"].astype(str) + "_" +
            df_input["Лиганд"].astype(str))

        t_range = t_act - t_syn
        df_input["T_range"] = t_range
        df_input["T_activation"] = t_act - 100.0
        df_input["T_dry_norm"] = (
            (t_dry - t_syn) / (t_range if t_range != 0 else 1e-9))

        for col in ['Металл', 'Лиганд', 'Растворитель',
                     'Metal_Ligand_Combo']:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)

        return df_input, constraint_penalty, penalty_reasons

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------
    def _predict_target(self, target_name: str,
                        df_input: pd.DataFrame,
                        trial: optuna.Trial
                        ) -> Tuple[float, float]:
        """Predict a single target. Returns (mean, calibrated_sigma)."""
        ensemble = self.models[target_name]
        model_features = ensemble[0].feature_names_

        missing = set(model_features) - set(df_input.columns)
        if missing:
            if trial.number == 0:
                print(f"Missing features for {target_name}: {missing}")
            raise optuna.TrialPruned(f"Missing features for {target_name}")

        df_slice = df_input[model_features]
        known_cats = ['Металл', 'Лиганд', 'Растворитель',
                      'Metal_Ligand_Combo']
        present_cats = [c for c in df_slice.columns if c in known_cats]
        pool = Pool(df_slice, cat_features=present_cats)

        preds = [m.predict(pool)[0] for m in ensemble]
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds))
        calibrated = self._calibrate_sigma(target_name, std_pred)
        return mean_pred, calibrated

    # ------------------------------------------------------------------
    # Multi-objective objective
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial,
                   targets: Dict[str, float],
                   weights: Dict[str, float]) -> Tuple[float, ...]:
        """Returns one loss value per target. Constraint penalty is stored
        as user_attr and handled by NSGAIISampler constraints_func."""

        try:
            df_input, constraint_penalty, penalty_reasons = \
                self._build_features(trial)
        except optuna.TrialPruned:
            # Return large losses for all targets so NSGA-II deprioritises
            trial.set_user_attr("ConstraintPenalty", 1e6)
            return tuple(1e6 for _ in targets)

        predictions = {}
        uncertainties = {}
        obj_values = []
        target_order = list(targets.keys())
        eps = 1e-8

        for target_name in target_order:
            target_val = targets[target_name]
            if target_name not in self.models:
                obj_values.append(1e6)
                continue
            try:
                mean_pred, cal_sigma = self._predict_target(
                    target_name, df_input, trial)
            except (optuna.TrialPruned, Exception):
                obj_values.append(1e6)
                continue

            predictions[target_name] = mean_pred
            uncertainties[target_name] = cal_sigma

            scale = abs(target_val) if target_val != 0 else max(
                abs(mean_pred), 1.0)
            err = ((mean_pred - target_val) / scale) ** 2
            unc = (cal_sigma / scale) ** 2
            obj = weights.get(target_name, 1.0) * (
                err + LAMBDA_UNCERTAINTY * unc)
            obj_values.append(obj)

        # Physics checks → add to constraint penalty
        e0_pred = predictions.get("E0, кДж/моль")
        if e0_pred is not None:
            lo_e0, hi_e0 = E0_BOUNDS_KJ_MOL
            v = _relative_violation(e0_pred, lo_e0, hi_e0)
            constraint_penalty += 300.0 * v

        # Store attrs
        for k, v in predictions.items():
            trial.set_user_attr(k, float(v))
        for k, v in uncertainties.items():
            trial.set_user_attr(f"Uncertainty_{k}", float(v))
        trial.set_user_attr("ConstraintPenalty", float(constraint_penalty))
        if penalty_reasons:
            trial.set_user_attr(
                "PenaltyReasons",
                [f"{r}:{p:.3f}" for r, p in penalty_reasons])

        return tuple(obj_values)

    # ------------------------------------------------------------------
    # Optimization entry point
    # ------------------------------------------------------------------
    def optimize(self, targets: Dict[str, float],
                 weights: Dict[str, float] = None) -> pd.DataFrame:
        if weights is None:
            weights = {k: 1.0 for k in targets}

        target_names = list(targets.keys())
        n_obj = len(target_names)
        print(f"\nMulti-objective optimization: {n_obj} targets, "
              f"{self.n_trials} trials")
        print(f"  Targets: {targets}")

        # NSGA-II with constraint handling
        def constraints_func(trial: optuna.trial.FrozenTrial) -> List[float]:
            cp = trial.user_attrs.get("ConstraintPenalty", 0.0)
            return [cp]  # positive = infeasible

        sampler = optuna.samplers.NSGAIISampler(
            constraints_func=constraints_func,
            seed=42,
        )
        study = optuna.create_study(
            directions=["minimize"] * n_obj,
            sampler=sampler,
        )
        study.optimize(
            lambda t: self._objective(t, targets, weights),
            n_trials=self.n_trials,
        )

        # --- Extract results ---
        # All completed trials
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print("No completed trials!")
            return pd.DataFrame()

        # Pareto front (non-dominated, feasible)
        pareto_trials = study.best_trials
        print(f"Pareto front: {len(pareto_trials)} non-dominated recipes "
              f"(of {len(completed)} completed)")

        # Build results DataFrame (Pareto first, then rest by scalarized)
        def _trial_to_row(t: optuna.trial.FrozenTrial,
                          is_pareto: bool) -> Dict:
            row = t.params.copy()
            row["Pareto"] = is_pareto
            # Per-target objective
            for i, name in enumerate(target_names):
                row[f"Obj_{name}"] = t.values[i]
            # Scalarized loss (sum of objectives)
            row["ScalarLoss"] = sum(t.values)
            # Predictions & uncertainties
            for k, v in t.user_attrs.items():
                row[f"Pred_{k}"] = v
            return row

        pareto_ids = {t.number for t in pareto_trials}
        rows = []
        for t in pareto_trials:
            rows.append(_trial_to_row(t, True))
        # Add non-Pareto, sorted by scalar loss, top 20
        non_pareto = sorted(
            [t for t in completed if t.number not in pareto_ids],
            key=lambda t: sum(t.values))[:20]
        for t in non_pareto:
            rows.append(_trial_to_row(t, False))

        return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective Inverse Design via Bayesian Optimization")

    parser.add_argument("--E0", type=float,
                        help="Target E0 (kJ/mol)")
    parser.add_argument("--x0", type=float,
                        help="Target x0 pore half-width (nm)")
    parser.add_argument("--Sme", type=float,
                        help="Target Sme mesopore surface area (m2/g)")

    parser.add_argument("--trials", type=int, default=300,
                        help="Number of optimization trials")
    parser.add_argument("--output", type=str,
                        default="predictions_bo.csv",
                        help="Output CSV file")
    parser.add_argument("--models", type=str,
                        default="artifacts/forward_models",
                        help="Path to trained models")
    parser.add_argument("--strict-validation", action="store_true",
                        help="Strict validation mode")

    args = parser.parse_args()

    targets = {}
    if args.E0 is not None:
        targets['E0, кДж/моль'] = args.E0
    if args.x0 is not None:
        targets['х0, нм'] = args.x0
    if args.Sme is not None:
        targets['Sme, м2/г'] = args.Sme

    if not targets:
        print("Error: specify at least one target (--E0, --x0, --Sme).")
        return

    optimizer = AdsorbentOptimizer(
        models_dir=args.models,
        data_path="data/SEC_SYN_with_features_enriched.csv",
        n_trials=args.trials,
        strict_validation=args.strict_validation,
    )

    df_results = optimizer.optimize(targets)

    # --- Display ---
    pareto_df = df_results[df_results["Pareto"] == True]
    print(f"\n=== Pareto Front ({len(pareto_df)} recipes) ===")

    obj_cols = [c for c in df_results.columns if c.startswith("Obj_")]
    pred_cols = [c for c in df_results.columns
                 if c.startswith("Pred_") and "Uncertainty" not in c
                 and "Constraint" not in c and "Penalty" not in c]
    unc_cols = [c for c in df_results.columns
                if c.startswith("Pred_Uncertainty")]
    display = (["ScalarLoss"] + obj_cols + pred_cols + unc_cols +
               ['Металл', 'Лиганд', 'Т.син., °С'])
    display = [c for c in display if c in df_results.columns]
    print(pareto_df[display].head(10).to_string(index=False))

    df_results.to_csv(args.output, index=False)
    print(f"\nFull results ({len(df_results)} rows) saved to {args.output}")


if __name__ == "__main__":
    main()
