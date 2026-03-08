#!/usr/bin/env python3
"""Target-oriented inverse design using BoFire domain models and trained surrogates."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.features.api import CategoricalInput, ContinuousInput, ContinuousOutput, DiscreteInput
from bofire.data_models.objectives.api import CloseToTargetObjective
from catboost import CatBoostRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.constants import (
    DEFAULT_STOICHIOMETRY_BOUNDS,
    E0_BOUNDS_KJ_MOL,
    FORWARD_MODEL_TARGETS,
    SOLVENT_BOILING_POINTS_C,
    STOICHIOMETRY_TARGETS,
)
from adsorb_synthesis.data_processing import (
    add_interaction_features,
    add_physicochemical_descriptors,
    add_salt_mass_features,
    build_lookup_tables,
    load_dataset,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ConstraintResult:
    feasible: bool
    reasons: List[str]


class BofireAdsorbentOptimizer:
    def __init__(
        self,
        models_dir: str,
        data_path: str,
        n_trials: int = 200,
        strict_validation: bool = False,
    ):
        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
        self.strict_validation = strict_validation

        validation_mode = "strict" if strict_validation else "warn"
        self.df_ref = load_dataset(data_path, validation_mode=validation_mode)
        self.lookup_tables = build_lookup_tables(self.df_ref)
        self.models = self._load_models()
        self.uq_models = self._load_uq_models()
        self.domain = self._build_domain()
        self.required_model_features = sorted({
            feature
            for ensemble in self.models.values()
            for model in ensemble
            for feature in model.feature_names_
        })
        self.input_keys = [
            "Металл",
            "Лиганд",
            "Растворитель",
            "m (соли), г",
            "m(кис-ты), г",
            "Vсин. (р-ля), мл",
            "Т.син., °С",
            "Т суш., °С",
            "Tрег, ᵒС",
        ]
        self.numeric_input_keys = [
            "m (соли), г",
            "m(кис-ты), г",
            "Vсин. (р-ля), мл",
            "Т.син., °С",
            "Т суш., °С",
            "Tрег, ᵒС",
        ]
        self.feasible_reference_rows = self._build_feasible_reference_pool()
        self.combo_reference_stats = self._build_combo_reference_stats()
        self.max_combo_solvents = max(len(stats["solvents"]) for stats in self.combo_reference_stats.values())
        self.max_temp_templates = max(len(stats["temp_triplets"]) for stats in self.combo_reference_stats.values())
        self.diversity_ranges = self._build_diversity_ranges()

    def _load_models(self) -> Dict[str, List[CatBoostRegressor]]:
        models: Dict[str, List[CatBoostRegressor]] = {}
        for target in FORWARD_MODEL_TARGETS:
            safe_target = target.replace('/', '_').replace(' ', '_')
            target_models = []
            for member_idx in range(25):
                model_path = os.path.join(self.models_dir, f"catboost_{safe_target}_ens{member_idx}.cbm")
                if not os.path.exists(model_path):
                    continue
                model = CatBoostRegressor()
                model.load_model(model_path)
                target_models.append(model)
            if target_models:
                models[target] = target_models
        if not models:
            raise RuntimeError("No trained production models found. Run train_forward_model.py first.")
        return models

    def _load_uq_models(self) -> Dict[str, object]:
        path = os.path.join(self.models_dir, "uncertainty_calibrators.joblib")
        if not os.path.exists(path):
            raise RuntimeError("Missing uncertainty_calibrators.joblib. Run train_forward_model.py first.")
        return joblib.load(path)

    def _build_domain(self) -> Domain:
        outputs = []
        for target in FORWARD_MODEL_TARGETS:
            outputs.append(
                ContinuousOutput(
                    key=target,
                    objective=CloseToTargetObjective(target_value=0.0, exponent=2.0, w=1.0),
                )
            )
        return Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(key="Металл", categories=sorted(self.df_ref["Металл"].dropna().astype(str).unique().tolist())),
                    CategoricalInput(key="Лиганд", categories=sorted(self.df_ref["Лиганд"].dropna().astype(str).unique().tolist())),
                    CategoricalInput(key="Растворитель", categories=sorted(self.df_ref["Растворитель"].dropna().astype(str).unique().tolist())),
                    ContinuousInput(key="m (соли), г", bounds=(float(self.df_ref["m (соли), г"].min()), float(self.df_ref["m (соли), г"].max())), allow_zero=False),
                    ContinuousInput(key="m(кис-ты), г", bounds=(float(self.df_ref["m(кис-ты), г"].min()), float(self.df_ref["m(кис-ты), г"].max())), allow_zero=False),
                    ContinuousInput(key="Vсин. (р-ля), мл", bounds=(float(self.df_ref["Vсин. (р-ля), мл"].min()), float(self.df_ref["Vсин. (р-ля), мл"].max())), allow_zero=False),
                    DiscreteInput(key="Т.син., °С", values=sorted(pd.to_numeric(self.df_ref["Т.син., °С"], errors="coerce").dropna().astype(float).unique().tolist())),
                    DiscreteInput(key="Т суш., °С", values=sorted(pd.to_numeric(self.df_ref["Т суш., °С"], errors="coerce").dropna().astype(float).unique().tolist())),
                    DiscreteInput(key="Tрег, ᵒС", values=sorted(pd.to_numeric(self.df_ref["Tрег, ᵒС"], errors="coerce").dropna().astype(float).unique().tolist())),
                ]
            ),
            outputs=Outputs(features=outputs),
        )

    def _build_feasible_reference_pool(self) -> pd.DataFrame:
        records = []
        for _, row in self.df_ref.iterrows():
            params = {
                key: row[key]
                for key in self.input_keys
                if key in row.index
            }
            constraint_result = self._constraint_check(params)
            if not constraint_result.feasible:
                continue
            record = row[self.input_keys + [target for target in FORWARD_MODEL_TARGETS if target in row.index]].to_dict()
            record["combo_key"] = self._combo_key(str(row["Металл"]), str(row["Лиганд"]))
            records.append(record)
        if not records:
            raise RuntimeError("No feasible reference rows found under current constraints.")
        return pd.DataFrame(records).reset_index(drop=True)

    def _build_combo_reference_stats(self) -> Dict[str, Dict[str, object]]:
        stats: Dict[str, Dict[str, object]] = {}
        for combo_key, combo_df in self.feasible_reference_rows.groupby("combo_key"):
            combo_stats: Dict[str, object] = {
                "metals": sorted(combo_df["Металл"].astype(str).unique().tolist()),
                "ligands": sorted(combo_df["Лиганд"].astype(str).unique().tolist()),
                "solvents": sorted(combo_df["Растворитель"].astype(str).unique().tolist()),
                "temp_triplets": combo_df[["Растворитель", "Т.син., °С", "Т суш., °С", "Tрег, ᵒС"]].drop_duplicates().to_dict("records"),
            }
            for feature in ["m (соли), г", "m(кис-ты), г", "Vсин. (р-ля), мл", "R_molar"]:
                if feature in combo_df.columns:
                    numeric = pd.to_numeric(combo_df[feature], errors="coerce").dropna()
                    if not numeric.empty:
                        combo_stats[feature] = {
                            "min": float(numeric.min()),
                            "max": float(numeric.max()),
                            "median": float(numeric.median()),
                        }
            stats[combo_key] = combo_stats
        return stats

    def _build_diversity_ranges(self) -> Dict[str, float]:
        ranges: Dict[str, float] = {}
        for feature in self.numeric_input_keys:
            numeric = pd.to_numeric(self.df_ref[feature], errors="coerce").dropna()
            if numeric.empty:
                ranges[feature] = 1.0
                continue
            feature_range = float(numeric.max() - numeric.min())
            ranges[feature] = feature_range if feature_range > 0 else 1.0
        return ranges

    @staticmethod
    def _combo_key(metal: str, ligand: str) -> str:
        return f"{metal}__{ligand}"

    def _get_ratio_bounds(self, metal: str, ligand: str) -> Tuple[float, float]:
        stoich = STOICHIOMETRY_TARGETS.get((metal, ligand))
        if stoich is None:
            return DEFAULT_STOICHIOMETRY_BOUNDS
        tolerance = stoich.get("tolerance", 0.1)
        return (
            stoich["ratio"] * (1 - tolerance),
            stoich["ratio"] * (1 + tolerance),
        )

    def _sample_near_anchor(
        self,
        trial: optuna.Trial,
        *,
        name: str,
        anchor: float,
        lower: float,
        upper: float,
        log: bool = False,
    ) -> float:
        if lower >= upper:
            return float(lower)
        span = upper - lower
        local_lower = max(lower, anchor - 0.35 * span)
        local_upper = min(upper, anchor + 0.35 * span)
        if local_lower >= local_upper:
            local_lower, local_upper = lower, upper
        return float(trial.suggest_float(name, local_lower, local_upper, log=log))

    def _sample_structured_params(
        self,
        trial: optuna.Trial,
        candidate_pool: pd.DataFrame,
    ) -> Dict[str, object]:
        anchor_idx = int(trial.suggest_categorical("anchor_idx", candidate_pool.index.tolist()))
        anchor = candidate_pool.loc[anchor_idx]
        combo_key = str(anchor["combo_key"])
        combo_stats = self.combo_reference_stats[combo_key]
        metal = str(anchor["Металл"])
        ligand = str(anchor["Лиганд"])
        params: Dict[str, object] = {
            "Металл": metal,
            "Лиганд": ligand,
        }

        solvent_choices = combo_stats["solvents"]
        solvent_rank = trial.suggest_int("solvent_rank", 0, max(0, self.max_combo_solvents - 1))
        params["Растворитель"] = solvent_choices[solvent_rank % len(solvent_choices)]

        temp_candidates = [
            record for record in combo_stats["temp_triplets"]
            if str(record["Растворитель"]) == str(params["Растворитель"])
        ]
        if not temp_candidates:
            temp_candidates = combo_stats["temp_triplets"]
        temp_choice_idx = int(
            trial.suggest_int("temp_template_rank", 0, max(0, self.max_temp_templates - 1))
        ) % len(temp_candidates)
        temp_choice = temp_candidates[temp_choice_idx]
        params["Т.син., °С"] = float(temp_choice["Т.син., °С"])
        params["Т суш., °С"] = float(temp_choice["Т суш., °С"])
        params["Tрег, ᵒС"] = float(temp_choice["Tрег, ᵒС"])

        ratio_lo, ratio_hi = self._get_ratio_bounds(metal, ligand)
        anchor_ratio = float(anchor.get("R_molar", (ratio_lo + ratio_hi) / 2.0))
        ratio = self._sample_near_anchor(
            trial,
            name="R_molar_target",
            anchor=anchor_ratio,
            lower=ratio_lo,
            upper=ratio_hi,
        )

        metal_row = self.lookup_tables.metal.loc[metal]
        ligand_row = self.lookup_tables.ligand.loc[ligand]
        if isinstance(metal_row, pd.DataFrame):
            metal_row = metal_row.iloc[0]
        if isinstance(ligand_row, pd.DataFrame):
            ligand_row = ligand_row.iloc[0]
        mw_salt = float(metal_row["Молярка_соли"])
        mw_acid = float(ligand_row["Молярка_кислоты"])

        salt_bounds = next(feature.bounds for feature in self.domain.inputs.features if feature.key == "m (соли), г")
        acid_bounds = next(feature.bounds for feature in self.domain.inputs.features if feature.key == "m(кис-ты), г")
        volume_bounds = next(feature.bounds for feature in self.domain.inputs.features if feature.key == "Vсин. (р-ля), мл")

        acid_from_salt_min = float(salt_bounds[0]) * mw_acid / (ratio * mw_salt)
        acid_from_salt_max = float(salt_bounds[1]) * mw_acid / (ratio * mw_salt)
        acid_lower = max(float(acid_bounds[0]), acid_from_salt_min)
        acid_upper = min(float(acid_bounds[1]), acid_from_salt_max)
        if acid_lower >= acid_upper:
            acid_lower = float(anchor["m(кис-ты), г"])
            acid_upper = float(anchor["m(кис-ты), г"])
        acid_mass = self._sample_near_anchor(
            trial,
            name="m(кис-ты), г",
            anchor=float(anchor["m(кис-ты), г"]),
            lower=acid_lower,
            upper=acid_upper,
            log=True,
        )
        salt_mass = ratio * acid_mass * mw_salt / mw_acid
        salt_mass = float(np.clip(salt_mass, float(salt_bounds[0]), float(salt_bounds[1])))

        params["m(кис-ты), г"] = acid_mass
        params["m (соли), г"] = salt_mass
        params["Vсин. (р-ля), мл"] = self._sample_near_anchor(
            trial,
            name="Vсин. (р-ля), мл",
            anchor=float(anchor["Vсин. (р-ля), мл"]),
            lower=float(volume_bounds[0]),
            upper=float(volume_bounds[1]),
        )
        return params

    def _constraint_check(self, params: Dict[str, object]) -> ConstraintResult:
        reasons: List[str] = []
        t_syn = float(params["Т.син., °С"])
        t_dry = float(params["Т суш., °С"])
        t_reg = float(params["Tрег, ᵒС"])
        solvent = str(params["Растворитель"])

        if t_dry > t_syn + 20:
            reasons.append("dry_above_synthesis_window")
        if t_dry > t_reg:
            reasons.append("dry_above_regeneration")
        boiling_point = (
            SOLVENT_BOILING_POINTS_C.get(solvent)
            or SOLVENT_BOILING_POINTS_C.get(solvent.capitalize())
            or SOLVENT_BOILING_POINTS_C.get(solvent.lower())
        )
        if boiling_point is not None and t_syn > boiling_point:
            reasons.append("synthesis_above_boiling")

        metal = str(params["Металл"])
        ligand = str(params["Лиганд"])
        metal_row = self.lookup_tables.metal.loc[metal]
        ligand_row = self.lookup_tables.ligand.loc[ligand]
        if isinstance(metal_row, pd.DataFrame):
            metal_row = metal_row.iloc[0]
        if isinstance(ligand_row, pd.DataFrame):
            ligand_row = ligand_row.iloc[0]
        mw_salt = float(metal_row["Молярка_соли"])
        mw_acid = float(ligand_row["Молярка_кислоты"])
        n_salt = float(params["m (соли), г"]) / mw_salt if mw_salt else np.nan
        n_acid = float(params["m(кис-ты), г"]) / mw_acid if mw_acid else np.nan
        if not np.isfinite(n_salt) or not np.isfinite(n_acid) or n_acid <= 0:
            reasons.append("invalid_stoichiometry_inputs")
        else:
            ratio = n_salt / n_acid
            stoich = STOICHIOMETRY_TARGETS.get((metal, ligand))
            if stoich is None:
                lower, upper = DEFAULT_STOICHIOMETRY_BOUNDS
            else:
                lower = stoich["ratio"] * (1 - stoich.get("tolerance", 0.1))
                upper = stoich["ratio"] * (1 + stoich.get("tolerance", 0.1))
            if ratio < lower or ratio > upper:
                reasons.append("stoichiometry_out_of_bounds")

        return ConstraintResult(feasible=not reasons, reasons=reasons)

    def _build_feature_row(self, params: Dict[str, object]) -> pd.DataFrame:
        recipe = pd.DataFrame([params])
        metal = str(params["Металл"])
        ligand = str(params["Лиганд"])
        solvent = str(params["Растворитель"])

        metal_desc = self.lookup_tables.metal.loc[metal]
        ligand_desc = self.lookup_tables.ligand.loc[ligand]
        solvent_desc = self.lookup_tables.solvent.loc[solvent]
        if isinstance(metal_desc, pd.DataFrame):
            metal_desc = metal_desc.iloc[0]
        if isinstance(ligand_desc, pd.DataFrame):
            ligand_desc = ligand_desc.iloc[0]
        if isinstance(solvent_desc, pd.DataFrame):
            solvent_desc = solvent_desc.iloc[0]

        for key, value in metal_desc.items():
            recipe[key] = value
        for key, value in ligand_desc.items():
            recipe[key] = value
        for key, value in solvent_desc.items():
            recipe[key] = value

        add_salt_mass_features(recipe, inplace=True)
        add_physicochemical_descriptors(recipe, inplace=True)
        add_interaction_features(recipe, inplace=True)

        reference_row = self._find_reference_row(metal=metal, ligand=ligand, solvent=solvent)
        if reference_row is not None:
            for feature in self.required_model_features:
                if feature not in recipe.columns and feature in reference_row.index:
                    recipe[feature] = reference_row[feature]
        return recipe

    def _find_reference_row(self, *, metal: str, ligand: str, solvent: str) -> Optional[pd.Series]:
        selectors = [
            (self.df_ref["Металл"].astype(str) == metal)
            & (self.df_ref["Лиганд"].astype(str) == ligand)
            & (self.df_ref["Растворитель"].astype(str) == solvent),
            (self.df_ref["Металл"].astype(str) == metal)
            & (self.df_ref["Лиганд"].astype(str) == ligand),
            self.df_ref["Металл"].astype(str) == metal,
        ]
        for mask in selectors:
            subset = self.df_ref.loc[mask]
            if not subset.empty:
                return subset.iloc[0]
        return None

    def _predict_target(self, target: str, feature_row: pd.DataFrame) -> Dict[str, float]:
        prod_models = self.models[target]
        model_features = prod_models[0].feature_names_
        prod_preds = np.asarray([model.predict(feature_row[model_features])[0] for model in prod_models], dtype=float)
        prod_mean = float(prod_preds.mean())

        uq_model = self.uq_models[target]
        interval_center, bounds = uq_model.predict_interval(feature_row, aggregate_predictions="mean")
        bounds = np.asarray(bounds, dtype=float).squeeze(-1)
        return {
            "mean": prod_mean,
            "interval_center": float(np.asarray(interval_center, dtype=float)[0]),
            "lo": float(bounds[0, 0]),
            "hi": float(bounds[0, 1]),
            "width": float(bounds[0, 1] - bounds[0, 0]),
        }

    @staticmethod
    def _target_score(target_value: float, prediction: float, exponent: float = 2.0) -> float:
        scale = abs(target_value) if target_value != 0 else max(abs(prediction), 1.0)
        return float((abs(prediction - target_value) / scale) ** exponent)

    def _build_candidate_pool(self, targets: Dict[str, float], max_templates: int = 48) -> pd.DataFrame:
        scored = self.feasible_reference_rows.copy()
        score = np.zeros(len(scored), dtype=float)
        for target_name, target_value in targets.items():
            if target_name not in scored.columns:
                continue
            observed = pd.to_numeric(scored[target_name], errors="coerce").to_numpy(dtype=float)
            scale = abs(target_value) if target_value != 0 else np.maximum(np.abs(observed), 1.0)
            score += np.square((observed - target_value) / scale)
        scored["reference_score"] = score
        scored = scored.sort_values(["reference_score"]).head(max_templates).reset_index(drop=True)
        return scored

    def _target_deviation(self, target_value: float, prediction: float) -> float:
        scale = abs(target_value) if target_value != 0 else max(abs(prediction), 1.0)
        return float(prediction - target_value) / scale

    @staticmethod
    def _chemistry_key(row: pd.Series) -> str:
        return "|".join(
            [
                str(row["Металл"]),
                str(row["Лиганд"]),
                str(row["Растворитель"]),
            ]
        )

    @staticmethod
    def _process_key(row: pd.Series) -> str:
        return "|".join(
            [
                f"{float(row['Т.син., °С']):.1f}",
                f"{float(row['Т суш., °С']):.1f}",
                f"{float(row['Tрег, ᵒС']):.1f}",
            ]
        )

    def _recipe_signature(self, row: pd.Series) -> Tuple[object, ...]:
        return (
            str(row["Металл"]),
            str(row["Лиганд"]),
            str(row["Растворитель"]),
            round(float(row["m (соли), г"]), 4),
            round(float(row["m(кис-ты), г"]), 4),
            round(float(row["Vсин. (р-ля), мл"]), 3),
            round(float(row["Т.син., °С"]), 1),
            round(float(row["Т суш., °С"]), 1),
            round(float(row["Tрег, ᵒС"]), 1),
        )

    def _recipe_distance(self, left: pd.Series, right: pd.Series) -> float:
        deltas = []
        for feature in self.numeric_input_keys:
            feature_range = self.diversity_ranges.get(feature, 1.0)
            if feature_range <= 0:
                feature_range = 1.0
            left_value = float(left[feature])
            right_value = float(right[feature])
            deltas.append(abs(left_value - right_value) / feature_range)
        return float(np.mean(deltas))

    def _select_diverse_shortlist(
        self,
        results: pd.DataFrame,
        *,
        shortlist_size: int,
        max_per_chemistry: int,
        max_per_process: int,
        min_distance: float,
    ) -> pd.DataFrame:
        if results.empty:
            return results

        pool = results.copy()
        feasible_pool = pool.loc[pool["feasible"]].copy()
        if not feasible_pool.empty:
            pool = feasible_pool

        pool["chemistry_key"] = pool.apply(self._chemistry_key, axis=1)
        pool["process_key"] = pool.apply(self._process_key, axis=1)
        pool["search_rank"] = np.arange(1, len(pool) + 1)

        selected_indices: List[int] = []
        seen_signatures = set()
        chemistry_counts: Dict[str, int] = {}
        process_counts: Dict[str, int] = {}

        def accept_candidate(
            candidate: pd.Series,
            *,
            enforce_chemistry_cap: bool,
            enforce_process_cap: bool,
            enforce_distance: bool,
        ) -> bool:
            signature = self._recipe_signature(candidate)
            if signature in seen_signatures:
                return False

            chemistry_key = str(candidate["chemistry_key"])
            process_key = str(candidate["process_key"])
            if enforce_chemistry_cap and chemistry_counts.get(chemistry_key, 0) >= max_per_chemistry:
                return False
            if enforce_process_cap and process_counts.get(process_key, 0) >= max_per_process:
                return False

            if enforce_distance:
                for selected_idx in selected_indices:
                    selected = pool.iloc[selected_idx]
                    if str(selected["chemistry_key"]) != chemistry_key:
                        continue
                    if self._recipe_distance(candidate, selected) < min_distance:
                        return False

            selected_indices.append(int(candidate.name))
            seen_signatures.add(signature)
            chemistry_counts[chemistry_key] = chemistry_counts.get(chemistry_key, 0) + 1
            process_counts[process_key] = process_counts.get(process_key, 0) + 1
            return True

        selection_passes = [
            (True, True, True),
            (True, False, True),
            (True, False, False),
            (False, False, False),
        ]
        for enforce_chemistry_cap, enforce_process_cap, enforce_distance in selection_passes:
            for _, candidate in pool.iterrows():
                if len(selected_indices) >= shortlist_size:
                    break
                accept_candidate(
                    candidate,
                    enforce_chemistry_cap=enforce_chemistry_cap,
                    enforce_process_cap=enforce_process_cap,
                    enforce_distance=enforce_distance,
                )
            if len(selected_indices) >= shortlist_size:
                break

        shortlist = pool.iloc[selected_indices].copy().reset_index(drop=True)
        shortlist["rank"] = np.arange(1, len(shortlist) + 1)
        shortlist["selection_mode"] = "diverse_shortlist"
        return shortlist

    def optimize(
        self,
        targets: Dict[str, float],
        *,
        shortlist_size: int,
        max_per_chemistry: int,
        max_per_process: int,
        min_distance: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not targets:
            raise ValueError("Specify at least one target.")

        candidate_pool = self._build_candidate_pool(targets)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        warm_start_count = min(12, len(candidate_pool))

        def objective(trial: optuna.Trial) -> float:
            if trial.number < warm_start_count:
                anchor = candidate_pool.iloc[trial.number]
                params = {
                    key: anchor[key]
                    for key in self.input_keys
                }
            else:
                params = self._sample_structured_params(trial, candidate_pool)
            for key, value in params.items():
                trial.set_user_attr(f"Param_{key}", value)
            constraint_result = self._constraint_check(params)
            trial.set_user_attr("feasible", constraint_result.feasible)
            trial.set_user_attr("constraint_reasons", constraint_result.reasons)
            if not constraint_result.feasible:
                return 1e6

            feature_row = self._build_feature_row(params)
            score = 0.0
            for target_name, target_value in targets.items():
                prediction = self._predict_target(target_name, feature_row)
                trial.set_user_attr(f"Pred_{target_name}", prediction["mean"])
                trial.set_user_attr(f"Pred_{target_name}_lo", prediction["lo"])
                trial.set_user_attr(f"Pred_{target_name}_hi", prediction["hi"])
                trial.set_user_attr(f"Pred_{target_name}_width", prediction["width"])
                trial.set_user_attr(f"Pred_{target_name}_center", prediction["interval_center"])
                trial.set_user_attr(
                    f"Pred_{target_name}_deviation",
                    self._target_deviation(target_value, prediction["mean"]),
                )
                score += self._target_score(target_value, prediction["mean"])

            e0_pred = trial.user_attrs.get("Pred_E0, кДж/моль")
            if e0_pred is not None and not (E0_BOUNDS_KJ_MOL[0] <= e0_pred <= E0_BOUNDS_KJ_MOL[1]):
                trial.set_user_attr("feasible", False)
                trial.set_user_attr("constraint_reasons", constraint_result.reasons + ["predicted_E0_out_of_bounds"])
                return 1e6

            return score

        study.optimize(objective, n_trials=self.n_trials)

        rows = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {}
            for key, value in trial.user_attrs.items():
                if key.startswith("Param_"):
                    row[key.removeprefix("Param_")] = value
            row["score"] = float(trial.value)
            row["feasible"] = bool(trial.user_attrs.get("feasible", False))
            row["constraint_reasons"] = ";".join(trial.user_attrs.get("constraint_reasons", []))
            for key, value in trial.user_attrs.items():
                if key.startswith("Pred_"):
                    row[key] = value
            rows.append(row)

        results = pd.DataFrame(rows)
        if results.empty:
            return results, results
        sort_columns = ["feasible", "score"]
        ascending = [False, True]
        width_columns = [column for column in results.columns if column.endswith("_width")]
        if width_columns:
            results["interval_width_total"] = results[width_columns].sum(axis=1)
            sort_columns.append("interval_width_total")
            ascending.append(True)
        results = results.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)
        feasible_count = int(results["feasible"].sum())
        print(f"Feasible candidates: {feasible_count}/{len(results)}")
        shortlist = self._select_diverse_shortlist(
            results,
            shortlist_size=shortlist_size,
            max_per_chemistry=max_per_chemistry,
            max_per_process=max_per_process,
            min_distance=min_distance,
        )
        return shortlist, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-oriented inverse design via BoFire domain models.")
    parser.add_argument("--E0", type=float, help="Target E0 (kJ/mol)")
    parser.add_argument("--x0", type=float, help="Target x0 (nm)")
    parser.add_argument("--Sme", type=float, help="Target Sme (m2/g)")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--max-per-chemistry", type=int, default=4)
    parser.add_argument("--max-per-process", type=int, default=3)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models-dir", "--models", dest="models_dir", type=str, default="artifacts/forward_models")
    parser.add_argument("--output", type=str, default="artifacts/predictions_bofire.csv")
    parser.add_argument("--all-output", type=str, help="Optional path to save the full searched candidate pool.")
    parser.add_argument("--strict-validation", action="store_true")
    args = parser.parse_args()

    targets: Dict[str, float] = {}
    if args.E0 is not None:
        targets["E0, кДж/моль"] = args.E0
    if args.x0 is not None:
        targets["х0, нм"] = args.x0
    if args.Sme is not None:
        targets["Sme, м2/г"] = args.Sme
    if not targets:
        raise SystemExit("Specify at least one target via --E0, --x0, or --Sme.")

    optimizer = BofireAdsorbentOptimizer(
        models_dir=args.models_dir,
        data_path=args.data,
        n_trials=args.trials,
        strict_validation=args.strict_validation,
    )
    results, all_results = optimizer.optimize(
        targets,
        shortlist_size=args.shortlist_size,
        max_per_chemistry=args.max_per_chemistry,
        max_per_process=args.max_per_process,
        min_distance=args.min_distance,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    if args.all_output:
        os.makedirs(os.path.dirname(args.all_output) or ".", exist_ok=True)
        all_results.to_csv(args.all_output, index=False)
        print(f"Saved full search pool ({len(all_results)} candidates) to {args.all_output}")
    print(f"Saved shortlist ({len(results)} candidates) to {args.output}")
    if not results.empty:
        print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
