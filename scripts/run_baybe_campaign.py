#!/usr/bin/env python3
"""Campaign-oriented inverse design scaffold using BayBE."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from baybe.campaign import Campaign
from baybe.objectives import DesirabilityObjective
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender, FPSRecommender, TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from catboost import CatBoostRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adsorb_synthesis.constants import (  # noqa: E402
    DEFAULT_STOICHIOMETRY_BOUNDS,
    E0_BOUNDS_KJ_MOL,
    FORWARD_MODEL_TARGETS,
    SOLVENT_BOILING_POINTS_C,
    STOICHIOMETRY_TARGETS,
)
from adsorb_synthesis.data_processing import (  # noqa: E402
    add_interaction_features,
    add_physicochemical_descriptors,
    add_salt_mass_features,
    build_lookup_tables,
    load_dataset,
)


@dataclass
class ConstraintResult:
    feasible: bool
    reasons: List[str]


class BaybeCampaignOptimizer:
    def __init__(
        self,
        models_dir: str,
        data_path: str,
        n_trials: int = 24,
        batch_size: int = 4,
        candidate_pool_size: int = 240,
        init_measurements: int = 12,
        strict_validation: bool = False,
        template_limit: int = 12,
    ):
        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.candidate_pool_size = max(candidate_pool_size, n_trials + init_measurements)
        self.init_measurements = init_measurements
        self.strict_validation = strict_validation
        self.template_limit = template_limit
        self.rng = np.random.default_rng(42)

        validation_mode = "strict" if strict_validation else "warn"
        self.df_ref = load_dataset(data_path, validation_mode=validation_mode)
        self.lookup_tables = build_lookup_tables(self.df_ref)
        self.models = self._load_models()
        self.uq_models = self._load_uq_models()
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
        self.salt_bounds = self._numeric_bounds("m (соли), г")
        self.acid_bounds = self._numeric_bounds("m(кис-ты), г")
        self.volume_bounds = self._numeric_bounds("Vсин. (р-ля), мл")
        self.feasible_reference_rows = self._build_feasible_reference_pool()
        self.combo_reference_stats = self._build_combo_reference_stats()
        self.max_combo_solvents = max(len(stats["solvents"]) for stats in self.combo_reference_stats.values())
        self.max_temp_templates = max(len(stats["temp_triplets"]) for stats in self.combo_reference_stats.values())
        self.diversity_ranges = self._build_diversity_ranges()
        self.campaign_variable_keys: List[str] = []
        self.campaign_fixed_values: Dict[str, object] = {}

    def _numeric_bounds(self, column: str) -> Tuple[float, float]:
        numeric = pd.to_numeric(self.df_ref[column], errors="coerce").dropna()
        return float(numeric.min()), float(numeric.max())

    def _load_models(self) -> Dict[str, List[CatBoostRegressor]]:
        models: Dict[str, List[CatBoostRegressor]] = {}
        for target in FORWARD_MODEL_TARGETS:
            safe_target = target.replace("/", "_").replace(" ", "_")
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
            raise RuntimeError("No trained production CatBoost models found. Run train_forward_model.py first.")
        return models

    def _load_uq_models(self) -> Dict[str, object]:
        path = os.path.join(self.models_dir, "uncertainty_calibrators.joblib")
        if not os.path.exists(path):
            raise RuntimeError("Missing uncertainty_calibrators.joblib. Run train_forward_model.py first.")
        return joblib.load(path)

    @staticmethod
    def _combo_key(metal: str, ligand: str) -> str:
        return f"{metal}__{ligand}"

    def _build_feasible_reference_pool(self) -> pd.DataFrame:
        records = []
        for _, row in self.df_ref.iterrows():
            params = {key: row[key] for key in self.input_keys if key in row.index}
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

    def _get_ratio_bounds(self, metal: str, ligand: str) -> Tuple[float, float]:
        stoich = STOICHIOMETRY_TARGETS.get((metal, ligand))
        if stoich is None:
            return DEFAULT_STOICHIOMETRY_BOUNDS
        tolerance = stoich.get("tolerance", 0.1)
        return (
            stoich["ratio"] * (1 - tolerance),
            stoich["ratio"] * (1 + tolerance),
        )

    def _constraint_check(self, params: Dict[str, object]) -> ConstraintResult:
        reasons: List[str] = []
        t_syn = float(params["Т.син., °С"])
        t_dry = float(params["Т суш., °С"])
        t_reg = float(params["Tрег, ᵒС"])
        solvent = str(params["Растворитель"])

        if not (self.salt_bounds[0] <= float(params["m (соли), г"]) <= self.salt_bounds[1]):
            reasons.append("salt_mass_out_of_bounds")
        if not (self.acid_bounds[0] <= float(params["m(кис-ты), г"]) <= self.acid_bounds[1]):
            reasons.append("acid_mass_out_of_bounds")
        if not (self.volume_bounds[0] <= float(params["Vсин. (р-ля), мл"]) <= self.volume_bounds[1]):
            reasons.append("volume_out_of_bounds")
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
            lower, upper = self._get_ratio_bounds(metal, ligand)
            if ratio < lower or ratio > upper:
                reasons.append("stoichiometry_out_of_bounds")
        return ConstraintResult(feasible=not reasons, reasons=reasons)

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

    def _target_deviation(self, target_value: float, prediction: float) -> float:
        scale = abs(target_value) if target_value != 0 else max(abs(prediction), 1.0)
        return float(prediction - target_value) / scale

    def _reference_score(self, row: pd.Series, targets: Dict[str, float]) -> float:
        score = 0.0
        for target_name, target_value in targets.items():
            observed = float(row[target_name])
            scale = abs(target_value) if target_value != 0 else max(abs(observed), 1.0)
            score += ((observed - target_value) / scale) ** 2
        return float(score)

    def _build_seed_measurements(self, targets: Dict[str, float]) -> pd.DataFrame:
        seed = self.feasible_reference_rows.copy()
        seed["reference_score"] = seed.apply(lambda row: self._reference_score(row, targets), axis=1)
        return seed.sort_values("reference_score").head(self.init_measurements).reset_index(drop=True)

    def _sample_candidate_from_anchor(self, anchor: pd.Series) -> Dict[str, object]:
        combo_key = str(anchor["combo_key"])
        combo_stats = self.combo_reference_stats[combo_key]
        metal = str(anchor["Металл"])
        ligand = str(anchor["Лиганд"])
        solvents = combo_stats["solvents"]
        solvent = solvents[int(self.rng.integers(0, len(solvents)))]
        temp_candidates = [
            record for record in combo_stats["temp_triplets"]
            if str(record["Растворитель"]) == solvent
        ]
        if not temp_candidates:
            temp_candidates = combo_stats["temp_triplets"]
        temp_choice = temp_candidates[int(self.rng.integers(0, len(temp_candidates)))]

        ratio_lo, ratio_hi = self._get_ratio_bounds(metal, ligand)
        anchor_ratio = float(anchor.get("R_molar", (ratio_lo + ratio_hi) / 2.0))
        ratio_span = ratio_hi - ratio_lo
        ratio = float(np.clip(
            self.rng.normal(anchor_ratio, 0.18 * max(ratio_span, 1e-6)),
            ratio_lo,
            ratio_hi,
        ))

        metal_row = self.lookup_tables.metal.loc[metal]
        ligand_row = self.lookup_tables.ligand.loc[ligand]
        if isinstance(metal_row, pd.DataFrame):
            metal_row = metal_row.iloc[0]
        if isinstance(ligand_row, pd.DataFrame):
            ligand_row = ligand_row.iloc[0]
        mw_salt = float(metal_row["Молярка_соли"])
        mw_acid = float(ligand_row["Молярка_кислоты"])

        acid_anchor = float(anchor["m(кис-ты), г"])
        acid_mass = float(np.clip(
            self.rng.normal(acid_anchor, max(acid_anchor * 0.25, 0.01)),
            self.acid_bounds[0],
            self.acid_bounds[1],
        ))
        salt_mass = float(np.clip(ratio * acid_mass * mw_salt / mw_acid, self.salt_bounds[0], self.salt_bounds[1]))
        volume_anchor = float(anchor["Vсин. (р-ля), мл"])
        volume = float(np.clip(
            self.rng.normal(volume_anchor, max(volume_anchor * 0.25, 1.0)),
            self.volume_bounds[0],
            self.volume_bounds[1],
        ))

        return {
            "Металл": metal,
            "Лиганд": ligand,
            "Растворитель": solvent,
            "m (соли), г": salt_mass,
            "m(кис-ты), г": acid_mass,
            "Vсин. (р-ля), мл": volume,
            "Т.син., °С": float(temp_choice["Т.син., °С"]),
            "Т суш., °С": float(temp_choice["Т суш., °С"]),
            "Tрег, ᵒС": float(temp_choice["Tрег, ᵒС"]),
        }

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

    def _build_candidate_pool(self, targets: Dict[str, float]) -> pd.DataFrame:
        seeds = self._build_seed_measurements(targets)
        anchors = seeds.head(min(self.template_limit, len(seeds)))
        records: List[dict] = []
        seen = set()

        for _, row in anchors.iterrows():
            params = {key: row[key] for key in self.input_keys}
            sig = self._recipe_signature(pd.Series(params))
            if sig not in seen:
                records.append(params)
                seen.add(sig)

        max_attempts = self.candidate_pool_size * 30
        attempts = 0
        anchor_records = list(anchors.to_dict("records"))
        while len(records) < self.candidate_pool_size and attempts < max_attempts:
            anchor = pd.Series(anchor_records[int(self.rng.integers(0, len(anchor_records)))])
            params = self._sample_candidate_from_anchor(anchor)
            attempts += 1
            constraint_result = self._constraint_check(params)
            if not constraint_result.feasible:
                continue
            sig = self._recipe_signature(pd.Series(params))
            if sig in seen:
                continue
            records.append(params)
            seen.add(sig)

        if not records:
            raise RuntimeError("Failed to construct BayBE candidate pool.")
        return pd.DataFrame(records).reset_index(drop=True)

    def _split_campaign_parameters(self, candidate_pool: pd.DataFrame) -> Tuple[List[str], Dict[str, object]]:
        variable_keys: List[str] = []
        fixed_values: Dict[str, object] = {}
        for key in self.input_keys:
            unique_values = candidate_pool[key].dropna().unique().tolist()
            if len(unique_values) >= 2:
                variable_keys.append(key)
            elif unique_values:
                fixed_values[key] = unique_values[0]
        if not variable_keys:
            raise RuntimeError("BayBE candidate pool has no varying parameters to optimize.")
        return variable_keys, fixed_values

    def _build_searchspace(self, candidate_pool: pd.DataFrame) -> SearchSpace:
        variable_keys, fixed_values = self._split_campaign_parameters(candidate_pool)
        self.campaign_variable_keys = variable_keys
        self.campaign_fixed_values = fixed_values

        params = []
        for key in variable_keys:
            if key in {"Металл", "Лиганд", "Растворитель"}:
                values = sorted(candidate_pool[key].astype(str).unique().tolist())
                params.append(CategoricalParameter(name=key, values=values))
            else:
                values = sorted(pd.to_numeric(candidate_pool[key], errors="coerce").dropna().astype(float).unique().tolist())
                params.append(NumericalDiscreteParameter(name=key, values=values))
        return SearchSpace.from_dataframe(candidate_pool[variable_keys], parameters=params)

    def _build_objective(self, targets: Dict[str, float]) -> DesirabilityObjective:
        baybe_targets = [
            NumericalTarget.match_quadratic(name=target_name, match_value=target_value)
            for target_name, target_value in targets.items()
        ]
        return DesirabilityObjective(targets=baybe_targets, require_normalization=False)

    def _build_campaign(self, candidate_pool: pd.DataFrame, targets: Dict[str, float]) -> Campaign:
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=FPSRecommender(),
            recommender=BotorchRecommender(),
            switch_after=1,
        )
        return Campaign(
            searchspace=self._build_searchspace(candidate_pool),
            objective=self._build_objective(targets),
            recommender=recommender,
        )

    def _with_fixed_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        for key, value in self.campaign_fixed_values.items():
            if key not in enriched.columns:
                enriched[key] = value
        return enriched[self.input_keys]

    def _evaluate_recommendation(
        self,
        params: Dict[str, object],
        targets: Dict[str, float],
        *,
        batch_id: int,
        batch_position: int,
    ) -> Dict[str, object]:
        row = {key: params[key] for key in self.input_keys}
        row["backend"] = "baybe"
        row["batch_id"] = batch_id
        row["batch_position"] = batch_position
        row["selection_mode"] = "campaign_recommendation"
        row["acquisition"] = "baybe_campaign"
        constraint_result = self._constraint_check(params)
        row["feasible"] = bool(constraint_result.feasible)
        row["constraint_reasons"] = ";".join(constraint_result.reasons)
        if not constraint_result.feasible:
            row["score"] = 1e6
            row["utility"] = -25.0
            return row

        feature_row = self._build_feature_row(params)
        score = 0.0
        width_sum = 0.0
        for target_name, target_value in targets.items():
            prediction = self._predict_target(target_name, feature_row)
            row[f"Pred_{target_name}"] = prediction["mean"]
            row[f"Pred_{target_name}_lo"] = prediction["lo"]
            row[f"Pred_{target_name}_hi"] = prediction["hi"]
            row[f"Pred_{target_name}_width"] = prediction["width"]
            row[f"Pred_{target_name}_center"] = prediction["interval_center"]
            row[f"Pred_{target_name}_deviation"] = self._target_deviation(target_value, prediction["mean"])
            row[target_name] = prediction["mean"]
            score += self._target_score(target_value, prediction["mean"])
            width_sum += prediction["width"]

        e0_pred = row.get("Pred_E0, кДж/моль")
        if e0_pred is not None and not (E0_BOUNDS_KJ_MOL[0] <= e0_pred <= E0_BOUNDS_KJ_MOL[1]):
            row["feasible"] = False
            reasons = constraint_result.reasons + ["predicted_E0_out_of_bounds"]
            row["constraint_reasons"] = ";".join(reasons)
            row["score"] = 1e6
            row["utility"] = -25.0
            return row

        row["score"] = float(score)
        row["utility"] = float(-(score + 0.001 * width_sum))
        return row

    @staticmethod
    def _chemistry_key(row: pd.Series) -> str:
        return "|".join([str(row["Металл"]), str(row["Лиганд"]), str(row["Растворитель"])])

    @staticmethod
    def _process_key(row: pd.Series) -> str:
        return "|".join(
            [
                f"{float(row['Т.син., °С']):.1f}",
                f"{float(row['Т суш., °С']):.1f}",
                f"{float(row['Tрег, ᵒС']):.1f}",
            ]
        )

    def _recipe_distance(self, left: pd.Series, right: pd.Series) -> float:
        deltas = []
        for feature in self.numeric_input_keys:
            feature_range = self.diversity_ranges.get(feature, 1.0)
            if feature_range <= 0:
                feature_range = 1.0
            deltas.append(abs(float(left[feature]) - float(right[feature])) / feature_range)
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

        def accept(candidate: pd.Series, enforce_distance: bool) -> bool:
            sig = self._recipe_signature(candidate)
            if sig in seen_signatures:
                return False
            chemistry_key = str(candidate["chemistry_key"])
            process_key = str(candidate["process_key"])
            if chemistry_counts.get(chemistry_key, 0) >= max_per_chemistry:
                return False
            if process_counts.get(process_key, 0) >= max_per_process:
                return False
            if enforce_distance:
                for idx in selected_indices:
                    chosen = pool.iloc[idx]
                    if str(chosen["chemistry_key"]) != chemistry_key:
                        continue
                    if self._recipe_distance(candidate, chosen) < min_distance:
                        return False
            selected_indices.append(int(candidate.name))
            seen_signatures.add(sig)
            chemistry_counts[chemistry_key] = chemistry_counts.get(chemistry_key, 0) + 1
            process_counts[process_key] = process_counts.get(process_key, 0) + 1
            return True

        for enforce_distance in (True, False):
            for _, candidate in pool.iterrows():
                if len(selected_indices) >= shortlist_size:
                    break
                accept(candidate, enforce_distance=enforce_distance)
            if len(selected_indices) >= shortlist_size:
                break

        shortlist = pool.iloc[selected_indices].copy().reset_index(drop=True)
        shortlist["rank"] = np.arange(1, len(shortlist) + 1)
        shortlist["selection_mode"] = "diverse_shortlist"
        return shortlist

    def run_campaign(
        self,
        targets: Dict[str, float],
        *,
        shortlist_size: int,
        max_per_chemistry: int,
        max_per_process: int,
        min_distance: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Campaign]:
        if not targets:
            raise ValueError("Specify at least one target.")

        candidate_pool = self._build_candidate_pool(targets)
        campaign = self._build_campaign(candidate_pool, targets)
        seed_measurements = self._build_seed_measurements(targets)
        seed_payload = self._with_fixed_values(seed_measurements[self.campaign_variable_keys])
        for target_name in targets:
            seed_payload[target_name] = seed_measurements[target_name].to_numpy()
        campaign.add_measurements(
            seed_payload[self.campaign_variable_keys + list(targets.keys())],
            numerical_measurements_must_be_within_tolerance=False,
        )

        history_rows: List[dict] = []
        remaining = self.n_trials
        n_batches = int(math.ceil(self.n_trials / self.batch_size))
        for batch_id in range(1, n_batches + 1):
            current_batch = min(self.batch_size, remaining)
            recommended = campaign.recommend(batch_size=current_batch)
            if recommended.empty:
                break
            recommended_full = self._with_fixed_values(recommended.reset_index(drop=True))
            measured_rows: List[dict] = []
            for batch_position, (_, rec_row) in enumerate(recommended_full.iterrows(), start=1):
                params = {key: rec_row[key] for key in self.input_keys}
                evaluated = self._evaluate_recommendation(params, targets, batch_id=batch_id, batch_position=batch_position)
                history_rows.append(evaluated)
                measured_rows.append({key: evaluated[key] for key in self.input_keys + list(targets.keys())})
            measured_df = pd.DataFrame(measured_rows)
            campaign.add_measurements(
                measured_df[self.campaign_variable_keys + list(targets.keys())],
                numerical_measurements_must_be_within_tolerance=False,
            )
            remaining -= current_batch
            if remaining <= 0:
                break

        history = pd.DataFrame(history_rows)
        if history.empty:
            return history, history, campaign

        width_columns = [column for column in history.columns if column.endswith("_width")]
        history["interval_width_total"] = history[width_columns].sum(axis=1) if width_columns else np.nan
        history = history.sort_values(by=["feasible", "score", "interval_width_total"], ascending=[False, True, True]).reset_index(drop=True)
        shortlist = self._select_diverse_shortlist(
            history,
            shortlist_size=shortlist_size,
            max_per_chemistry=max_per_chemistry,
            max_per_process=max_per_process,
            min_distance=min_distance,
        )
        return shortlist, history, campaign


def default_campaign_paths(output_path: str) -> Tuple[str, str]:
    output = Path(output_path)
    stem = output.stem
    parent = output.parent
    history = parent / f"{stem}_history.csv"
    state = parent / f"{stem}_campaign.json"
    return str(history), str(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign-oriented inverse design via BayBE.")
    parser.add_argument("--E0", type=float, help="Target E0 (kJ/mol)")
    parser.add_argument("--x0", type=float, help="Target x0 (nm)")
    parser.add_argument("--Sme", type=float, help="Target Sme (m2/g)")
    parser.add_argument("--trials", type=int, default=24, help="Total number of campaign recommendations to generate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Recommendation batch size per campaign iteration.")
    parser.add_argument("--candidate-pool-size", type=int, default=240)
    parser.add_argument("--init-measurements", type=int, default=12)
    parser.add_argument("--template-limit", type=int, default=12)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--max-per-chemistry", type=int, default=4)
    parser.add_argument("--max-per-process", type=int, default=3)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models-dir", "--models", dest="models_dir", type=str, default="artifacts/forward_models")
    parser.add_argument("--output", type=str, default="artifacts/predictions_baybe.csv")
    parser.add_argument("--history-output", type=str, help="Optional path for full campaign history CSV.")
    parser.add_argument("--campaign-state", type=str, help="Optional path for serialized BayBE campaign state.")
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

    history_output, campaign_state = default_campaign_paths(args.output)
    if args.history_output:
        history_output = args.history_output
    if args.campaign_state:
        campaign_state = args.campaign_state

    optimizer = BaybeCampaignOptimizer(
        models_dir=args.models_dir,
        data_path=args.data,
        n_trials=args.trials,
        batch_size=args.batch_size,
        candidate_pool_size=args.candidate_pool_size,
        init_measurements=args.init_measurements,
        strict_validation=args.strict_validation,
        template_limit=args.template_limit,
    )
    shortlist, history, campaign = optimizer.run_campaign(
        targets,
        shortlist_size=args.shortlist_size,
        max_per_chemistry=args.max_per_chemistry,
        max_per_process=args.max_per_process,
        min_distance=args.min_distance,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    shortlist.to_csv(args.output, index=False)
    os.makedirs(os.path.dirname(history_output) or ".", exist_ok=True)
    history.to_csv(history_output, index=False)
    os.makedirs(os.path.dirname(campaign_state) or ".", exist_ok=True)
    Path(campaign_state).write_text(campaign.to_json(), encoding="utf-8")

    feasible_count = int(history["feasible"].sum()) if not history.empty else 0
    print(f"Feasible recommendations: {feasible_count}/{len(history)}")
    print(f"Saved shortlist ({len(shortlist)} candidates) to {args.output}")
    print(f"Saved campaign history ({len(history)} rows) to {history_output}")
    print(f"Saved campaign state to {campaign_state}")
    if not shortlist.empty:
        print(shortlist.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
