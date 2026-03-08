"""Shared inverse-design helpers and optimizer backends."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from bofire.data_models.acquisition_functions.api import qLogNEI
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.features.api import CategoricalInput, ContinuousInput, ContinuousOutput, DiscreteInput
from bofire.data_models.objectives.api import CloseToTargetObjective
from bofire.strategies.predictives.qparego import QparegoStrategy
from catboost import CatBoostRegressor

from .constants import (
    DEFAULT_STOICHIOMETRY_BOUNDS,
    E0_BOUNDS_KJ_MOL,
    FORWARD_MODEL_TARGETS,
    SOLVENT_BOILING_POINTS_C,
    STOICHIOMETRY_TARGETS,
)
from .data_processing import (
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


class AdsorbentOptimizerContext:
    def __init__(
        self,
        *,
        models_dir: str,
        data_path: str,
        n_trials: int = 200,
        strict_validation: bool = False,
    ) -> None:
        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
        self.strict_validation = strict_validation

        validation_mode = "strict" if strict_validation else "warn"
        self.df_ref = load_dataset(data_path, validation_mode=validation_mode)
        self.lookup_tables = build_lookup_tables(self.df_ref)
        self.models = self._load_models()
        self.uq_models = self._load_uq_models()
        self.target_feature_specs = self._load_feature_specs()
        self.required_model_features = sorted(
            {
                feature
                for spec in self.target_feature_specs.values()
                for feature in spec["selected_features"]
            }
        )
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
            safe_target = target.replace("/", "_").replace(" ", "_")
            target_models: List[CatBoostRegressor] = []
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

    def _load_feature_specs(self) -> Dict[str, Dict[str, List[str]]]:
        feature_meta_path = os.path.join(self.models_dir, "feature_meta.joblib")
        if os.path.exists(feature_meta_path):
            feature_meta = joblib.load(feature_meta_path)
            target_specs = feature_meta.get("targets", {})
            if target_specs:
                return {
                    target: {
                        "selected_features": list(spec.get("selected_features", [])),
                        "categorical_features": list(spec.get("categorical_features", [])),
                    }
                    for target, spec in target_specs.items()
                    if spec.get("selected_features")
                }
        return {
            target: {
                "selected_features": list(ensemble[0].feature_names_),
                "categorical_features": [],
            }
            for target, ensemble in self.models.items()
        }

    def _build_domain(self, targets: Dict[str, float]) -> Domain:
        outputs = []
        for target in FORWARD_MODEL_TARGETS:
            if target not in targets:
                continue
            outputs.append(
                ContinuousOutput(
                    key=target,
                    objective=CloseToTargetObjective(target_value=float(targets[target]), exponent=2.0, w=1.0),
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
            feature_range = float(numeric.max() - numeric.min()) if not numeric.empty else 0.0
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
        return (stoich["ratio"] * (1 - tolerance), stoich["ratio"] * (1 + tolerance))

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
        ratio = n_salt / n_acid if np.isfinite(n_salt) and np.isfinite(n_acid) and n_acid > 0 else np.nan
        ratio_lo, ratio_hi = self._get_ratio_bounds(metal, ligand)
        if not np.isfinite(ratio) or ratio < ratio_lo or ratio > ratio_hi:
            reasons.append("stoichiometry_out_of_bounds")

        return ConstraintResult(feasible=not reasons, reasons=reasons)

    def _target_score(self, target_value: float, prediction: float, exponent: float = 2.0) -> float:
        scale = max(abs(target_value), 1.0)
        return float(abs((prediction - target_value) / scale) ** exponent)

    def _target_deviation(self, target_value: float, prediction: float) -> float:
        scale = max(abs(target_value), 1.0)
        return float((prediction - target_value) / scale)

    def _build_candidate_pool(self, targets: Dict[str, float], max_templates: int = 24) -> pd.DataFrame:
        scored = self.feasible_reference_rows.copy()
        score = np.zeros(len(scored), dtype=float)
        for target_name, target_value in targets.items():
            if target_name not in scored.columns:
                continue
            observed = pd.to_numeric(scored[target_name], errors="coerce").to_numpy(dtype=float)
            scale = max(abs(target_value), 1.0)
            score += np.square((observed - target_value) / scale)
        scored["reference_score"] = score
        return scored.sort_values(["reference_score"]).head(max_templates).reset_index(drop=True)

    @staticmethod
    def _chemistry_key(row: pd.Series) -> str:
        return "|".join(str(row[key]) for key in ["Металл", "Лиганд", "Растворитель"])

    @staticmethod
    def _process_key(row: pd.Series) -> str:
        return "|".join(str(row[key]) for key in ["Т.син., °С", "Т суш., °С", "Tрег, ᵒС"])

    def _distance_within_chemistry(self, left: pd.Series, right: pd.Series) -> float:
        distance = 0.0
        for feature in self.numeric_input_keys:
            scale = self.diversity_ranges.get(feature, 1.0)
            distance += abs(float(left[feature]) - float(right[feature])) / scale
        return float(distance / len(self.numeric_input_keys))

    def _select_diverse_shortlist(
        self,
        pool: pd.DataFrame,
        *,
        shortlist_size: int,
        max_per_chemistry: int,
        max_per_process: int,
        min_distance: float,
    ) -> pd.DataFrame:
        if pool.empty:
            return pool.copy()
        pool = pool.copy().reset_index(drop=True)
        pool["chemistry_key"] = pool.apply(self._chemistry_key, axis=1)
        pool["process_key"] = pool.apply(self._process_key, axis=1)
        chemistry_counts: Dict[str, int] = {}
        process_counts: Dict[str, int] = {}
        selected_indices: List[int] = []

        def accept(candidate: pd.Series, *, enforce_chemistry: bool, enforce_process: bool, enforce_distance: bool) -> bool:
            chemistry_key = str(candidate["chemistry_key"])
            process_key = str(candidate["process_key"])
            if enforce_chemistry and chemistry_counts.get(chemistry_key, 0) >= max_per_chemistry:
                return False
            if enforce_process and process_counts.get(process_key, 0) >= max_per_process:
                return False
            if enforce_distance:
                for selected_idx in selected_indices:
                    selected = pool.iloc[selected_idx]
                    if str(selected["chemistry_key"]) != chemistry_key:
                        continue
                    if self._distance_within_chemistry(candidate, selected) < min_distance:
                        return False
            selected_indices.append(int(candidate.name))
            chemistry_counts[chemistry_key] = chemistry_counts.get(chemistry_key, 0) + 1
            process_counts[process_key] = process_counts.get(process_key, 0) + 1
            return True

        for enforce_chemistry in (True, False):
            for enforce_process in (True, False):
                for enforce_distance in (True, False):
                    for _, candidate in pool.iterrows():
                        if len(selected_indices) >= shortlist_size:
                            break
                        accept(
                            candidate,
                            enforce_chemistry=enforce_chemistry,
                            enforce_process=enforce_process,
                            enforce_distance=enforce_distance,
                        )
                    if len(selected_indices) >= shortlist_size:
                        break
                if len(selected_indices) >= shortlist_size:
                    break
            if len(selected_indices) >= shortlist_size:
                break

        shortlist = pool.iloc[selected_indices].copy().reset_index(drop=True)
        shortlist["rank"] = np.arange(1, len(shortlist) + 1)
        shortlist["selection_mode"] = "diverse_shortlist"
        return shortlist

    def _build_feature_row(self, params: Dict[str, object]) -> pd.DataFrame:
        combo_mask = (
            (self.feasible_reference_rows["Металл"].astype(str) == str(params["Металл"]))
            & (self.feasible_reference_rows["Лиганд"].astype(str) == str(params["Лиганд"]))
            & (self.feasible_reference_rows["Растворитель"].astype(str) == str(params["Растворитель"]))
        )
        if combo_mask.any():
            reference = self.feasible_reference_rows.loc[combo_mask].iloc[[0]].reset_index(drop=True)
        else:
            reference = self.feasible_reference_rows.iloc[[0]].reset_index(drop=True)
        merged = reference.copy()
        for key, value in params.items():
            merged.loc[:, key] = value
        merged = add_salt_mass_features(merged, inplace=False)
        merged = add_physicochemical_descriptors(merged, inplace=False)
        merged = add_interaction_features(merged, inplace=False)

        for feature in self.required_model_features:
            if feature not in merged.columns:
                merged[feature] = np.nan
        return merged

    def _predict_target(self, target_name: str, feature_row: pd.DataFrame) -> Dict[str, float]:
        feature_spec = self.target_feature_specs[target_name]
        selected_features = feature_spec["selected_features"]
        categorical_features = set(feature_spec["categorical_features"])
        model_input = feature_row.copy()
        for feature in selected_features:
            if feature not in model_input.columns:
                model_input[feature] = np.nan
        model_input = model_input[selected_features].copy()
        for feature in categorical_features:
            if feature in model_input.columns:
                model_input[feature] = model_input[feature].astype(str)

        ensemble = self.models[target_name]
        predictions = np.asarray([model.predict(model_input)[0] for model in ensemble], dtype=float)
        mean = float(predictions.mean())
        uq_model = self.uq_models[target_name]
        interval_center, interval_bounds = uq_model.predict_interval(model_input, aggregate_predictions="mean")
        interval_bounds = np.asarray(interval_bounds, dtype=float).squeeze(0).squeeze(-1)
        lo = float(interval_bounds[0])
        hi = float(interval_bounds[1])
        return {
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "width": float(hi - lo),
            "interval_center": float(np.asarray(interval_center, dtype=float).reshape(-1)[0]),
        }

    def _nearest_feasible_reference(self, params: Dict[str, object]) -> Optional[pd.Series]:
        candidates = self.feasible_reference_rows
        metal = str(params["Металл"])
        ligand = str(params["Лиганд"])
        solvent = str(params["Растворитель"])
        selectors = [
            (candidates["Металл"].astype(str) == metal)
            & (candidates["Лиганд"].astype(str) == ligand)
            & (candidates["Растворитель"].astype(str) == solvent),
            (candidates["Металл"].astype(str) == metal) & (candidates["Лиганд"].astype(str) == ligand),
            candidates["Металл"].astype(str) == metal,
            pd.Series([True] * len(candidates)),
        ]
        subset = None
        for mask in selectors:
            current = candidates.loc[mask]
            if not current.empty:
                subset = current.copy()
                break
        if subset is None or subset.empty:
            return None
        score = np.zeros(len(subset), dtype=float)
        for feature in self.numeric_input_keys:
            scale = self.diversity_ranges.get(feature, 1.0)
            observed = pd.to_numeric(subset[feature], errors="coerce").to_numpy(dtype=float)
            score += np.abs(observed - float(params[feature])) / scale
        subset["_distance"] = score
        return subset.sort_values("_distance").iloc[0]

    def _project_to_feasible_candidate(self, params: Dict[str, object]) -> Tuple[Dict[str, object], str]:
        projected = {key: params[key] for key in self.input_keys}
        mode = "native"
        constraint_result = self._constraint_check(projected)
        if constraint_result.feasible:
            return projected, mode
        reference = self._nearest_feasible_reference(projected)
        if reference is None:
            return projected, "native_infeasible"
        for key in self.input_keys:
            projected[key] = reference[key]
        return projected, "nearest_feasible_reference"

    def _score_candidate(self, params: Dict[str, object], targets: Dict[str, float]) -> Dict[str, object]:
        row: Dict[str, object] = {key: params[key] for key in self.input_keys}
        constraint_result = self._constraint_check(params)
        row["feasible"] = bool(constraint_result.feasible)
        row["constraint_reasons"] = ";".join(constraint_result.reasons)
        if not constraint_result.feasible:
            row["score"] = 1e6
            return row
        feature_row = self._build_feature_row(params)
        score = 0.0
        interval_width_total = 0.0
        fantasy_outputs: Dict[str, float] = {}
        for target_name, target_value in targets.items():
            prediction = self._predict_target(target_name, feature_row)
            fantasy_outputs[target_name] = prediction["mean"]
            row[f"Pred_{target_name}"] = prediction["mean"]
            row[f"Pred_{target_name}_lo"] = prediction["lo"]
            row[f"Pred_{target_name}_hi"] = prediction["hi"]
            row[f"Pred_{target_name}_width"] = prediction["width"]
            row[f"Pred_{target_name}_center"] = prediction["interval_center"]
            row[f"Pred_{target_name}_deviation"] = self._target_deviation(target_value, prediction["mean"])
            interval_width_total += prediction["width"]
            score += self._target_score(target_value, prediction["mean"])
        e0_pred = fantasy_outputs.get("E0, кДж/моль")
        if e0_pred is not None and not (E0_BOUNDS_KJ_MOL[0] <= e0_pred <= E0_BOUNDS_KJ_MOL[1]):
            row["feasible"] = False
            reasons = [reason for reason in row["constraint_reasons"].split(";") if reason]
            reasons.append("predicted_E0_out_of_bounds")
            row["constraint_reasons"] = ";".join(reasons)
            row["score"] = 1e6
            return row
        row["score"] = float(score)
        row["interval_width_total"] = float(interval_width_total)
        row["_fantasy_outputs"] = fantasy_outputs
        return row


class LegacyOptunaBofireOptimizer(AdsorbentOptimizerContext):
    def _sample_near_anchor(self, trial: optuna.Trial, *, name: str, anchor: float, lower: float, upper: float, log: bool = False) -> float:
        if lower >= upper:
            return float(lower)
        span = upper - lower
        local_lower = max(lower, anchor - 0.35 * span)
        local_upper = min(upper, anchor + 0.35 * span)
        if local_lower >= local_upper:
            local_lower, local_upper = lower, upper
        return float(trial.suggest_float(name, local_lower, local_upper, log=log))

    def _sample_structured_params(self, trial: optuna.Trial, candidate_pool: pd.DataFrame) -> Dict[str, object]:
        anchor_idx = int(trial.suggest_categorical("anchor_idx", candidate_pool.index.tolist()))
        anchor = candidate_pool.loc[anchor_idx]
        combo_key = str(anchor["combo_key"])
        combo_stats = self.combo_reference_stats[combo_key]
        metal = str(anchor["Металл"])
        ligand = str(anchor["Лиганд"])
        params: Dict[str, object] = {"Металл": metal, "Лиганд": ligand}
        solvent_choices = combo_stats["solvents"]
        solvent_rank = trial.suggest_int("solvent_rank", 0, max(0, self.max_combo_solvents - 1))
        params["Растворитель"] = solvent_choices[solvent_rank % len(solvent_choices)]
        temp_candidates = [record for record in combo_stats["temp_triplets"] if str(record["Растворитель"]) == str(params["Растворитель"])]
        if not temp_candidates:
            temp_candidates = combo_stats["temp_triplets"]
        temp_choice_idx = int(trial.suggest_int("temp_template_rank", 0, max(0, self.max_temp_templates - 1))) % len(temp_candidates)
        temp_choice = temp_candidates[temp_choice_idx]
        params["Т.син., °С"] = float(temp_choice["Т.син., °С"])
        params["Т суш., °С"] = float(temp_choice["Т суш., °С"])
        params["Tрег, ᵒС"] = float(temp_choice["Tрег, ᵒС"])

        ratio_lo, ratio_hi = self._get_ratio_bounds(metal, ligand)
        anchor_ratio = float(anchor.get("R_molar", (ratio_lo + ratio_hi) / 2.0))
        ratio = self._sample_near_anchor(
            trial, name="R_molar_target", anchor=anchor_ratio, lower=ratio_lo, upper=ratio_hi,
        )

        metal_row = self.lookup_tables.metal.loc[metal]
        ligand_row = self.lookup_tables.ligand.loc[ligand]
        if isinstance(metal_row, pd.DataFrame):
            metal_row = metal_row.iloc[0]
        if isinstance(ligand_row, pd.DataFrame):
            ligand_row = ligand_row.iloc[0]
        mw_salt = float(metal_row["Молярка_соли"])
        mw_acid = float(ligand_row["Молярка_кислоты"])

        domain = self._build_domain({"E0, кДж/моль": 0.0, "х0, нм": 0.0, "Sme, м2/г": 0.0})
        salt_bounds = next(feature.bounds for feature in domain.inputs.features if feature.key == "m (соли), г")
        acid_bounds = next(feature.bounds for feature in domain.inputs.features if feature.key == "m(кис-ты), г")
        volume_bounds = next(feature.bounds for feature in domain.inputs.features if feature.key == "Vсин. (р-ля), мл")
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

    def optimize(self, targets: Dict[str, float], *, shortlist_size: int, max_per_chemistry: int, max_per_process: int, min_distance: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        candidate_pool = self._build_candidate_pool(targets)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        warm_start_count = min(12, len(candidate_pool))

        def objective(trial: optuna.Trial) -> float:
            if trial.number < warm_start_count:
                anchor = candidate_pool.iloc[trial.number]
                params = {key: anchor[key] for key in self.input_keys}
            else:
                params = self._sample_structured_params(trial, candidate_pool)
            for key, value in params.items():
                trial.set_user_attr(f"Param_{key}", value)
            result = self._score_candidate(params, targets)
            trial.set_user_attr("feasible", result["feasible"])
            trial.set_user_attr("constraint_reasons", result["constraint_reasons"].split(";") if result["constraint_reasons"] else [])
            for key, value in result.items():
                if key.startswith("Pred_"):
                    trial.set_user_attr(key, value)
            return float(result["score"])

        study.optimize(objective, n_trials=self.n_trials)
        rows = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {key.removeprefix("Param_"): value for key, value in trial.user_attrs.items() if key.startswith("Param_")}
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
        width_columns = [column for column in results.columns if column.endswith("_width")]
        if width_columns:
            results["interval_width_total"] = results[width_columns].sum(axis=1)
        results = results.sort_values(by=["feasible", "score", "interval_width_total"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
        results["backend"] = "bofire_optuna_legacy"
        feasible_count = int(results["feasible"].sum())
        print(f"Feasible candidates: {feasible_count}/{len(results)}")
        shortlist = self._select_diverse_shortlist(results, shortlist_size=shortlist_size, max_per_chemistry=max_per_chemistry, max_per_process=max_per_process, min_distance=min_distance)
        return shortlist, results


class NativeBofireOptimizer(AdsorbentOptimizerContext):
    def _build_initial_experiments(self, targets: Dict[str, float]) -> pd.DataFrame:
        max_initial_experiments = 4
        columns = self.input_keys + list(targets.keys())
        experiments = self.feasible_reference_rows[columns].copy()
        experiments = experiments.dropna(subset=list(targets.keys())).reset_index(drop=True)
        if len(experiments) <= max_initial_experiments:
            return experiments

        experiments = experiments.copy()
        experiments["_row_id"] = np.arange(len(experiments))
        experiments["chemistry_key"] = (
            experiments["Металл"].astype(str)
            + "|"
            + experiments["Лиганд"].astype(str)
            + "|"
            + experiments["Растворитель"].astype(str)
        )
        experiments["process_key"] = (
            experiments["Т.син., °С"].astype(str)
            + "|"
            + experiments["Т суш., °С"].astype(str)
            + "|"
            + experiments["Tрег, ᵒС"].astype(str)
        )

        selected_parts = [
            group.sample(n=1, random_state=42)
            for _, group in experiments.groupby(["chemistry_key", "process_key"], sort=True)
        ]
        selected = pd.concat(selected_parts, ignore_index=True)
        if len(selected) < max_initial_experiments:
            remainder = experiments.loc[~experiments["_row_id"].isin(selected["_row_id"])].copy()
            needed = min(max_initial_experiments - len(selected), len(remainder))
            if needed > 0:
                selected = pd.concat(
                    [selected, remainder.sample(n=needed, random_state=42)],
                    ignore_index=True,
                )
        elif len(selected) > max_initial_experiments:
            selected = selected.sample(n=max_initial_experiments, random_state=42).reset_index(drop=True)

        return selected[columns].reset_index(drop=True)

    def optimize(self, targets: Dict[str, float], *, shortlist_size: int, max_per_chemistry: int, max_per_process: int, min_distance: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not targets:
            raise ValueError("Specify at least one target.")
        domain = self._build_domain(targets)
        strategy = QparegoStrategy.make(domain=domain, acquisition_function=qLogNEI(), seed=42)
        initial_experiments = self._build_initial_experiments(targets)
        strategy.tell(initial_experiments, replace=True)

        rows: List[Dict[str, object]] = []
        seen_keys: set[Tuple[object, ...]] = set()
        ask_budget = max(self.n_trials * 3, self.n_trials + 10)
        ask_count = 0
        while len(rows) < self.n_trials and ask_count < ask_budget:
            ask_count += 1
            candidates = strategy.ask(candidate_count=1, add_pending=False)
            params = candidates.iloc[0][self.input_keys].to_dict()
            projected_params, projection_mode = self._project_to_feasible_candidate(params)
            key = tuple(projected_params[column] for column in self.input_keys)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            result = self._score_candidate(projected_params, targets)
            result["backend"] = "bofire"
            result["projection_mode"] = projection_mode
            result["search_rank"] = len(rows) + 1
            for column in self.input_keys:
                result[column] = projected_params[column]
            rows.append(result)

            if result["feasible"]:
                fantasy_row = {column: projected_params[column] for column in self.input_keys}
                fantasy_row.update(result["_fantasy_outputs"])
                strategy.tell(pd.DataFrame([fantasy_row]), replace=False)

        results = pd.DataFrame(rows)
        if results.empty:
            return results, results
        if "_fantasy_outputs" in results.columns:
            results = results.drop(columns=["_fantasy_outputs"])
        if "interval_width_total" not in results.columns:
            width_columns = [column for column in results.columns if column.endswith("_width")]
            if width_columns:
                results["interval_width_total"] = results[width_columns].sum(axis=1)
        results = results.sort_values(by=["feasible", "score", "interval_width_total"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
        feasible_count = int(results["feasible"].sum())
        print(f"Feasible candidates: {feasible_count}/{len(results)}")
        shortlist = self._select_diverse_shortlist(results, shortlist_size=shortlist_size, max_per_chemistry=max_per_chemistry, max_per_process=max_per_process, min_distance=min_distance)
        return shortlist, results
