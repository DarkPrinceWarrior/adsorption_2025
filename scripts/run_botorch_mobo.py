#!/usr/bin/env python3
"""Research-grade inverse design using BoTorch-guided local search over feasible templates."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from catboost import CatBoostRegressor
from gpytorch.mlls import ExactMarginalLogLikelihood

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


TORCH_DTYPE = torch.double


@dataclass
class ConstraintResult:
    feasible: bool
    reasons: List[str]


class BotorchAdsorbentOptimizer:
    def __init__(
        self,
        models_dir: str,
        data_path: str,
        n_trials: int = 200,
        strict_validation: bool = False,
        template_limit: int = 12,
    ):
        self.models_dir = models_dir
        self.data_path = data_path
        self.n_trials = n_trials
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

    def _build_candidate_pool(self, targets: Dict[str, float], max_templates: int = 64) -> pd.DataFrame:
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

    def _template_dataframe(self, candidate_pool: pd.DataFrame) -> pd.DataFrame:
        template_cols = ["Металл", "Лиганд", "Растворитель", "Т.син., °С", "Т суш., °С", "Tрег, ᵒС"]
        templates = (
            candidate_pool
            .sort_values("reference_score")
            .drop_duplicates(subset=template_cols, keep="first")
            .reset_index(drop=True)
        )
        max_templates = min(len(templates), self.template_limit, max(1, self.n_trials // 4))
        return templates.head(max_templates).reset_index(drop=True)

    def _template_budgets(self, n_templates: int) -> List[int]:
        base = max(4, self.n_trials // max(n_templates, 1))
        budgets = [base for _ in range(n_templates)]
        total = sum(budgets)
        idx = 0
        while total < self.n_trials:
            budgets[idx % n_templates] += 1
            total += 1
            idx += 1
        while total - max(budgets) >= self.n_trials:
            max_idx = int(np.argmax(budgets))
            if budgets[max_idx] <= 4:
                break
            budgets[max_idx] -= 1
            total -= 1
        return budgets

    def _template_search_box(self, anchor: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        metal = str(anchor["Металл"])
        ligand = str(anchor["Лиганд"])
        ratio_lo, ratio_hi = self._get_ratio_bounds(metal, ligand)
        anchor_ratio = float(anchor.get("R_molar", (ratio_lo + ratio_hi) / 2.0))
        anchor_ratio = float(np.clip(anchor_ratio, ratio_lo, ratio_hi))

        acid_anchor = float(anchor["m(кис-ты), г"])
        volume_anchor = float(anchor["Vсин. (р-ля), мл"])

        acid_lo = max(self.acid_bounds[0], acid_anchor * 0.45)
        acid_hi = min(self.acid_bounds[1], acid_anchor * 2.20)
        if acid_lo >= acid_hi:
            acid_lo, acid_hi = self.acid_bounds

        volume_lo = max(self.volume_bounds[0], volume_anchor * 0.50)
        volume_hi = min(self.volume_bounds[1], volume_anchor * 1.80)
        if volume_lo >= volume_hi:
            volume_lo, volume_hi = self.volume_bounds

        ratio_span = ratio_hi - ratio_lo
        local_ratio_lo = max(ratio_lo, anchor_ratio - 0.45 * ratio_span)
        local_ratio_hi = min(ratio_hi, anchor_ratio + 0.45 * ratio_span)
        if local_ratio_lo >= local_ratio_hi:
            local_ratio_lo, local_ratio_hi = ratio_lo, ratio_hi

        lower = np.asarray([acid_lo, volume_lo, local_ratio_lo], dtype=float)
        upper = np.asarray([acid_hi, volume_hi, local_ratio_hi], dtype=float)
        anchor_vec = np.asarray([acid_anchor, volume_anchor, anchor_ratio], dtype=float)
        anchor_vec = np.clip(anchor_vec, lower, upper)
        return lower, upper, anchor_vec

    def _vector_to_params(self, anchor: pd.Series, vector: np.ndarray) -> Dict[str, object]:
        metal = str(anchor["Металл"])
        ligand = str(anchor["Лиганд"])
        solvent = str(anchor["Растворитель"])

        metal_row = self.lookup_tables.metal.loc[metal]
        ligand_row = self.lookup_tables.ligand.loc[ligand]
        if isinstance(metal_row, pd.DataFrame):
            metal_row = metal_row.iloc[0]
        if isinstance(ligand_row, pd.DataFrame):
            ligand_row = ligand_row.iloc[0]
        mw_salt = float(metal_row["Молярка_соли"])
        mw_acid = float(ligand_row["Молярка_кислоты"])

        acid_mass = float(vector[0])
        volume = float(vector[1])
        ratio = float(vector[2])
        salt_mass = ratio * acid_mass * mw_salt / mw_acid

        return {
            "Металл": metal,
            "Лиганд": ligand,
            "Растворитель": solvent,
            "m (соли), г": float(salt_mass),
            "m(кис-ты), г": acid_mass,
            "Vсин. (р-ля), мл": volume,
            "Т.син., °С": float(anchor["Т.син., °С"]),
            "Т суш., °С": float(anchor["Т суш., °С"]),
            "Tрег, ᵒС": float(anchor["Tрег, ᵒС"]),
        }

    @staticmethod
    def _normalize(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        span = np.maximum(upper - lower, 1e-9)
        return (points - lower) / span

    @staticmethod
    def _denormalize(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        span = np.maximum(upper - lower, 1e-9)
        return lower + points * span

    def _evaluate_candidate(
        self,
        params: Dict[str, object],
        targets: Dict[str, float],
        *,
        backend: str,
        source_template_rank: int,
        iteration: int,
        acquisition: str,
    ) -> Dict[str, object]:
        row = {key: params[key] for key in self.input_keys}
        row["backend"] = backend
        row["source_template_rank"] = source_template_rank
        row["iteration"] = iteration
        row["acquisition"] = acquisition

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

    def _propose_botorch_candidate(
        self,
        train_X: np.ndarray,
        train_utility: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Optional[np.ndarray]:
        if len(train_X) < 3:
            return None
        if np.ptp(train_utility) < 1e-8:
            return None

        try:
            norm_X = self._normalize(train_X, lower, upper)
            train_X_t = torch.tensor(norm_X, dtype=TORCH_DTYPE)
            train_Y_t = torch.tensor(train_utility[:, None], dtype=TORCH_DTYPE)
            model = SingleTaskGP(train_X_t, train_Y_t, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acqf = LogExpectedImprovement(model=model, best_f=train_Y_t.max())
            bounds = torch.stack(
                [
                    torch.zeros(train_X.shape[1], dtype=TORCH_DTYPE),
                    torch.ones(train_X.shape[1], dtype=TORCH_DTYPE),
                ]
            )
            candidate_t, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=8,
                raw_samples=128,
                options={"batch_limit": 4, "maxiter": 200},
            )
            candidate = candidate_t.detach().cpu().numpy().reshape(-1)
            return self._denormalize(candidate, lower, upper)
        except Exception:
            return None

    def _random_candidate(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return self.rng.uniform(lower, upper)

    def _run_local_template_search(
        self,
        anchor: pd.Series,
        targets: Dict[str, float],
        *,
        template_rank: int,
        budget: int,
    ) -> List[Dict[str, object]]:
        lower, upper, anchor_vec = self._template_search_box(anchor)
        vectors: List[np.ndarray] = []
        rows: List[Dict[str, object]] = []
        signatures = set()

        def evaluate_vector(vector: np.ndarray, acquisition: str) -> None:
            params = self._vector_to_params(anchor, vector)
            row = self._evaluate_candidate(
                params,
                targets,
                backend="botorch",
                source_template_rank=template_rank,
                iteration=len(rows),
                acquisition=acquisition,
            )
            signature = self._recipe_signature(pd.Series(row))
            if signature in signatures:
                return
            signatures.add(signature)
            vectors.append(vector)
            rows.append(row)

        evaluate_vector(anchor_vec, "anchor")
        initial_random = max(0, min(3, budget - 1))
        for _ in range(initial_random):
            evaluate_vector(self._random_candidate(lower, upper), "random_init")

        while len(rows) < budget:
            train_X = np.asarray(vectors, dtype=float)
            train_utility = np.asarray([float(row["utility"]) for row in rows], dtype=float)
            proposal = self._propose_botorch_candidate(train_X, train_utility, lower, upper)
            acquisition = "botorch_ei"
            if proposal is None:
                proposal = self._random_candidate(lower, upper)
                acquisition = "random_fallback"
            for _ in range(5):
                params = self._vector_to_params(anchor, proposal)
                signature = self._recipe_signature(pd.Series(params))
                if signature not in signatures:
                    break
                proposal = self._random_candidate(lower, upper)
                acquisition = "random_retry"
            evaluate_vector(proposal, acquisition)

        return rows

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
        templates = self._template_dataframe(candidate_pool)
        budgets = self._template_budgets(len(templates))

        rows: List[Dict[str, object]] = []
        for template_rank, (_, anchor) in enumerate(templates.iterrows(), start=1):
            rows.extend(
                self._run_local_template_search(
                    anchor,
                    targets,
                    template_rank=template_rank,
                    budget=budgets[template_rank - 1],
                )
            )

        results = pd.DataFrame(rows)
        if results.empty:
            return results, results

        width_columns = [column for column in results.columns if column.endswith("_width")]
        if width_columns:
            results["interval_width_total"] = results[width_columns].sum(axis=1)
        else:
            results["interval_width_total"] = np.nan

        results = results.sort_values(
            by=["feasible", "score", "interval_width_total"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        results["search_rank"] = np.arange(1, len(results) + 1)
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
    parser = argparse.ArgumentParser(description="Research-grade inverse design via BoTorch-guided local BO.")
    parser.add_argument("--E0", type=float, help="Target E0 (kJ/mol)")
    parser.add_argument("--x0", type=float, help="Target x0 (nm)")
    parser.add_argument("--Sme", type=float, help="Target Sme (m2/g)")
    parser.add_argument("--trials", type=int, default=240)
    parser.add_argument("--template-limit", type=int, default=12)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--max-per-chemistry", type=int, default=4)
    parser.add_argument("--max-per-process", type=int, default=3)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models-dir", "--models", dest="models_dir", type=str, default="artifacts/forward_models")
    parser.add_argument("--output", type=str, default="artifacts/predictions_botorch.csv")
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

    optimizer = BotorchAdsorbentOptimizer(
        models_dir=args.models_dir,
        data_path=args.data,
        n_trials=args.trials,
        strict_validation=args.strict_validation,
        template_limit=args.template_limit,
    )
    shortlist, all_results = optimizer.optimize(
        targets,
        shortlist_size=args.shortlist_size,
        max_per_chemistry=args.max_per_chemistry,
        max_per_process=args.max_per_process,
        min_distance=args.min_distance,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    shortlist.to_csv(args.output, index=False)
    if args.all_output:
        os.makedirs(os.path.dirname(args.all_output) or ".", exist_ok=True)
        all_results.to_csv(args.all_output, index=False)
        print(f"Saved full search pool ({len(all_results)} candidates) to {args.all_output}")
    print(f"Saved shortlist ({len(shortlist)} candidates) to {args.output}")
    if not shortlist.empty:
        print(shortlist.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
