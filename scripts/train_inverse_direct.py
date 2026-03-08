#!/usr/bin/env python3
"""Train and evaluate a direct-inverse baseline: target properties -> recipe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "..", "src"))

from run_bofire_opt import BofireAdsorbentOptimizer  # noqa: E402

from adsorb_synthesis.constants import (  # noqa: E402
    FORWARD_MODEL_TARGETS,
    RANDOM_SEED,
)
from adsorb_synthesis.data_processing import load_dataset  # noqa: E402


TARGET_COLUMNS = list(FORWARD_MODEL_TARGETS)
CATEGORICAL_RECIPE_COLUMNS = ["Металл", "Лиганд", "Растворитель"]
NUMERIC_RECIPE_COLUMNS = [
    "m (соли), г",
    "m(кис-ты), г",
    "Vсин. (р-ля), мл",
    "Т.син., °С",
    "Т суш., °С",
    "Tрег, ᵒС",
]
RECIPE_COLUMNS = CATEGORICAL_RECIPE_COLUMNS + NUMERIC_RECIPE_COLUMNS
DISCRETE_TEMPERATURE_COLUMNS = ["Т.син., °С", "Т суш., °С", "Tрег, ᵒС"]


@dataclass
class FoldModels:
    categorical_models: Dict[str, RandomForestClassifier]
    numeric_models: Dict[str, RandomForestRegressor]
    numeric_feature_columns: List[str]


class DirectInverseBaseline:
    def __init__(
        self,
        *,
        data_path: str,
        models_dir: str,
        cv_folds: int = 5,
        strict_validation: bool = False,
    ) -> None:
        self.data_path = data_path
        self.models_dir = models_dir
        self.cv_folds = cv_folds
        self.strict_validation = strict_validation

        validation_mode = "strict" if strict_validation else "warn"
        self.df = load_dataset(data_path, validation_mode=validation_mode).reset_index(drop=True)
        self.forward_recheck = BofireAdsorbentOptimizer(
            models_dir=models_dir,
            data_path=data_path,
            n_trials=1,
            strict_validation=strict_validation,
        )
        self.target_df = self.df[TARGET_COLUMNS].copy()
        self.recipe_df = self.df[RECIPE_COLUMNS].copy()
        self.category_defaults = {
            column: str(self.df[column].mode(dropna=True).iloc[0])
            for column in CATEGORICAL_RECIPE_COLUMNS
        }
        self.numeric_bounds = {
            column: (
                float(pd.to_numeric(self.df[column], errors="coerce").min()),
                float(pd.to_numeric(self.df[column], errors="coerce").max()),
            )
            for column in NUMERIC_RECIPE_COLUMNS
        }
        self.discrete_values = {
            column: sorted(pd.to_numeric(self.df[column], errors="coerce").dropna().astype(float).unique().tolist())
            for column in DISCRETE_TEMPERATURE_COLUMNS
        }

    def _build_numeric_training_frame(
        self,
        targets: pd.DataFrame,
        categories: pd.DataFrame,
        *,
        reference_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        frame = pd.concat(
            [
                targets.reset_index(drop=True),
                pd.get_dummies(categories.reset_index(drop=True), prefix=CATEGORICAL_RECIPE_COLUMNS, dtype=float),
            ],
            axis=1,
        )
        if reference_columns is None:
            return frame, list(frame.columns)
        return frame.reindex(columns=reference_columns, fill_value=0.0), reference_columns

    def _fit_fold_models(self, train_idx: np.ndarray) -> FoldModels:
        x_train = self.target_df.iloc[train_idx].reset_index(drop=True)
        y_train = self.recipe_df.iloc[train_idx].reset_index(drop=True)

        categorical_models: Dict[str, RandomForestClassifier] = {}
        for column in CATEGORICAL_RECIPE_COLUMNS:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(x_train, y_train[column].astype(str))
            categorical_models[column] = model

        numeric_train_frame, numeric_feature_columns = self._build_numeric_training_frame(
            x_train,
            y_train[CATEGORICAL_RECIPE_COLUMNS],
        )
        numeric_models: Dict[str, RandomForestRegressor] = {}
        for column in NUMERIC_RECIPE_COLUMNS:
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(numeric_train_frame, pd.to_numeric(y_train[column], errors="coerce"))
            numeric_models[column] = model

        return FoldModels(
            categorical_models=categorical_models,
            numeric_models=numeric_models,
            numeric_feature_columns=numeric_feature_columns,
        )

    def _predict_recipe_frame(self, models: FoldModels, x_target: pd.DataFrame) -> pd.DataFrame:
        x_frame = x_target.reset_index(drop=True)
        categories = pd.DataFrame(index=x_frame.index)
        for column, model in models.categorical_models.items():
            categories[column] = model.predict(x_frame)

        numeric_frame, _ = self._build_numeric_training_frame(
            x_frame,
            categories,
            reference_columns=models.numeric_feature_columns,
        )
        predictions = categories.copy()
        for column, model in models.numeric_models.items():
            predictions[column] = model.predict(numeric_frame)
        return predictions[RECIPE_COLUMNS]

    def _snap_to_discrete(self, column: str, value: float) -> float:
        allowed = self.discrete_values[column]
        if not allowed:
            return float(value)
        return float(min(allowed, key=lambda candidate: abs(candidate - float(value))))

    def _find_nearest_reference(
        self,
        params: Dict[str, object],
        *,
        require_exact_solvent: bool,
    ) -> Optional[pd.Series]:
        candidates = self.forward_recheck.feasible_reference_rows
        metal = str(params["Металл"])
        ligand = str(params["Лиганд"])
        solvent = str(params["Растворитель"])

        selectors = []
        if require_exact_solvent:
            selectors.append(
                (candidates["Металл"].astype(str) == metal)
                & (candidates["Лиганд"].astype(str) == ligand)
                & (candidates["Растворитель"].astype(str) == solvent)
            )
        selectors.extend(
            [
                (candidates["Металл"].astype(str) == metal) & (candidates["Лиганд"].astype(str) == ligand),
                candidates["Металл"].astype(str) == metal,
                pd.Series([True] * len(candidates)),
            ]
        )

        best_subset: Optional[pd.DataFrame] = None
        for mask in selectors:
            subset = candidates.loc[mask]
            if not subset.empty:
                best_subset = subset
                break
        if best_subset is None or best_subset.empty:
            return None

        scoring = best_subset.copy()
        score = np.zeros(len(scoring), dtype=float)
        for column in NUMERIC_RECIPE_COLUMNS:
            lower, upper = self.numeric_bounds[column]
            scale = max(upper - lower, 1e-6)
            observed = pd.to_numeric(scoring[column], errors="coerce").to_numpy(dtype=float)
            score += np.abs(observed - float(params[column])) / scale
        scoring["_distance"] = score
        return scoring.sort_values("_distance").iloc[0]

    def _repair_recipe(self, raw_params: Dict[str, object]) -> Tuple[Dict[str, object], str]:
        params = {key: raw_params[key] for key in RECIPE_COLUMNS}
        projection_mode = "model_predicted"

        for column in CATEGORICAL_RECIPE_COLUMNS:
            value = str(params[column])
            reference_values = self.df[column].astype(str)
            if value not in set(reference_values):
                params[column] = self.category_defaults[column]
                projection_mode = "category_fallback"

        for column in ["m(кис-ты), г", "Vсин. (р-ля), мл"]:
            lower, upper = self.numeric_bounds[column]
            params[column] = float(np.clip(float(params[column]), lower, upper))

        for column in DISCRETE_TEMPERATURE_COLUMNS:
            params[column] = self._snap_to_discrete(column, float(params[column]))

        nearest_reference = self._find_nearest_reference(params, require_exact_solvent=True)
        if nearest_reference is not None:
            params["Т.син., °С"] = float(nearest_reference["Т.син., °С"])
            params["Т суш., °С"] = float(nearest_reference["Т суш., °С"])
            params["Tрег, ᵒС"] = float(nearest_reference["Tрег, ᵒС"])
            projection_mode = "template_projection"

        metal = str(params["Металл"])
        ligand = str(params["Лиганд"])
        metal_row = self.forward_recheck.lookup_tables.metal.loc[metal]
        ligand_row = self.forward_recheck.lookup_tables.ligand.loc[ligand]
        if isinstance(metal_row, pd.DataFrame):
            metal_row = metal_row.iloc[0]
        if isinstance(ligand_row, pd.DataFrame):
            ligand_row = ligand_row.iloc[0]
        mw_salt = float(metal_row["Молярка_соли"])
        mw_acid = float(ligand_row["Молярка_кислоты"])

        acid_lower, acid_upper = self.numeric_bounds["m(кис-ты), г"]
        salt_lower, salt_upper = self.numeric_bounds["m (соли), г"]
        acid_mass = float(np.clip(float(params["m(кис-ты), г"]), acid_lower, acid_upper))

        predicted_ratio = np.nan
        if float(params["m(кис-ты), г"]) > 0 and mw_salt and mw_acid:
            n_salt_pred = float(params["m (соли), г"]) / mw_salt
            n_acid_pred = float(params["m(кис-ты), г"]) / mw_acid
            if np.isfinite(n_salt_pred) and np.isfinite(n_acid_pred) and n_acid_pred > 0:
                predicted_ratio = n_salt_pred / n_acid_pred
        ratio_lower, ratio_upper = self.forward_recheck._get_ratio_bounds(metal, ligand)
        if not np.isfinite(predicted_ratio):
            predicted_ratio = 0.5 * (ratio_lower + ratio_upper)
        clipped_ratio = float(np.clip(predicted_ratio, ratio_lower, ratio_upper))
        salt_mass = clipped_ratio * acid_mass * mw_salt / mw_acid
        if salt_mass < salt_lower or salt_mass > salt_upper:
            salt_mass = float(np.clip(salt_mass, salt_lower, salt_upper))
            acid_mass = float(
                np.clip(
                    salt_mass / clipped_ratio * mw_acid / mw_salt,
                    acid_lower,
                    acid_upper,
                )
            )
            salt_mass = float(np.clip(clipped_ratio * acid_mass * mw_salt / mw_acid, salt_lower, salt_upper))
        params["m(кис-ты), г"] = acid_mass
        params["m (соли), г"] = salt_mass

        constraint_result = self.forward_recheck._constraint_check(params)
        if not constraint_result.feasible:
            fallback = self._find_nearest_reference(params, require_exact_solvent=False)
            if fallback is not None:
                for column in RECIPE_COLUMNS:
                    params[column] = fallback[column]
                projection_mode = "nearest_feasible_reference"

        for column in NUMERIC_RECIPE_COLUMNS:
            params[column] = float(params[column])
        return params, projection_mode

    def _recipe_numeric_error(self, actual: pd.Series, predicted: Dict[str, object]) -> float:
        errors = []
        for column in NUMERIC_RECIPE_COLUMNS:
            lower, upper = self.numeric_bounds[column]
            scale = max(upper - lower, 1e-6)
            errors.append(abs(float(actual[column]) - float(predicted[column])) / scale)
        return float(np.mean(errors))

    def _recheck_prediction(
        self,
        params: Dict[str, object],
        target_row: pd.Series,
    ) -> Dict[str, object]:
        record: Dict[str, object] = {}
        constraint_result = self.forward_recheck._constraint_check(params)
        record["feasible"] = bool(constraint_result.feasible)
        record["constraint_reasons"] = ";".join(constraint_result.reasons)
        if not constraint_result.feasible:
            record["score"] = 1e6
            return record

        feature_row = self.forward_recheck._build_feature_row(params)
        score = 0.0
        width_total = 0.0
        for target_name in TARGET_COLUMNS:
            prediction = self.forward_recheck._predict_target(target_name, feature_row)
            record[f"Pred_{target_name}"] = prediction["mean"]
            record[f"Pred_{target_name}_lo"] = prediction["lo"]
            record[f"Pred_{target_name}_hi"] = prediction["hi"]
            record[f"Pred_{target_name}_width"] = prediction["width"]
            record[f"Pred_{target_name}_center"] = prediction["interval_center"]
            deviation = self.forward_recheck._target_deviation(float(target_row[target_name]), prediction["mean"])
            record[f"Pred_{target_name}_deviation"] = deviation
            score += self.forward_recheck._target_score(float(target_row[target_name]), prediction["mean"])
            width_total += prediction["width"]
        record["interval_width_total"] = float(width_total)
        record["score"] = float(score)
        return record

    def evaluate_oof(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_SEED)
        records: List[Dict[str, object]] = []

        for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(self.target_df), start=1):
            models = self._fit_fold_models(train_idx)
            x_valid = self.target_df.iloc[valid_idx].reset_index(drop=True)
            y_valid = self.recipe_df.iloc[valid_idx].reset_index(drop=True)
            predicted_frame = self._predict_recipe_frame(models, x_valid)

            for row_offset in range(len(x_valid)):
                raw_prediction = predicted_frame.iloc[row_offset].to_dict()
                repaired_prediction, projection_mode = self._repair_recipe(raw_prediction)
                actual_recipe = y_valid.iloc[row_offset]
                actual_targets = x_valid.iloc[row_offset]

                record: Dict[str, object] = {
                    "backend": "inverse_direct",
                    "fold_id": fold_id,
                    "selection_mode": "direct_inverse_oof",
                    "search_rank": len(records) + 1,
                    "rank": len(records) + 1,
                    "projection_mode": projection_mode,
                }
                for column in TARGET_COLUMNS:
                    record[f"Target_{column}"] = float(actual_targets[column])
                for column in RECIPE_COLUMNS:
                    record[f"Actual_{column}"] = actual_recipe[column]
                    record[column] = repaired_prediction[column]
                for column in CATEGORICAL_RECIPE_COLUMNS:
                    record[f"{column}_match"] = int(str(actual_recipe[column]) == str(repaired_prediction[column]))
                record["recipe_match_count"] = int(sum(record[f"{column}_match"] for column in CATEGORICAL_RECIPE_COLUMNS))
                record["recipe_numeric_error"] = self._recipe_numeric_error(actual_recipe, repaired_prediction)
                record["chemistry_key"] = "|".join(str(repaired_prediction[column]) for column in CATEGORICAL_RECIPE_COLUMNS)
                record["process_key"] = "|".join(
                    f"{float(repaired_prediction[column]):.1f}" for column in DISCRETE_TEMPERATURE_COLUMNS
                )
                record.update(self._recheck_prediction(repaired_prediction, actual_targets))
                records.append(record)

        predictions = pd.DataFrame(records)
        predictions["search_rank"] = np.arange(1, len(predictions) + 1)
        predictions["rank"] = predictions["search_rank"]
        metrics = self._build_metrics(predictions)
        return predictions, metrics

    def _build_metrics(self, predictions: pd.DataFrame) -> Dict[str, object]:
        metrics: Dict[str, object] = {
            "backend": "inverse_direct",
            "cv_folds": self.cv_folds,
            "n_rows": int(len(predictions)),
            "feasibility_rate": float(predictions["feasible"].fillna(False).mean()) if len(predictions) else np.nan,
            "mean_score": float(pd.to_numeric(predictions["score"], errors="coerce").mean()) if len(predictions) else np.nan,
            "best_score": float(pd.to_numeric(predictions["score"], errors="coerce").min()) if len(predictions) else np.nan,
            "mean_interval_width_total": float(pd.to_numeric(predictions["interval_width_total"], errors="coerce").mean()) if "interval_width_total" in predictions.columns and len(predictions) else np.nan,
        }

        metrics["categorical_accuracy"] = {
            column: float(pd.to_numeric(predictions[f"{column}_match"], errors="coerce").mean())
            for column in CATEGORICAL_RECIPE_COLUMNS
        }
        metrics["recipe_numeric_error_mean"] = float(pd.to_numeric(predictions["recipe_numeric_error"], errors="coerce").mean())
        metrics["projection_mode_share"] = {
            str(mode): float(count / len(predictions))
            for mode, count in predictions["projection_mode"].value_counts(dropna=False).items()
        }
        metrics["target_mae_recheck"] = {}
        metrics["target_hit_rmse_recheck"] = {}
        for target_name in TARGET_COLUMNS:
            pred_col = f"Pred_{target_name}"
            target_col = f"Target_{target_name}"
            diff = pd.to_numeric(predictions[pred_col], errors="coerce") - pd.to_numeric(predictions[target_col], errors="coerce")
            metrics["target_mae_recheck"][target_name] = float(diff.abs().mean())
            metrics["target_hit_rmse_recheck"][target_name] = float(np.sqrt(np.nanmean(np.square(diff))))
        return metrics

    def fit_full_model(self) -> Dict[str, object]:
        full_idx = np.arange(len(self.df))
        models = self._fit_fold_models(full_idx)
        return {
            "backend": "inverse_direct",
            "target_columns": TARGET_COLUMNS,
            "recipe_columns": RECIPE_COLUMNS,
            "categorical_recipe_columns": CATEGORICAL_RECIPE_COLUMNS,
            "numeric_recipe_columns": NUMERIC_RECIPE_COLUMNS,
            "category_defaults": self.category_defaults,
            "numeric_bounds": self.numeric_bounds,
            "discrete_values": self.discrete_values,
            "models": models,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a direct inverse baseline (targets -> recipe).")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models-dir", "--models", dest="models_dir", type=str, default="artifacts/forward_models")
    parser.add_argument("--output-dir", type=str, default="artifacts/inverse_direct")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--strict-validation", action="store_true")
    args = parser.parse_args()

    baseline = DirectInverseBaseline(
        data_path=args.data,
        models_dir=args.models_dir,
        cv_folds=args.cv_folds,
        strict_validation=args.strict_validation,
    )
    predictions, metrics = baseline.evaluate_oof()
    model_payload = baseline.fit_full_model()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    metrics_path = os.path.join(output_dir, "metrics.json")
    model_path = os.path.join(output_dir, "inverse_direct_model.joblib")

    predictions.to_csv(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    joblib.dump(model_payload, model_path)

    print(f"Feasibility rate: {metrics['feasibility_rate']:.3f}")
    print(f"Mean target score: {metrics['mean_score']:.4f}")
    for target_name in TARGET_COLUMNS:
        print(
            f"  {target_name}: "
            f"MAE={metrics['target_mae_recheck'][target_name]:.4f} "
            f"RMSE={metrics['target_hit_rmse_recheck'][target_name]:.4f}"
        )
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved trained baseline to {model_path}")


if __name__ == "__main__":
    main()
