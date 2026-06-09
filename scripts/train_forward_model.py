#!/usr/bin/env python3
"""Train forward models with backend-specific challengers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from mapie.regression import CrossConformalRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adsorb_synthesis.config import FORWARD_MODEL_CONFIG, get_catboost_config
from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS, RANDOM_SEED
from adsorb_synthesis.data_processing import (
    build_lookup_tables,
    load_dataset,
    prepare_forward_dataset,
)
from adsorb_synthesis.forward_modeling import (
    PrecomputedSplitCV,
    SelectedFeatureCatBoostRegressor,
    build_recipe_group_keys,
    build_stratification_key,
    compute_quality_weights,
    prepare_tabpfn_inference_frame,
    prepare_tabpfn_regression_frame,
    select_curated_features,
)

SUPPORTED_BACKENDS = ("catboost", "tabpfn")


def _override_iterations(params: Dict, iterations: int | None) -> Dict:
    updated = dict(params)
    if iterations is not None:
        updated["iterations"] = iterations
    return updated


def _safe_target_name(target: str) -> str:
    return target.replace("/", "_").replace(" ", "_")


def _resolve_backend_output_dir(output_dir: str, backend: str, use_subdirs: bool) -> str:
    return os.path.join(output_dir, backend) if use_subdirs else output_dir


def _feature_selection_for_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: List[str],
    *,
    use_feature_selection: bool,
) -> Tuple[List[str], Dict]:
    if not use_feature_selection:
        return list(X_train.columns), {"removed_correlation": [], "removed_vif": []}
    return select_curated_features(
        X_train,
        y_train,
        categorical_cols,
        corr_threshold=FORWARD_MODEL_CONFIG.feature_selection_corr_threshold,
        vif_threshold=FORWARD_MODEL_CONFIG.feature_selection_vif_threshold,
        max_features=FORWARD_MODEL_CONFIG.feature_selection_max_features,
        verbose=False,
    )


def _build_training_context(
    data_path: str,
    *,
    validation_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
    print(f"Loading dataset from {data_path}...")
    df_raw = load_dataset(data_path, validation_mode=validation_mode)
    lookup_tables = build_lookup_tables(df_raw)
    X, y = prepare_forward_dataset(df_raw, lookup_tables=lookup_tables)
    cat_features = [col for col in X.columns if X[col].dtype.name in ["object", "category"]]

    # Audit #12: surface per-feature missingness so data gaps are not silently
    # absorbed by NaN-tolerant models (CatBoost) or coerced away (TabPFN).
    nan_frac = X.isna().mean().sort_values(ascending=False)
    high_nan = nan_frac[nan_frac > 0.05]
    if len(high_nan):
        print(f"  Features with >5% missing values ({len(high_nan)}):")
        for feat, frac in high_nan.items():
            print(f"    {feat:<40} {frac:6.1%} NaN")
    else:
        print("  Feature missingness: all features <5% NaN.")

    sample_weights = compute_quality_weights(
        df_raw.reindex(X.index),
        penalty_weight=FORWARD_MODEL_CONFIG.physics_penalty_weight,
        minimum_weight=FORWARD_MODEL_CONFIG.minimum_sample_weight,
    ).to_numpy(dtype=float)
    return df_raw, X, y, cat_features, sample_weights


def _write_backend_metadata(
    output_dir: str,
    *,
    metrics: Dict[str, Dict],
    feature_meta: Dict[str, Dict],
    uq_models: Dict[str, object] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    joblib.dump(feature_meta, os.path.join(output_dir, "feature_meta.joblib"))
    if uq_models is not None:
        joblib.dump(uq_models, os.path.join(output_dir, "uncertainty_calibrators.joblib"))


def _import_tabpfn():
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "TabPFN backend requires the optional `tabpfn` package (>=8.0.0 for the "
            "TabPFN-3 default checkpoint). Install it with "
            "`.venv/bin/pip install -U tabpfn` or use `--backend catboost`."
        ) from exc
    return TabPFNRegressor


def _fit_tabpfn_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    try:
        model.fit(X_train, y_train.to_numpy(dtype=float))
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a license/auth error
        message = str(exc)
        license_markers = (
            "license acceptance",
            "TABPFN_TOKEN",
            "gated",
            "HuggingFace authentication error",
            "accept its terms",
        )
        is_license_error = type(exc).__name__ in {
            "TabPFNLicenseError",
            "TabPFNHuggingFaceGatedRepoError",
        } or any(marker in message for marker in license_markers)
        if is_license_error:
            raise RuntimeError(
                "TabPFN-3 (tabpfn>=8.0.0) needs a one-time license acceptance before the "
                "weights can be downloaded. Either run this once in an interactive terminal "
                "(a browser opens to accept the license, then the token is cached), or, for a "
                "headless run, get an API key from https://ux.priorlabs.ai/account and set "
                'TABPFN_TOKEN="<api-key>". Alternatively use `--backend catboost`.'
            ) from exc
        raise


def _make_tabpfn_regressor(tabpfn_regressor_cls):
    # TabPFN-3 is the default checkpoint in tabpfn>=8.0.0. Pin it explicitly so the
    # challenger stays on V3 even if a future package release changes the default.
    try:
        from tabpfn.constants import ModelVersion  # type: ignore

        return tabpfn_regressor_cls.create_default_for_version(ModelVersion.V3)
    except (ImportError, AttributeError):  # older tabpfn without explicit V3 selection
        return tabpfn_regressor_cls()


def train_catboost_models(
    data_path: str,
    output_dir: str,
    *,
    iterations: int | None = None,
    validation_mode: str = "warn",
    use_feature_selection: bool = True,
) -> None:
    _, X, y, cat_features, sample_weights = _build_training_context(
        data_path,
        validation_mode=validation_mode,
    )

    os.makedirs(output_dir, exist_ok=True)
    metrics: Dict[str, Dict] = {}
    uq_models: Dict[str, object] = {}
    feature_meta: Dict[str, Dict] = {
        "backend": "catboost",
        "all_feature_names": list(X.columns),
        "categorical_features": cat_features,
        "targets": {},
    }

    for target in FORWARD_MODEL_TARGETS:
        if target not in y.columns:
            print(f"Skipping {target}: target not present in dataset.")
            continue

        print(f"\n=== Training target: {target} [catboost] ===")
        y_target = y[target]
        strat_key = build_stratification_key(X, y_target)
        recipe_groups = build_recipe_group_keys(X)
        outer_cv = StratifiedGroupKFold(
            n_splits=FORWARD_MODEL_CONFIG.n_ensemble_splits,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        cv_splits = list(outer_cv.split(X, strat_key, groups=recipe_groups))
        oof_preds = np.full(len(X), np.nan, dtype=float)
        fold_ids = np.full(len(X), -1, dtype=int)
        fold_feature_sets: List[List[str]] = []
        removed_corr_total = 0
        removed_vif_total = 0

        cb_params = _override_iterations(
            get_catboost_config(target).to_params(random_state=RANDOM_SEED),
            iterations,
        )
        cb_params.pop("random_seed", None)
        cb_params["verbose"] = False
        cb_params["allow_writing_files"] = False

        for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y_target.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y_target.iloc[valid_idx]
            selected_features, selection_report = _feature_selection_for_fold(
                X_train,
                y_train,
                cat_features,
                use_feature_selection=use_feature_selection,
            )
            fold_feature_sets.append(selected_features)
            removed_corr_total += len(selection_report.get("removed_correlation", []))
            removed_vif_total += len(selection_report.get("removed_vif", []))
            selected_cat_features = [feature for feature in selected_features if feature in cat_features]

            model = CatBoostRegressor(
                **cb_params,
                random_seed=RANDOM_SEED + fold_idx,
                cat_features=selected_cat_features,
            )
            model.fit(
                X_train[selected_features],
                y_train,
                sample_weight=sample_weights[train_idx],
                eval_set=(X_valid[selected_features], y_valid),
                early_stopping_rounds=FORWARD_MODEL_CONFIG.early_stopping_rounds,
                use_best_model=True,
            )
            oof_preds[valid_idx] = model.predict(X_valid[selected_features])
            fold_ids[valid_idx] = fold_idx

        r2_oof = r2_score(y_target, oof_preds)
        rmse_oof = float(np.sqrt(mean_squared_error(y_target, oof_preds)))
        mae_oof = float(mean_absolute_error(y_target, oof_preds))
        print(f"  OOF R2={r2_oof:.4f} RMSE={rmse_oof:.4f} MAE={mae_oof:.4f}")

        selected_features_full, full_selection_report = _feature_selection_for_fold(
            X,
            y_target,
            cat_features,
            use_feature_selection=use_feature_selection,
        )
        selected_cat_full = [feature for feature in selected_features_full if feature in cat_features]

        production_models = []
        model_paths = []
        safe_target = _safe_target_name(target)
        for member_idx in range(FORWARD_MODEL_CONFIG.n_ensemble_members):
            member_seed = RANDOM_SEED + (member_idx + 1) * FORWARD_MODEL_CONFIG.ensemble_seed_step
            model = CatBoostRegressor(
                **cb_params,
                random_seed=member_seed,
                cat_features=selected_cat_full,
            )
            model.fit(
                X[selected_features_full],
                y_target,
                sample_weight=sample_weights,
            )
            model_path = os.path.join(output_dir, f"catboost_{safe_target}_ens{member_idx}.cbm")
            model.save_model(model_path)
            production_models.append(model)
            model_paths.append(model_path)

        prod_predictions = np.stack(
            [model.predict(X[selected_features_full]) for model in production_models],
            axis=1,
        )
        prod_mean = prod_predictions.mean(axis=1)
        r2_prod = r2_score(y_target, prod_mean)
        rmse_prod = float(np.sqrt(mean_squared_error(y_target, prod_mean)))
        mae_prod = float(mean_absolute_error(y_target, prod_mean))

        if use_feature_selection:
            estimator = SelectedFeatureCatBoostRegressor(
                catboost_params=cb_params,
                categorical_cols=cat_features,
                corr_threshold=FORWARD_MODEL_CONFIG.feature_selection_corr_threshold,
                vif_threshold=FORWARD_MODEL_CONFIG.feature_selection_vif_threshold,
                max_features=FORWARD_MODEL_CONFIG.feature_selection_max_features,
            )
        else:
            estimator = SelectedFeatureCatBoostRegressor(
                catboost_params=cb_params,
                categorical_cols=cat_features,
                corr_threshold=1.0,
                vif_threshold=float("inf"),
                max_features=max(1, len(X.columns)),
            )

        confidence_level = 1.0 - FORWARD_MODEL_CONFIG.conformal_alpha
        uq_model = CrossConformalRegressor(
            estimator=estimator,
            confidence_level=confidence_level,
            method=FORWARD_MODEL_CONFIG.mapie_method,
            cv=PrecomputedSplitCV(cv_splits),
            random_state=RANDOM_SEED,
        )
        uq_model.fit_conformalize(
            X,
            y_target,
            fit_params={"sample_weight": sample_weights},
        )
        interval_center, interval_bounds = uq_model.predict_interval(X, aggregate_predictions="mean")
        interval_bounds = np.asarray(interval_bounds, dtype=float).squeeze(-1)
        y_lo = interval_bounds[:, 0]
        y_hi = interval_bounds[:, 1]
        interval_width = y_hi - y_lo
        coverage = float(
            np.mean(
                (y_target.to_numpy(dtype=float) >= y_lo)
                & (y_target.to_numpy(dtype=float) <= y_hi)
            )
        )
        uq_models[target] = uq_model

        predictions_df = pd.DataFrame({
            "y_actual": y_target.to_numpy(dtype=float),
            "y_oof": oof_preds,
            "fold_id": fold_ids,
            "y_prod_mean": prod_mean,
            "y_interval_center": np.asarray(interval_center, dtype=float),
            "y_lo": y_lo,
            "y_hi": y_hi,
            "interval_width": interval_width,
        })
        predictions_path = os.path.join(output_dir, f"predictions_{safe_target}.csv")
        predictions_df.to_csv(predictions_path, index=False)

        feature_meta["targets"][target] = {
            "selected_features": selected_features_full,
            "categorical_features": selected_cat_full,
            "model_paths": model_paths,
        }
        metrics[target] = {
            "backend": "catboost",
            "R2": float(r2_oof),
            "R2_oof": float(r2_oof),
            "RMSE": rmse_oof,
            "RMSE_oof": rmse_oof,
            "MAE": mae_oof,
            "MAE_oof": mae_oof,
            "R2_production": float(r2_prod),
            "RMSE_production": rmse_prod,
            "MAE_production": mae_prod,
            "selected_features": selected_features_full,
            "fold_feature_sets": fold_feature_sets,
            "n_removed_multicollinear_oof": int(removed_corr_total + removed_vif_total),
            "n_removed_multicollinear_full": int(
                len(full_selection_report.get("removed_correlation", []))
                + len(full_selection_report.get("removed_vif", []))
            ),
            "interval_confidence_level": confidence_level,
            "interval_coverage": coverage,
            "interval_width_mean": float(np.mean(interval_width)),
            # Audit #6: practical UQ KPIs — coverage gap vs target and width
            # normalized by target spread (a wide normalized width = uninformative
            # interval even at nominal coverage).
            "interval_coverage_gap": float(coverage - confidence_level),
            "interval_width_normalized": float(
                np.mean(interval_width) / (float(y_target.std()) + 1e-9)
            ),
            "cv_folds": FORWARD_MODEL_CONFIG.n_ensemble_splits,
            "ensemble_members": FORWARD_MODEL_CONFIG.n_ensemble_members,
        }

        target_std = float(y_target.std()) + 1e-9
        print(
            f"  Production R2={r2_prod:.4f} RMSE={rmse_prod:.4f} "
            f"coverage={coverage:.1%} (target {confidence_level:.0%}, "
            f"gap {coverage - confidence_level:+.1%}) "
            f"width={np.mean(interval_width):.4f} "
            f"(norm {np.mean(interval_width) / target_std:.2f})"
        )

    _write_backend_metadata(
        output_dir,
        metrics=metrics,
        feature_meta=feature_meta,
        uq_models=uq_models,
    )
    print(f"\nTraining complete. CatBoost models and intervals saved to {output_dir}")


def train_tabpfn_models(
    data_path: str,
    output_dir: str,
    *,
    validation_mode: str = "warn",
    use_feature_selection: bool = True,
) -> None:
    TabPFNRegressor = _import_tabpfn()
    _, X, y, cat_features, _ = _build_training_context(
        data_path,
        validation_mode=validation_mode,
    )

    os.makedirs(output_dir, exist_ok=True)
    metrics: Dict[str, Dict] = {}
    feature_meta: Dict[str, Dict] = {
        "backend": "tabpfn",
        "all_feature_names": list(X.columns),
        "categorical_features": cat_features,
        "targets": {},
    }

    for target in FORWARD_MODEL_TARGETS:
        if target not in y.columns:
            print(f"Skipping {target}: target not present in dataset.")
            continue

        print(f"\n=== Training target: {target} [tabpfn] ===")
        y_target = y[target]
        strat_key = build_stratification_key(X, y_target)
        recipe_groups = build_recipe_group_keys(X)
        outer_cv = StratifiedGroupKFold(
            n_splits=FORWARD_MODEL_CONFIG.n_ensemble_splits,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        cv_splits = list(outer_cv.split(X, strat_key, groups=recipe_groups))
        oof_preds = np.full(len(X), np.nan, dtype=float)
        fold_ids = np.full(len(X), -1, dtype=int)
        fold_feature_sets: List[List[str]] = []
        dropped_constant_features: List[str] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y_target.iloc[train_idx]
            X_valid = X.iloc[valid_idx]

            selected_features, _ = _feature_selection_for_fold(
                X_train,
                y_train,
                cat_features,
                use_feature_selection=use_feature_selection,
            )
            X_train_tabpfn, tabpfn_features, dropped_constant = prepare_tabpfn_regression_frame(
                X_train,
                candidate_features=selected_features,
            )
            X_valid_tabpfn = prepare_tabpfn_inference_frame(X_valid, tabpfn_features)
            fold_feature_sets.append(tabpfn_features)
            dropped_constant_features.extend(dropped_constant)

            model = _make_tabpfn_regressor(TabPFNRegressor)
            _fit_tabpfn_model(model, X_train_tabpfn, y_train)
            oof_preds[valid_idx] = np.asarray(model.predict(X_valid_tabpfn), dtype=float)
            fold_ids[valid_idx] = fold_idx

        r2_oof = r2_score(y_target, oof_preds)
        rmse_oof = float(np.sqrt(mean_squared_error(y_target, oof_preds)))
        mae_oof = float(mean_absolute_error(y_target, oof_preds))
        print(f"  OOF R2={r2_oof:.4f} RMSE={rmse_oof:.4f} MAE={mae_oof:.4f}")

        selected_features_full, _ = _feature_selection_for_fold(
            X,
            y_target,
            cat_features,
            use_feature_selection=use_feature_selection,
        )
        X_tabpfn_full, tabpfn_features_full, dropped_constant_full = prepare_tabpfn_regression_frame(
            X,
            candidate_features=selected_features_full,
        )
        dropped_constant_features.extend(dropped_constant_full)

        production_model = _make_tabpfn_regressor(TabPFNRegressor)
        _fit_tabpfn_model(production_model, X_tabpfn_full, y_target)
        prod_mean = np.asarray(production_model.predict(X_tabpfn_full), dtype=float)
        r2_prod = r2_score(y_target, prod_mean)
        rmse_prod = float(np.sqrt(mean_squared_error(y_target, prod_mean)))
        mae_prod = float(mean_absolute_error(y_target, prod_mean))

        safe_target = _safe_target_name(target)
        model_path = os.path.join(output_dir, f"tabpfn_{safe_target}.joblib")
        joblib.dump(production_model, model_path)

        predictions_df = pd.DataFrame({
            "y_actual": y_target.to_numpy(dtype=float),
            "y_oof": oof_preds,
            "fold_id": fold_ids,
            "y_prod_mean": prod_mean,
            "y_interval_center": np.full(len(X), np.nan, dtype=float),
            "y_lo": np.full(len(X), np.nan, dtype=float),
            "y_hi": np.full(len(X), np.nan, dtype=float),
            "interval_width": np.full(len(X), np.nan, dtype=float),
        })
        predictions_path = os.path.join(output_dir, f"predictions_{safe_target}.csv")
        predictions_df.to_csv(predictions_path, index=False)

        dedup_dropped = sorted(set(dropped_constant_features))
        feature_meta["targets"][target] = {
            "selected_features": tabpfn_features_full,
            "categorical_features": [],
            "dropped_constant_features": dedup_dropped,
            "model_paths": [model_path],
        }
        metrics[target] = {
            "backend": "tabpfn",
            "R2": float(r2_oof),
            "R2_oof": float(r2_oof),
            "RMSE": rmse_oof,
            "RMSE_oof": rmse_oof,
            "MAE": mae_oof,
            "MAE_oof": mae_oof,
            "R2_production": float(r2_prod),
            "RMSE_production": rmse_prod,
            "MAE_production": mae_prod,
            "selected_features": tabpfn_features_full,
            "fold_feature_sets": fold_feature_sets,
            "dropped_constant_features": dedup_dropped,
            "interval_confidence_level": None,
            "interval_coverage": None,
            "interval_width_mean": None,
            "cv_folds": FORWARD_MODEL_CONFIG.n_ensemble_splits,
            "ensemble_members": 1,
        }

        print(f"  Production R2={r2_prod:.4f} RMSE={rmse_prod:.4f} (TabPFN challenger)")

    _write_backend_metadata(
        output_dir,
        metrics=metrics,
        feature_meta=feature_meta,
    )
    print(f"\nTraining complete. TabPFN models saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forward models for adsorbent synthesis.")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--output", type=str, default="artifacts/forward_models")
    parser.add_argument("--iterations", type=int, default=None, help="Override CatBoost iterations.")
    parser.add_argument("--backend", choices=["catboost", "tabpfn", "all"], default="catboost")
    parser.add_argument("--no-feature-selection", action="store_true")
    parser.add_argument(
        "--validation-mode",
        type=str,
        default="warn",
        choices=["warn", "strict"],
    )
    args = parser.parse_args()

    use_subdirs = args.backend == "all"
    backends = SUPPORTED_BACKENDS if args.backend == "all" else (args.backend,)
    for backend in backends:
        backend_output_dir = _resolve_backend_output_dir(args.output, backend, use_subdirs)
        if backend == "catboost":
            train_catboost_models(
                args.data,
                backend_output_dir,
                iterations=args.iterations,
                validation_mode=args.validation_mode,
                use_feature_selection=not args.no_feature_selection,
            )
        elif backend == "tabpfn":
            if args.iterations is not None:
                print("Warning: --iterations is ignored for the TabPFN backend.")
            train_tabpfn_models(
                args.data,
                backend_output_dir,
                validation_mode=args.validation_mode,
                use_feature_selection=not args.no_feature_selection,
            )
        else:  # pragma: no cover - protected by argparse choices
            raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    main()
