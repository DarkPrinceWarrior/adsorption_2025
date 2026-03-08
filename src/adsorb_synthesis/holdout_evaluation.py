"""Helpers for deterministic external holdout evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from mapie.regression import CrossConformalRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .constants import FORWARD_MODEL_TARGETS
from .data_processing import build_lookup_tables, prepare_forward_dataset
from .forward_modeling import prepare_tabpfn_inference_frame


DEFAULT_HOLDOUT_GROUP_COLUMNS: Tuple[str, ...] = ("Металл", "Лиганд")


@dataclass(frozen=True)
class HoldoutSplit:
    train_indices: List[int]
    holdout_indices: List[int]
    train_groups: List[str]
    holdout_groups: List[str]
    group_columns: Tuple[str, ...]
    holdout_fraction: float
    seed: int

    def to_manifest(self) -> Dict[str, object]:
        return {
            "group_columns": list(self.group_columns),
            "holdout_fraction": self.holdout_fraction,
            "seed": self.seed,
            "train_indices": self.train_indices,
            "holdout_indices": self.holdout_indices,
            "train_groups": self.train_groups,
            "holdout_groups": self.holdout_groups,
        }


def safe_target_name(target: str) -> str:
    return target.replace("/", "_").replace(" ", "_")


def build_chemistry_group_keys(
    df: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_HOLDOUT_GROUP_COLUMNS,
) -> pd.Series:
    missing = [column for column in group_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing group columns: {missing}")
    return df[list(group_columns)].astype(str).agg("|".join, axis=1)


def build_chemistry_holdout_split(
    df: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_HOLDOUT_GROUP_COLUMNS,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> HoldoutSplit:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")

    group_keys = build_chemistry_group_keys(df, group_columns=group_columns)
    group_counts = group_keys.value_counts().sort_index()
    if len(group_counts) < 2:
        raise ValueError("Chemistry holdout requires at least two distinct groups.")

    hashed_groups = sorted(
        group_counts.index.tolist(),
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    target_rows = max(1, int(round(len(df) * holdout_fraction)))

    holdout_groups: List[str] = []
    holdout_rows = 0
    for group in hashed_groups:
        remaining_groups = len(group_counts) - len(holdout_groups) - 1
        if remaining_groups < 1:
            break
        holdout_groups.append(group)
        holdout_rows += int(group_counts[group])
        if holdout_rows >= target_rows:
            break

    if not holdout_groups:
        holdout_groups = [hashed_groups[0]]

    holdout_mask = group_keys.isin(holdout_groups)
    holdout_indices = df.index[holdout_mask].tolist()
    train_indices = df.index[~holdout_mask].tolist()
    if not train_indices or not holdout_indices:
        raise ValueError("Chemistry holdout split collapsed into an empty train or holdout partition.")

    train_groups = sorted(group_keys.iloc[train_indices].unique().tolist())
    holdout_groups = sorted(group_keys.iloc[holdout_indices].unique().tolist())
    return HoldoutSplit(
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        train_groups=train_groups,
        holdout_groups=holdout_groups,
        group_columns=tuple(group_columns),
        holdout_fraction=holdout_fraction,
        seed=seed,
    )


def write_split_manifest(split: HoldoutSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> HoldoutSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return HoldoutSplit(
        train_indices=[int(value) for value in payload["train_indices"]],
        holdout_indices=[int(value) for value in payload["holdout_indices"]],
        train_groups=[str(value) for value in payload["train_groups"]],
        holdout_groups=[str(value) for value in payload["holdout_groups"]],
        group_columns=tuple(payload["group_columns"]),
        holdout_fraction=float(payload["holdout_fraction"]),
        seed=int(payload["seed"]),
    )


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2_holdout": float(r2_score(y_true, y_pred)),
        "RMSE_holdout": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_holdout": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_catboost_holdout(
    *,
    models_dir: Path,
    df_train_raw: pd.DataFrame,
    df_holdout_raw: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, pd.DataFrame]]:
    lookup_tables = build_lookup_tables(df_train_raw)
    X_holdout, y_holdout = prepare_forward_dataset(df_holdout_raw, lookup_tables=lookup_tables)
    feature_meta = joblib.load(models_dir / "feature_meta.joblib")
    uq_models: Dict[str, CrossConformalRegressor] = joblib.load(models_dir / "uncertainty_calibrators.joblib")

    metrics: Dict[str, Dict[str, object]] = {"backend": "catboost"}
    predictions: Dict[str, pd.DataFrame] = {}
    for target in FORWARD_MODEL_TARGETS:
        target_meta = feature_meta["targets"].get(target)
        if not target_meta:
            continue
        safe_target = safe_target_name(target)
        selected_features = list(target_meta["selected_features"])
        model_paths = [Path(path) for path in target_meta["model_paths"]]
        ensemble: List[CatBoostRegressor] = []
        for model_path in model_paths:
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            ensemble.append(model)
        prod_predictions = np.stack(
            [model.predict(X_holdout[selected_features]) for model in ensemble],
            axis=1,
        )
        y_pred = prod_predictions.mean(axis=1)
        y_true = y_holdout[target].to_numpy(dtype=float)
        uq_model = uq_models[target]
        interval_center, interval_bounds = uq_model.predict_interval(X_holdout, aggregate_predictions="mean")
        interval_bounds = np.asarray(interval_bounds, dtype=float).squeeze(-1)
        y_lo = interval_bounds[:, 0]
        y_hi = interval_bounds[:, 1]
        interval_width = y_hi - y_lo
        metrics[target] = {
            **_metric_bundle(y_true, y_pred),
            "backend": "catboost",
            "interval_supported": True,
            "holdout_interval_coverage": float(np.mean((y_true >= y_lo) & (y_true <= y_hi))),
            "holdout_interval_width_mean": float(np.mean(interval_width)),
            "holdout_rows": int(len(y_true)),
        }
        predictions[target] = pd.DataFrame({
            "y_actual": y_true,
            "y_pred": y_pred,
            "y_interval_center": np.asarray(interval_center, dtype=float),
            "y_lo": y_lo,
            "y_hi": y_hi,
            "interval_width": interval_width,
        })
    return metrics, predictions


def evaluate_tabpfn_holdout(
    *,
    models_dir: Path,
    df_train_raw: pd.DataFrame,
    df_holdout_raw: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, pd.DataFrame]]:
    lookup_tables = build_lookup_tables(df_train_raw)
    X_holdout, y_holdout = prepare_forward_dataset(df_holdout_raw, lookup_tables=lookup_tables)
    feature_meta = joblib.load(models_dir / "feature_meta.joblib")

    metrics: Dict[str, Dict[str, object]] = {"backend": "tabpfn"}
    predictions: Dict[str, pd.DataFrame] = {}
    for target in FORWARD_MODEL_TARGETS:
        target_meta = feature_meta["targets"].get(target)
        if not target_meta:
            continue
        model_path = Path(target_meta["model_paths"][0])
        model = joblib.load(model_path)
        inference_features = list(target_meta["selected_features"])
        X_eval = prepare_tabpfn_inference_frame(X_holdout, inference_features)
        y_true = y_holdout[target].to_numpy(dtype=float)
        y_pred = np.asarray(model.predict(X_eval), dtype=float)
        metrics[target] = {
            **_metric_bundle(y_true, y_pred),
            "backend": "tabpfn",
            "interval_supported": False,
            "holdout_interval_coverage": None,
            "holdout_interval_width_mean": None,
            "holdout_rows": int(len(y_true)),
        }
        predictions[target] = pd.DataFrame({
            "y_actual": y_true,
            "y_pred": y_pred,
            "y_interval_center": np.full(len(y_true), np.nan, dtype=float),
            "y_lo": np.full(len(y_true), np.nan, dtype=float),
            "y_hi": np.full(len(y_true), np.nan, dtype=float),
            "interval_width": np.full(len(y_true), np.nan, dtype=float),
        })
    return metrics, predictions
