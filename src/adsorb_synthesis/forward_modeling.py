"""Shared helpers for forward-model training, feature selection, and UQ."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from .constants import RARE_METALS_THRESHOLD
from .feature_selection import get_curated_features, select_features_advanced
from .physics_losses import compute_physics_penalty


def build_stratification_key(X: pd.DataFrame, y_target: pd.Series) -> pd.Series:
    """Build a stable stratification key from metal identity and target bins."""
    if 'Металл' in X.columns:
        metal_counts = X['Металл'].value_counts()
        rare = metal_counts[metal_counts < RARE_METALS_THRESHOLD].index.tolist()
        metal_group = X['Металл'].apply(lambda metal: 'Other' if metal in rare else metal)
    else:
        metal_group = pd.Series(['Unknown'] * len(X), index=X.index)
    try:
        bins = pd.qcut(y_target, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        bins = pd.Series(['All'] * len(y_target), index=y_target.index)
    return metal_group.astype(str) + '_' + bins.astype(str)


def compute_quality_weights(
    df: pd.DataFrame,
    *,
    penalty_weight: float,
    minimum_weight: float = 0.2,
) -> pd.Series:
    """Downweight rows with strong physics violations instead of upweighting them."""
    penalty = compute_physics_penalty(df)
    weights = 1.0 / (1.0 + penalty_weight * penalty)
    weights = weights.clip(lower=minimum_weight, upper=1.0)
    return weights.rename("sample_weight")


def select_curated_features(
    X: pd.DataFrame,
    y_target: pd.Series,
    categorical_cols: Sequence[str],
    *,
    corr_threshold: float,
    vif_threshold: float,
    max_features: int,
    verbose: bool = False,
) -> Tuple[List[str], Dict]:
    """Apply curated keep/drop rules before advanced selection."""
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    keep_features, drop_features = get_curated_features()
    available_keep = [feature for feature in keep_features if feature in X.columns]
    available_drop = {feature for feature in drop_features if feature in X.columns}
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    curated_numeric = [col for col in numeric_cols if col not in available_drop]
    selected_features, report = select_features_advanced(
        X[categorical_cols + curated_numeric],
        y_target,
        categorical_cols=list(categorical_cols),
        hard_keep_features=available_keep,
        corr_threshold=corr_threshold,
        vif_threshold=vif_threshold,
        max_features=max_features,
        verbose=verbose,
    )
    report["available_keep_features"] = available_keep
    report["available_drop_features"] = sorted(available_drop)
    return selected_features, report


def prepare_tabpfn_regression_frame(
    X: pd.DataFrame,
    *,
    candidate_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build a numeric-only feature matrix for TabPFN regression.

    TabPFN expects scalar tabular values. We keep only numeric and boolean
    features, coerce them to numeric dtypes, and drop constant columns that can
    destabilize small-data fits.
    """
    if candidate_features is None:
        candidate_features = list(X.columns)

    selected_columns = [column for column in candidate_features if column in X.columns]
    numeric_frame = X[selected_columns].select_dtypes(include=[np.number, "bool"]).copy()
    for column in numeric_frame.columns:
        if pd.api.types.is_bool_dtype(numeric_frame[column]):
            numeric_frame[column] = numeric_frame[column].astype(float)

    numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
    dropped_constant = [
        column
        for column in numeric_frame.columns
        if numeric_frame[column].nunique(dropna=False) <= 1
    ]
    if dropped_constant:
        numeric_frame = numeric_frame.drop(columns=dropped_constant)

    if numeric_frame.shape[1] == 0:
        raise ValueError("TabPFN backend requires at least one non-constant numeric feature.")

    return numeric_frame, list(numeric_frame.columns), dropped_constant


class PrecomputedSplitCV(BaseCrossValidator):
    """A lightweight cross-validator backed by already-materialized splits."""

    def __init__(self, splits: Iterable[Tuple[Sequence[int], Sequence[int]]]):
        self.splits = [
            (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
            for train_idx, valid_idx in splits
        ]

    def split(self, X, y=None, groups=None):
        for train_idx, valid_idx in self.splits:
            yield train_idx.copy(), valid_idx.copy()

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.splits)


class SelectedFeatureCatBoostRegressor(BaseEstimator, RegressorMixin):
    """CatBoost regressor that performs curated feature selection inside fit."""

    def __init__(
        self,
        catboost_params: Optional[Dict] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        corr_threshold: float = 0.85,
        vif_threshold: float = 10.0,
        max_features: int = 20,
    ) -> None:
        self.catboost_params = catboost_params
        self.categorical_cols = categorical_cols
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)

        categorical_cols = list(self.categorical_cols or [])
        self.selected_features_, self.selection_report_ = select_curated_features(
            X,
            y,
            categorical_cols,
            corr_threshold=self.corr_threshold,
            vif_threshold=self.vif_threshold,
            max_features=self.max_features,
            verbose=False,
        )
        self.selected_cat_features_ = [
            column for column in self.selected_features_
            if column in categorical_cols
        ]
        self.model_ = CatBoostRegressor(
            **(self.catboost_params or {}),
            cat_features=self.selected_cat_features_,
        )
        self.model_.fit(
            X[self.selected_features_],
            y,
            sample_weight=sample_weight,
        )
        self.feature_names_in_ = np.asarray(self.selected_features_, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return np.asarray(self.model_.predict(X[self.selected_features_]), dtype=float)
