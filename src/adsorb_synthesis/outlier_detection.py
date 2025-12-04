"""Robust outlier detection strategies for regression targets.

This module provides alternatives to IsolationForest that better preserve
valid rare samples in long-tail distributions typical of adsorption data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


class OutlierMethod(str, Enum):
    """Supported outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    MAD = "mad"  # Median Absolute Deviation - robust to skew
    TUKEY = "tukey"  # Tukey's fences with configurable k
    WINSORIZE = "winsorize"  # Clip extremes rather than remove
    NONE = "none"


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for outlier handling in a pipeline stage.

    Attributes:
        method: Detection algorithm to use.
        threshold: Method-specific threshold (IQR multiplier, z-score cutoff, etc.).
        min_samples: Minimum samples to retain; if removal would drop below this,
            fall back to winsorizing instead.
        use_features: If True, detect outliers in feature+target space (multivariate).
            If False (default), use target-only detection.
        winsorize_percentile: For WINSORIZE method, clip to this percentile range.
    """

    method: OutlierMethod = OutlierMethod.MAD
    threshold: float = 3.5
    min_samples: int = 30
    use_features: bool = False
    winsorize_percentile: float = 0.05


def detect_outliers(
    values: np.ndarray,
    *,
    method: OutlierMethod = OutlierMethod.MAD,
    threshold: float = 3.5,
) -> np.ndarray:
    """Detect outliers in a 1D array and return boolean mask of inliers.

    Args:
        values: 1D array of values to check.
        method: Detection algorithm.
        threshold: Method-specific threshold.

    Returns:
        Boolean array where True = inlier (keep), False = outlier (remove).
    """
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.isfinite(values)

    if not valid_mask.any():
        return np.ones(len(values), dtype=bool)

    valid_values = values[valid_mask]

    if method == OutlierMethod.NONE:
        return np.ones(len(values), dtype=bool)

    if method == OutlierMethod.IQR:
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        inlier_mask = (values >= lower) & (values <= upper)

    elif method == OutlierMethod.ZSCORE:
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        if std < 1e-10:
            return np.ones(len(values), dtype=bool)
        z_scores = np.abs((values - mean) / std)
        inlier_mask = z_scores <= threshold

    elif method == OutlierMethod.MAD:
        # Median Absolute Deviation - robust to skewed distributions
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))
        if mad < 1e-10:
            # Fall back to IQR if MAD is zero
            return detect_outliers(values, method=OutlierMethod.IQR, threshold=threshold)
        # Modified z-score using MAD
        modified_z = 0.6745 * (values - median) / mad
        inlier_mask = np.abs(modified_z) <= threshold

    elif method == OutlierMethod.TUKEY:
        # Tukey's fences with configurable k (default 1.5, use higher for less aggressive)
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        inlier_mask = (values >= lower) & (values <= upper)

    elif method == OutlierMethod.WINSORIZE:
        # Winsorize doesn't remove, just marks all as inliers
        # Actual winsorization happens in filter_outliers
        return np.ones(len(values), dtype=bool)

    else:
        raise ValueError(f"Unknown outlier method: {method}")

    # NaN values are treated as inliers (handled elsewhere)
    inlier_mask = inlier_mask | ~valid_mask
    return inlier_mask


def detect_multivariate_outliers(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    *,
    threshold: float = 3.5,
) -> np.ndarray:
    """Detect outliers using Mahalanobis distance in feature+target space.

    More conservative than univariate detection; identifies points that are
    anomalous relative to the joint distribution.

    Args:
        df: DataFrame with features and target.
        feature_columns: Columns to include in multivariate analysis.
        target_column: Target column to include.
        threshold: Chi-squared p-value threshold for outlier detection.

    Returns:
        Boolean array where True = inlier.
    """
    cols = list(feature_columns) + [target_column]
    available_cols = [c for c in cols if c in df.columns]

    if len(available_cols) < 2:
        return np.ones(len(df), dtype=bool)

    data = df[available_cols].copy()

    # Only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return np.ones(len(df), dtype=bool)

    data = data[numeric_cols].dropna()

    if len(data) < len(numeric_cols) + 1:
        return np.ones(len(df), dtype=bool)

    try:
        # Compute Mahalanobis distance
        mean = data.mean()
        cov = data.cov()

        # Regularize covariance matrix
        cov_reg = cov + np.eye(len(cov)) * 1e-6

        cov_inv = np.linalg.inv(cov_reg)
        diff = data - mean
        mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        # Chi-squared test for outliers
        p_values = 1 - stats.chi2.cdf(mahal ** 2, df=len(numeric_cols))
        inlier_mask_subset = p_values > (1 - threshold / 100)

        # Map back to original indices
        inlier_mask = np.ones(len(df), dtype=bool)
        inlier_mask[data.index] = inlier_mask_subset

    except (np.linalg.LinAlgError, ValueError):
        # Fall back to univariate if covariance is singular
        return np.ones(len(df), dtype=bool)

    return inlier_mask


def filter_outliers(
    df: pd.DataFrame,
    target_column: str,
    config: OutlierConfig,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter outliers from DataFrame using configured strategy.

    This is a safer alternative to IsolationForest that:
    1. Preserves minimum sample count
    2. Uses robust statistics (MAD by default)
    3. Can optionally winsorize instead of removing

    Args:
        df: Input DataFrame.
        target_column: Column to check for outliers.
        config: Outlier detection configuration.
        feature_columns: Optional feature columns for multivariate detection.

    Returns:
        Filtered DataFrame (copy, never mutates input).
    """
    if config.method == OutlierMethod.NONE:
        return df.copy()

    if len(df) <= config.min_samples:
        return df.copy()

    target_values = pd.to_numeric(df[target_column], errors="coerce").values

    if config.use_features and feature_columns:
        inlier_mask = detect_multivariate_outliers(
            df,
            feature_columns=feature_columns,
            target_column=target_column,
            threshold=config.threshold,
        )
    else:
        inlier_mask = detect_outliers(
            target_values,
            method=config.method,
            threshold=config.threshold,
        )

    n_outliers = (~inlier_mask).sum()
    n_remaining = inlier_mask.sum()

    # Safety: if removing outliers would leave too few samples, winsorize instead
    if n_remaining < config.min_samples and n_outliers > 0:
        return _winsorize_target(df, target_column, config.winsorize_percentile)

    if config.method == OutlierMethod.WINSORIZE:
        return _winsorize_target(df, target_column, config.winsorize_percentile)

    return df.loc[inlier_mask].copy()


def _winsorize_target(
    df: pd.DataFrame,
    target_column: str,
    percentile: float,
) -> pd.DataFrame:
    """Clip target values to percentile bounds instead of removing rows.

    Args:
        df: Input DataFrame.
        target_column: Column to winsorize.
        percentile: Fraction to clip from each tail (e.g., 0.05 = 5%).

    Returns:
        DataFrame with winsorized target (copy).
    """
    result = df.copy()
    values = pd.to_numeric(result[target_column], errors="coerce")
    lower = values.quantile(percentile)
    upper = values.quantile(1 - percentile)
    result[target_column] = values.clip(lower=lower, upper=upper)
    return result


# Default configuration - conservative MAD-based detection
DEFAULT_OUTLIER_CONFIG = OutlierConfig(
    method=OutlierMethod.MAD,
    threshold=3.5,  # ~3.5 modified z-scores ≈ 99.9% under normality
    min_samples=30,
    use_features=False,
)
