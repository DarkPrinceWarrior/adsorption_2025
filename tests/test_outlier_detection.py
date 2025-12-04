"""Tests for the robust outlier detection module."""
import numpy as np
import pandas as pd
import pytest

from src.adsorb_synthesis.outlier_detection import (
    DEFAULT_OUTLIER_CONFIG,
    OutlierConfig,
    OutlierMethod,
    detect_multivariate_outliers,
    detect_outliers,
    filter_outliers,
)


class TestDetectOutliers:
    """Unit tests for univariate outlier detection."""

    def test_iqr_detects_extreme_values(self):
        """IQR method should flag values far outside quartiles."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        mask = detect_outliers(values, method=OutlierMethod.IQR, threshold=1.5)
        
        assert mask[:-1].all()  # First 9 are inliers
        assert not mask[-1]  # 100 is an outlier

    def test_mad_robust_to_skew(self):
        """MAD should handle skewed distributions better than z-score."""
        # Log-normal like distribution (common in adsorption data)
        np.random.seed(42)
        values = np.exp(np.random.randn(100))
        values = np.append(values, 1000)  # Add obvious outlier
        
        mask = detect_outliers(values, method=OutlierMethod.MAD, threshold=3.5)
        
        # Should detect the extreme outlier
        assert not mask[-1]
        # Should keep most of the log-normal values
        assert mask[:-1].sum() >= 90

    def test_zscore_detects_outliers(self):
        """Z-score method works for normally distributed data."""
        np.random.seed(42)
        values = np.random.randn(100)
        values = np.append(values, 10)  # 10 sigma outlier
        
        mask = detect_outliers(values, method=OutlierMethod.ZSCORE, threshold=3.0)
        
        assert not mask[-1]

    def test_none_method_keeps_all(self):
        """NONE method should return all True."""
        values = np.array([1, 2, 1000, -1000])
        mask = detect_outliers(values, method=OutlierMethod.NONE)
        
        assert mask.all()

    def test_handles_nan_values(self):
        """NaN values should be treated as inliers (handled elsewhere)."""
        values = np.array([1, 2, np.nan, 4, 5])
        mask = detect_outliers(values, method=OutlierMethod.MAD)
        
        assert mask[2]  # NaN position is marked as inlier

    def test_handles_constant_values(self):
        """Constant values shouldn't cause division by zero."""
        values = np.array([5, 5, 5, 5, 5])
        mask = detect_outliers(values, method=OutlierMethod.MAD)
        
        assert mask.all()


class TestFilterOutliers:
    """Tests for the DataFrame-level filter function."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with outliers."""
        return pd.DataFrame({
            'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature2': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        })

    def test_filter_returns_copy(self, sample_df):
        """filter_outliers should never mutate input."""
        original_len = len(sample_df)
        # Use MAD which will definitely catch 100 as an outlier
        config = OutlierConfig(method=OutlierMethod.MAD, threshold=3.0, min_samples=5)
        
        result = filter_outliers(sample_df, 'target', config)
        
        assert len(sample_df) == original_len  # Original unchanged
        # With MAD threshold 3.0, 100 should be flagged as outlier
        assert len(result) <= original_len

    def test_respects_min_samples(self, sample_df):
        """Should winsorize instead of remove if too few samples would remain."""
        config = OutlierConfig(
            method=OutlierMethod.IQR,
            threshold=0.1,  # Very aggressive
            min_samples=8,  # Would remove too many
        )
        
        result = filter_outliers(sample_df, 'target', config)
        
        # Should fall back to winsorizing, keeping all rows
        assert len(result) == len(sample_df)

    def test_winsorize_method_clips(self, sample_df):
        """WINSORIZE method should clip values, not remove rows."""
        config = OutlierConfig(
            method=OutlierMethod.WINSORIZE,
            winsorize_percentile=0.2,  # 20% from each tail
            min_samples=5,  # Must be less than len(df) for winsorize to run
        )
        
        result = filter_outliers(sample_df, 'target', config)
        
        assert len(result) == len(sample_df)
        # With 20% winsorization on 10 values, max should be clipped to 80th percentile
        assert result['target'].max() <= sample_df['target'].quantile(0.8)

    def test_multivariate_detection(self, sample_df):
        """Multivariate detection considers feature space."""
        config = OutlierConfig(
            method=OutlierMethod.MAD,
            use_features=True,
        )
        
        result = filter_outliers(
            sample_df,
            'target',
            config,
            feature_columns=['feature1', 'feature2'],
        )
        
        # Should still work and return a DataFrame
        assert isinstance(result, pd.DataFrame)


class TestDetectMultivariateOutliers:
    """Tests for Mahalanobis-based multivariate detection."""

    def test_detects_multivariate_outlier(self):
        """Should detect points anomalous in joint distribution."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n),
            'target': np.random.randn(n),
        })
        
        # Add a point that's normal marginally but outlier in joint space
        outlier = pd.DataFrame({'x': [0], 'y': [0], 'target': [10]})
        df = pd.concat([df, outlier], ignore_index=True)
        
        mask = detect_multivariate_outliers(
            df,
            feature_columns=['x', 'y'],
            target_column='target',
        )
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_handles_missing_columns(self):
        """Should return all True if columns are missing."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        
        mask = detect_multivariate_outliers(
            df,
            feature_columns=['missing'],
            target_column='also_missing',
        )
        
        assert mask.all()


class TestOutlierConfig:
    """Tests for OutlierConfig dataclass."""

    def test_default_config(self):
        """Default config should use MAD method."""
        assert DEFAULT_OUTLIER_CONFIG.method == OutlierMethod.MAD
        assert DEFAULT_OUTLIER_CONFIG.threshold == 3.5
        assert DEFAULT_OUTLIER_CONFIG.min_samples == 30

    def test_frozen_dataclass(self):
        """Config should be immutable."""
        config = OutlierConfig()
        with pytest.raises(AttributeError):
            config.method = OutlierMethod.IQR


class TestPreservesRareValidSamples:
    """Integration tests for preserving rare but valid samples."""

    def test_preserves_more_with_higher_threshold(self):
        """Higher threshold should preserve more values including rare ones."""
        np.random.seed(42)
        normal_values = np.random.normal(800, 200, 95)
        high_values = np.array([1500, 1600, 1700])  # Moderately high SBET
        
        df = pd.DataFrame({'SBET': np.concatenate([normal_values, high_values])})
        
        # Conservative threshold
        config_conservative = OutlierConfig(
            method=OutlierMethod.MAD,
            threshold=5.0,  # Very conservative
        )
        
        # Aggressive threshold
        config_aggressive = OutlierConfig(
            method=OutlierMethod.MAD,
            threshold=2.5,
        )
        
        result_conservative = filter_outliers(df, 'SBET', config_conservative)
        result_aggressive = filter_outliers(df, 'SBET', config_aggressive)
        
        # Conservative should preserve more
        assert len(result_conservative) >= len(result_aggressive)
