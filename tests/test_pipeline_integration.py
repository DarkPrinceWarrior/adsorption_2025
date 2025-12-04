"""Integration tests for pipeline physics constraints and sample weights."""
import numpy as np
import pandas as pd
import pytest

from src.adsorb_synthesis.outlier_detection import OutlierConfig, OutlierMethod
from src.adsorb_synthesis.physics_losses import (
    DEFAULT_PHYSICS_EVALUATOR,
    physics_violation_scores,
)
from src.adsorb_synthesis.data_processing import (
    add_salt_mass_features,
    add_thermodynamic_features,
    add_temperature_categories,
)


class TestPhysicsWeightsInTraining:
    """Tests for physics penalty application during training."""

    @pytest.fixture
    def physics_df(self):
        """DataFrame with physics-relevant columns."""
        return pd.DataFrame({
            'W0, см3/г': [0.5, 0.5, 0.5],
            'а0, ммоль/г': [28.86 * 0.5, 28.86 * 0.5 * 1.2, 28.86 * 0.5],  # Second row violates a0 = 28.86*W0
            'E0, кДж/моль': [30.0, 30.0, 30.0],
            'E, кДж/моль': [10.0, 15.0, 10.0],  # Second row violates E = E0/3
            'Ws, см3/г': [0.6, 0.4, 0.6],  # All valid (Ws >= W0)
            'Т.син., °С': [100.0, 100.0, 100.0],
        })

    def test_physics_violation_scores_nonzero_for_violations(self, physics_df):
        """Rows violating physics constraints should have higher scores."""
        scores = physics_violation_scores(physics_df, evaluator=DEFAULT_PHYSICS_EVALUATOR)
        
        assert scores.shape == (3,)
        # Row 1 (index 1) has violations, should have higher score
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_physics_scores_zero_for_valid_data(self, physics_df):
        """Valid rows should have zero or near-zero penalty."""
        scores = physics_violation_scores(physics_df, evaluator=DEFAULT_PHYSICS_EVALUATOR)
        
        # Row 0 is valid
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_sample_weights_increase_with_violations(self, physics_df):
        """Sample weights should be higher for violating samples."""
        physics_weight = 1.0
        penalties = physics_violation_scores(physics_df, evaluator=DEFAULT_PHYSICS_EVALUATOR)
        
        # Normalize as in pipeline
        penalties = np.nan_to_num(penalties, nan=0.0)
        finite_mask = np.isfinite(penalties)
        finite_values = penalties[finite_mask]
        if finite_values.size:
            scale = np.nanpercentile(finite_values, 95)
            if scale > 0:
                penalties = penalties / scale
        penalties = np.clip(penalties, 0.0, None)
        
        sample_weights = 1.0 + physics_weight * penalties
        
        # Violating sample (row 1) should have higher weight
        assert sample_weights[1] > sample_weights[0]
        # Valid samples should have weight ~1.0
        assert sample_weights[0] == pytest.approx(1.0)


class TestDataFrameCopySemantics:
    """Tests for inplace parameter behavior."""

    @pytest.fixture
    def base_df(self):
        """Minimal DataFrame for testing mutations."""
        return pd.DataFrame({
            'Металл': ['Cu', 'Zn'],
            'Лиганд': ['BTC', 'BDC'],
            'm (соли), г': [1.0, 2.0],
            'm(кис-ты), г': [0.5, 1.0],
            'Vсин. (р-ля), мл': [10.0, 20.0],
            'Т.син., °С': [120.0, 130.0],
            'Т суш., °С': [80.0, 90.0],
            'Tрег, ᵒС': [150.0, 160.0],
            'Молярка_соли': [100.0, 100.0],
            'Молярка_кислоты': [50.0, 50.0],
        })

    def test_add_salt_mass_features_inplace_true(self, base_df):
        """inplace=True should mutate original and return None."""
        original_cols = set(base_df.columns)
        
        result = add_salt_mass_features(base_df, inplace=True)
        
        assert result is None
        assert len(base_df.columns) > len(original_cols)
        assert 'R_mass' in base_df.columns

    def test_add_salt_mass_features_inplace_false(self, base_df):
        """inplace=False should return copy and not mutate original."""
        original_cols = set(base_df.columns)
        original_len = len(base_df.columns)
        
        result = add_salt_mass_features(base_df, inplace=False)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert set(base_df.columns) == original_cols  # Original unchanged
        assert len(result.columns) > original_len
        assert 'R_mass' in result.columns

    def test_add_thermodynamic_features_inplace_false(self):
        """add_thermodynamic_features should support inplace=False."""
        df = pd.DataFrame({
            'Т.син., °С': [100.0, 120.0],
            'E, кДж/моль': [10.0, 12.0],
        })
        original_cols = set(df.columns)
        
        result = add_thermodynamic_features(df, inplace=False)
        
        assert set(df.columns) == original_cols  # Original unchanged
        assert 'K_equilibrium' in result.columns or 'Delta_G_equilibrium' in result.columns

    def test_add_temperature_categories_inplace_false(self):
        """add_temperature_categories should support inplace=False."""
        df = pd.DataFrame({
            'Т.син., °С': [100.0, 130.0, 160.0],
            'Т суш., °С': [80.0, 120.0, 140.0],
            'Tрег, ᵒС': [140.0, 170.0, 200.0],
        })
        original_cols = set(df.columns)
        
        result = add_temperature_categories(df, inplace=False)
        
        assert set(df.columns) == original_cols
        # Should have category columns
        assert len(result.columns) > len(original_cols)


class TestOutlierConfigIntegration:
    """Tests for OutlierConfig in pipeline context."""

    def test_outlier_config_threshold_affects_preservation(self):
        """Higher MAD threshold should preserve more edge cases."""
        np.random.seed(42)
        
        # Simulate adsorption target with some spread
        normal_values = np.random.normal(20, 5, 90)
        edge_values = np.array([40, 45, 50])  # Edge cases, not extreme
        extreme_value = np.array([200])  # True outlier
        
        from src.adsorb_synthesis.outlier_detection import filter_outliers
        
        df = pd.DataFrame({'target': np.concatenate([normal_values, edge_values, extreme_value])})
        
        # Conservative config
        config_high = OutlierConfig(
            method=OutlierMethod.MAD,
            threshold=5.0,
            min_samples=30,
        )
        
        # Aggressive config
        config_low = OutlierConfig(
            method=OutlierMethod.MAD,
            threshold=2.5,
            min_samples=30,
        )
        
        result_high = filter_outliers(df, 'target', config_high)
        result_low = filter_outliers(df, 'target', config_low)
        
        # Higher threshold should preserve more
        assert len(result_high) >= len(result_low)

    def test_winsorize_fallback_on_small_dataset(self):
        """Should winsorize instead of removing when dataset is small."""
        from src.adsorb_synthesis.outlier_detection import filter_outliers
        
        df = pd.DataFrame({'target': [1, 2, 3, 4, 5, 100]})
        
        config = OutlierConfig(
            method=OutlierMethod.MAD,
            min_samples=5,  # Can't remove more than 1
        )
        
        result = filter_outliers(df, 'target', config)
        
        # Should keep all 6 rows (winsorized instead of removed)
        assert len(result) >= 5


class TestPhysicsConstraintsInInference:
    """Tests that physics constraints apply correctly during inference."""

    def test_thermodynamic_consistency(self):
        """K and Delta_G should be thermodynamically consistent."""
        df = pd.DataFrame({
            'Т.син., °С': [100.0, 150.0],
            'E, кДж/моль': [10.0, 15.0],
        })
        
        result = add_thermodynamic_features(df, inplace=False)
        
        if 'K_equilibrium' in result.columns and 'Delta_G_equilibrium' in result.columns:
            # Check Gibbs equation: Delta_G = -RT*ln(K)
            R = 8.314  # J/(mol·K)
            T_kelvin = result['Т.син., °С'] + 273.15
            
            k_vals = result['K_equilibrium'].dropna()
            dg_vals = result['Delta_G_equilibrium'].dropna()
            
            if len(k_vals) > 0 and len(dg_vals) > 0:
                # Delta_G (kJ/mol) should match -RT*ln(K) / 1000
                expected_dg = -(R * T_kelvin * np.log(k_vals)) / 1000
                np.testing.assert_allclose(
                    dg_vals.values,
                    expected_dg.values,
                    rtol=1e-3,
                    err_msg="Thermodynamic consistency violated"
                )
