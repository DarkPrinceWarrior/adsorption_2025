import numpy as np
import pandas as pd

from src.adsorb_synthesis.pipeline import (
    StageConfig,
    _apply_target_transform,
    _ensure_stage_target_column,
    _invert_target_transform,
)


def _dummy_estimator_factory(_seed: int):
    return None


def test_log1p_transform_round_trip():
    series = pd.Series([0.0, 1.0, 4.5], name='m (соли), г')

    transformed = _apply_target_transform(series, 'log1p')
    inverted = _invert_target_transform(transformed, 'log1p')

    np.testing.assert_allclose(transformed.to_numpy(), np.log1p(series.to_numpy()))
    np.testing.assert_allclose(inverted.to_numpy(), series.to_numpy())


def test_ensure_stage_target_column_builds_missing_log_column():
    df = pd.DataFrame({'m (соли), г': [0.0, 1.5, 3.0]})
    stage = StageConfig(
        name='salt_mass',
        target='log_salt_mass',
        problem_type='regression',
        feature_columns=(),
        estimator_factory=_dummy_estimator_factory,
        target_transform='log1p',
        invert_target_to='m (соли), г',
    )

    _ensure_stage_target_column(df, stage)

    assert 'log_salt_mass' in df.columns
    expected = np.log1p(df['m (соли), г'].to_numpy())
    np.testing.assert_allclose(df['log_salt_mass'].to_numpy(), expected)
