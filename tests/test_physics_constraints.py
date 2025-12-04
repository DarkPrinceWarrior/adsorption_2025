import numpy as np
import pandas as pd
import pytest

from src.adsorb_synthesis.constants import DEFAULT_STOICHIOMETRY_BOUNDS
from src.adsorb_synthesis.physics_losses import (
    DEFAULT_PHYSICS_EVALUATOR,
    EqualityConstraint,
    InequalityConstraint,
    PhysicsConstraintEvaluator,
    RatioConstraint,
    project_thermodynamics,
)
from src.adsorb_synthesis.pipeline import (
    _enforce_temperature_limits,
    _project_stoichiometry,
    _update_stoichiometry_features,
)


def test_project_thermodynamics_aligns_k_and_delta_g():
    df = pd.DataFrame({
        'Т.син., °С': [100.0, 120.0],
        'Delta_G_equilibrium': [-10.0, -12.0],
        'K_equilibrium': [1.0, 1.0],
    })

    projected = project_thermodynamics(df, evaluator=DEFAULT_PHYSICS_EVALUATOR, overwrite=True)

    T_kelvin = projected['Т.син., °С'] + 273.15
    expected_k = np.exp(-(projected['Delta_G_equilibrium'] * 1000) / (DEFAULT_PHYSICS_EVALUATOR.gas_constant * T_kelvin))

    np.testing.assert_allclose(projected['K_equilibrium'], expected_k, rtol=1e-6, atol=0)


def test_project_stoichiometry_respects_targets_and_fallback():
    default_lower, default_upper = DEFAULT_STOICHIOMETRY_BOUNDS
    df = pd.DataFrame({
        'Металл': ['Cu', 'Cu', 'Xx'],
        'Лиганд': ['BTC', 'BTC', 'Unknown'],
        'm (соли), г': [10.0, 10.0, 10.0],
        'Молярка_соли': [100.0, 100.0, 100.0],
        'm(кис-ты), г': [13.333333, 8.0, 1.0],
        'Молярка_кислоты': [200.0, 200.0, 500.0],
    })
    _update_stoichiometry_features(df)
    _project_stoichiometry(df)
    _update_stoichiometry_features(df)

    # Row 0 already within tolerance so mass stays unchanged
    assert np.isclose(df.loc[0, 'n_ratio'], 1.5, atol=1e-6)
    assert abs(df.loc[0, 'n_ratio_residual']) < 1e-6

    # Row 1 is pushed to the exact Cu-BTC target ratio
    assert np.isclose(df.loc[1, 'n_ratio_target'], 1.5, atol=1e-6)
    assert abs(df.loc[1, 'n_ratio'] - 1.5) < 1e-6
    assert df.loc[1, 'm(кис-ты), г'] != pytest.approx(8.0)

    # Row 2 uses fallback bounds and is clipped to default upper limit
    assert np.isnan(df.loc[2, 'n_ratio_target'])
    assert np.isclose(df.loc[2, 'n_ratio'], default_upper, atol=1e-6)
    assert np.isclose(df.loc[2, 'n_ratio_upper'], default_upper, atol=1e-6)
    assert df.loc[2, 'n_ratio'] <= df.loc[2, 'n_ratio_upper']


def test_enforce_temperature_limits_monotonic_and_boiling():
    # Use exact category labels from TEMPERATURE_CATEGORIES in constants.py:
    # Tsyn_Category: ['Низкая (<115°C)', 'Средняя (115-135°C)', 'Высокая (>135°C)']
    # Tdry_Category: ['Низкая (<115°C)', 'Средняя (115-135°C)', 'Высокая (>135°C)']
    # Treg_Category: ['Низкая (<150°C)', 'Средняя (150-250°C)', 'Высокая (>250°C)']
    df = pd.DataFrame({
        'Tsyn_Category': ['Средняя (115-135°C)', 'Высокая (>135°C)'],
        'Tdry_Category': ['Низкая (<115°C)', 'Высокая (>135°C)'],
        'Treg_Category': ['Низкая (<150°C)', 'Средняя (150-250°C)'],  # Fixed label
        'Растворитель': ['ДМФА', 'Метанол'],
        'Т.син., °С': [np.nan, np.nan],
        'Т суш., °С': [np.nan, np.nan],
        'Tрег, ᵒС': [np.nan, np.nan],
    })

    _enforce_temperature_limits(df)

    # Row 0: drying/regeneration cannot be below synthesis category
    assert df.loc[0, 'Tdry_Category'] != 'Низкая (<115°C)'
    assert df.loc[0, 'Т суш., °С'] >= df.loc[0, 'Т.син., °С']
    assert df.loc[0, 'Tрег, ᵒС'] >= df.loc[0, 'Т суш., °С']

    # Row 1: synthesis temperature must stay below methanol boiling point (~65°C)
    assert df.loc[1, 'Tsyn_Category'] == 'Низкая (<115°C)'
    assert df.loc[1, 'Т.син., °С'] < 65.0
    assert df.loc[1, 'Т суш., °С'] >= df.loc[1, 'Т.син., °С']
    assert df.loc[1, 'Tрег, ᵒС'] >= df.loc[1, 'Т суш., °С']


def test_physics_evaluator_enforces_new_constraint_types():
    df = pd.DataFrame({
        'W0, см3/г': [0.5, 0.5],
        'а0, ммоль/г': [28.86 * 0.5, 28.86 * 0.5 * 1.2],
        'E0, кДж/моль': [30.0, 30.0],
        'E, кДж/моль': [10.0, 15.0],
        'Ws, см3/г': [0.6, 0.2],
    })

    evaluator = PhysicsConstraintEvaluator(
        equality_constraints=(
            EqualityConstraint(column_a='а0, ммоль/г', column_b='W0, см3/г', coefficient=28.86, tolerance=0.05),
        ),
        ratio_constraints=(
            RatioConstraint(column_a='E, кДж/моль', column_b='E0, кДж/моль', target_ratio=1.0 / 3.0, tolerance=0.05),
        ),
        inequality_constraints=(
            InequalityConstraint(column_left='Ws, см3/г', column_right='W0, см3/г', operator='gte', tolerance=0.0),
        ),
    )

    penalties = evaluator.penalties(df)
    assert penalties.shape == (2,)
    assert penalties[0] == pytest.approx(0.0)
    assert penalties[1] > penalties[0]
