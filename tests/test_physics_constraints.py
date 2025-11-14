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
    _enforce_temperature_order,
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
        'Металл': ['Cu', 'Xx'],
        'Лиганд': ['BTC', 'Unknown'],
        'm (соли), г': [10.0, 10.0],
        'Молярка_соли': [100.0, 100.0],
        'm(кис-ты), г': [10.0, 25.0],
        'Молярка_кислоты': [200.0, 500.0],
    })
    _update_stoichiometry_features(df)
    _project_stoichiometry(df)
    _update_stoichiometry_features(df)

    # Row 0 should be projected to Cu-BTC target (1.5 ± tol)
    target_ratio = df.loc[0, 'n_ratio_target']
    assert np.isclose(target_ratio, 1.5, atol=1e-6)
    assert abs(df.loc[0, 'n_ratio_residual']) < 1e-6

    # Row 1 uses fallback bounds
    assert np.isnan(df.loc[1, 'n_ratio_target'])
    assert df.loc[1, 'n_ratio_lower'] <= df.loc[1, 'n_ratio'] <= df.loc[1, 'n_ratio_upper']
    assert np.isclose(df.loc[1, 'n_ratio_lower'], default_lower)
    assert np.isclose(df.loc[1, 'n_ratio_upper'], default_upper)
    assert abs(df.loc[1, 'n_ratio_residual']) < 1e-6


def test_enforce_temperature_order_monotonic():
    df = pd.DataFrame({
        'Tsyn_Category': ['Низкая (<115°C)', 'Средняя (115-135°C)', np.nan],
        'Tdry_Category': ['Низкая (<115°C)', 'Низкая (<115°C)', 'Высокая (>135°C)'],
        'Treg_Category': ['Высокая (>265°C)', 'Средняя (155-265°C)', 'Низкая (<155°C)'],
    })

    _enforce_temperature_order(df)

    categories = ['Низкая (<115°C)', 'Средняя (115-135°C)', 'Высокая (>135°C)']
    reg_categories = ['Низкая (<155°C)', 'Средняя (155-265°C)', 'Высокая (>265°C)']

    def order(series, order_list):
        mapping = {label: idx for idx, label in enumerate(order_list)}
        return series.map(mapping)

    syn_ord = order(df['Tsyn_Category'], categories)
    dry_ord = order(df['Tdry_Category'], categories)
    reg_ord = order(df['Treg_Category'], reg_categories)

    dry_ok = (dry_ord >= syn_ord) | syn_ord.isna()
    reg_ok = (reg_ord >= dry_ord) | dry_ord.isna()
    assert dry_ok.fillna(True).all()
    assert reg_ok.fillna(True).all()


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
