import numpy as np
import pandas as pd

from src.adsorb_synthesis.constants import N_RATIO_BOUNDS
from src.adsorb_synthesis.physics_losses import (
    DEFAULT_PHYSICS_EVALUATOR,
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


def test_project_stoichiometry_clips_ratio():
    df = pd.DataFrame({
        'm (соли), г': [10.0, 1.0],
        'Молярка_соли': [100.0, 50.0],
        'm(кис-ты), г': [1.0, 10.0],
        'Молярка_кислоты': [10.0, 20.0],
    })
    _update_stoichiometry_features(df)

    assert (df['n_ratio'] > 0).all()

    _project_stoichiometry(df)

    lower, upper = N_RATIO_BOUNDS
    assert df['n_ratio'].between(lower, upper).all()
    assert np.allclose(df['n_ratio_residual'].fillna(0.0), 0.0)


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

    assert (dry_ord >= syn_ord).fillna(True).all()
    assert (reg_ord >= dry_ord).fillna(True).all()
