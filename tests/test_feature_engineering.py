import numpy as np
import pandas as pd

from src.adsorb_synthesis.data_processing import add_salt_mass_features


def test_add_salt_mass_features_populates_engineered_columns():
    df = pd.DataFrame({
        'Металл': ['Cu'],
        'Лиганд': ['BTC'],
        'm (соли), г': [2.0],
        'm(кис-ты), г': [1.0],
        'Vсин. (р-ля), мл': [10.0],
        'Молярка_соли': [241.60],
        'Молярка_кислоты': [210.14],
        'Т.син., °С': [130.0],
        'Т суш., °С': [140.0],
        'Tрег, ᵒС': [200.0],
        'W0, см3/г': [0.5],
        'а0, ммоль/г': [14.43],
        'E0, кДж/моль': [12.0],
        'E, кДж/моль': [4.0],
        'Ws, см3/г': [0.6],
        'SБЭТ, м2/г': [1000.0],
    })

    add_salt_mass_features(df)

    assert np.isclose(df.loc[0, 'C_metal'], 0.2)
    assert np.isclose(df.loc[0, 'C_ligand'], 0.1)
    assert np.isclose(df.loc[0, 'R_mass'], 2.0)

    expected_r_molar = (2.0 / 241.60) / (1.0 / 210.14)
    assert np.isclose(df.loc[0, 'R_molar'], expected_r_molar)

    assert np.isclose(df.loc[0, 'T_range'], 70.0)
    assert np.isclose(df.loc[0, 'T_activation'], 100.0)
    assert np.isclose(df.loc[0, 'T_dry_norm'], 10.0 / 70.0)

    assert np.isclose(df.loc[0, 'a0_calc'], 14.43, atol=1e-2)
    assert np.isclose(df.loc[0, 'delta_a0'], 0.0, atol=1e-2)
    assert np.isclose(df.loc[0, 'E_calc'], 4.0, atol=1e-6)
    assert np.isclose(df.loc[0, 'delta_E'], 0.0, atol=1e-6)
    assert np.isclose(df.loc[0, 'Ws_W0_ratio'], 1.2)
    assert np.isclose(df.loc[0, 'W0_per_SBET'], 0.0005)
