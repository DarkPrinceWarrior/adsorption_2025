import numpy as np
import pandas as pd
import pytest

from src.adsorb_synthesis import molar_masses
from src.adsorb_synthesis.molar_masses import (
    METAL_SALTS_MOLAR_MASSES,
    add_molar_mass_columns,
    metal_salt_molar_mass,
)


def test_add_molar_mass_columns_populates_expected_values():
    df = pd.DataFrame({
        'Металл': ['Cu', 'Zn'],
        'Лиганд': ['BTC', 'H2BDC'],
    })

    add_molar_mass_columns(df)

    assert np.isclose(df.loc[0, 'Молярка_соли'], METAL_SALTS_MOLAR_MASSES['Cu(NO3)2·3H2O'])
    assert np.isclose(df.loc[1, 'Молярка_соли'], METAL_SALTS_MOLAR_MASSES['Zn(NO3)2·6H2O'])
    assert np.isclose(df.loc[0, 'Молярка_кислоты'], 210.14)
    assert np.isclose(df.loc[1, 'Молярка_кислоты'], 166.13)


def test_existing_values_not_overwritten():
    df = pd.DataFrame({
        'Металл': ['Cu'],
        'Лиганд': ['BTC'],
        'Молярка_соли': [999.0],
        'Молярка_кислоты': [np.nan],
    })

    add_molar_mass_columns(df)

    # Existing non-null value should stay intact
    assert df.loc[0, 'Молярка_соли'] == 999.0
    # Missing value should be filled
    assert np.isclose(df.loc[0, 'Молярка_кислоты'], 210.14)


def test_metal_fallback_warns_when_typical_missing(monkeypatch):
    monkeypatch.delitem(molar_masses.TYPICAL_SALTS, 'Cu', raising=False)
    with pytest.warns(RuntimeWarning):
        mass, salt = metal_salt_molar_mass('Cu', warn_on_fallback=True)
        assert salt == molar_masses.ANHYDROUS_FALLBACKS['Cu']
        assert np.isclose(mass, molar_masses.METAL_SALTS_MOLAR_MASSES[salt])
