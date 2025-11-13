import pandas as pd
import pytest

from src.adsorb_synthesis.data_validation import validate_SEH_data, validate_synthesis_data


def test_validate_seh_data_detects_inconsistent_a0():
    df = pd.DataFrame({
        'W0, см3/г': [0.5],
        'а0, ммоль/г': [10.0],  # expected 14.43
        'E0, кДж/моль': [12.0],
        'E, кДж/моль': [4.0],
        'Ws, см3/г': [0.6],
    })

    report = validate_SEH_data(df, mode="warn")

    assert len(report.errors) == 1
    assert report.errors[0].column == 'а0, ммоль/г'


def test_validate_synthesis_data_strict_mode_raises_on_errors():
    df = pd.DataFrame({
        'm (соли), г': [1.0],
        'm(кис-ты), г': [-0.1],  # invalid
        'Vсин. (р-ля), мл': [0.0],  # invalid
        'Т.син., °С': [170.0],  # above DMF boiling point
        'Т суш., °С': [160.0],
        'Tрег, ᵒС': [180.0],
        'Растворитель': ['DMF'],
    })

    report = validate_synthesis_data(df, boiling_points={'DMF': 153.0}, mode="warn")
    assert len(report.errors) == 3

    with pytest.raises(ValueError):
        validate_synthesis_data(df, boiling_points={'DMF': 153.0}, mode="strict")
