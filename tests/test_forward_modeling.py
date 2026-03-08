import numpy as np
import pandas as pd

from src.adsorb_synthesis.data_processing import add_interaction_features
from src.adsorb_synthesis.forward_modeling import select_curated_features


def test_add_interaction_features_is_row_level_for_same_metal():
    df = pd.DataFrame({
        "Металл": ["Cu", "Cu"],
        "Лиганд": ["BTC", "BDC"],
        "ionic_radius_pm (metal_coord)": [73.0, 73.0],
        "RadiusOfGyration (ligand_3d)": [2.0, 4.0],
        "electronegativity_pauling (metal_coord)": [1.90, 1.90],
        "d_electrons (metal_coord)": [9, 9],
    })

    add_interaction_features(df, inplace=True)

    assert np.isclose(df.loc[0, "Metal_Ligand_Size_Ratio"], 0.365)
    assert np.isclose(df.loc[1, "Metal_Ligand_Size_Ratio"], 0.1825)
    assert df.loc[0, "Metal_Ligand_Size_Ratio"] != df.loc[1, "Metal_Ligand_Size_Ratio"]
    assert np.isclose(df.loc[0, "Metal_O_Electronegativity_Diff"], 1.54)
    assert df.loc[0, "Jahn_Teller_Active"] == 1


def test_select_curated_features_keeps_primary_features():
    n_rows = 32
    r_molar = np.linspace(0.5, 2.0, n_rows)
    c_metal = np.linspace(0.1, 1.0, n_rows)
    x = pd.DataFrame({
        "R_molar": r_molar,
        "C_metal": c_metal,
        "Vсин. (р-ля), мл": np.linspace(10.0, 50.0, n_rows),
        "R_mass": r_molar * 1.05,
        "noise": np.sin(np.linspace(0, 3.14, n_rows)),
    })
    y = pd.Series(5 * r_molar + 2 * c_metal + np.linspace(0, 0.1, n_rows))

    selected_features, report = select_curated_features(
        x,
        y,
        categorical_cols=[],
        corr_threshold=0.85,
        vif_threshold=10.0,
        max_features=2,
        verbose=False,
    )

    assert "R_molar" in selected_features
    assert "C_metal" in selected_features
    assert "R_mass" not in report["available_keep_features"]
    assert "R_mass" not in selected_features
