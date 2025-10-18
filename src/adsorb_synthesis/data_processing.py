"""Utilities for loading and enriching the adsorption synthesis dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .constants import (
    ADSORPTION_FEATURES,
    LIGAND_DESCRIPTOR_FEATURES,
    METAL_DESCRIPTOR_FEATURES,
    SOLVENT_DESCRIPTOR_FEATURES,
    TEMPERATURE_CATEGORIES,
)

EPSILON = 1e-9
R_GAS_CONSTANT = 8.314  # J/(mol*K)
TEMPERATURE_DEFAULT_K = 298.15


@dataclass(frozen=True)
class LookupTables:
    """Collections of descriptor lookups keyed by entity name."""

    metal: pd.DataFrame
    ligand: pd.DataFrame
    solvent: pd.DataFrame

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            "metal": self.metal.copy(),
            "ligand": self.ligand.copy(),
            "solvent": self.solvent.copy(),
        }


def load_dataset(csv_path: str, *, add_categories: bool = True, add_salt_features: bool = True) -> pd.DataFrame:
    """Load the prepared dataset and ensure derived features exist."""

    df = pd.read_csv(csv_path)
    df = df.copy()

    _ensure_adsorption_features(df)

    if add_categories:
        add_temperature_categories(df)
    
    if add_salt_features:
        add_salt_mass_features(df)
    
    add_thermodynamic_features(df)

    return df


def add_salt_mass_features(df: pd.DataFrame) -> None:
    """Add engineered features for salt_mass prediction."""
    
    # Metal-Ligand interaction (categorical combination)
    if 'Металл' in df.columns and 'Лиганд' in df.columns:
        df['Metal_Ligand_Combo'] = df['Металл'].astype(str) + '_' + df['Лиганд'].astype(str)
    
    # Log-transformed molecular weights if available
    if 'Total molecular weight (metal)' in df.columns:
        df['Log_Metal_MW'] = np.log1p(df['Total molecular weight (metal)'])
    
    # Metal-specific indicator for Cu (high salt mass)
    if 'Металл' in df.columns:
        df['Is_Cu'] = (df['Металл'] == 'Cu').astype(int)
        df['Is_Zn'] = (df['Металл'] == 'Zn').astype(int)
    
    # Log-transform salt mass target (handles extreme right-skew)
    if 'm (соли), г' in df.columns:
        df['log_salt_mass'] = np.log1p(df['m (соли), г'])  # log1p = log(1 + x)


def add_thermodynamic_features(df: pd.DataFrame) -> None:
    """Derive thermodynamic helper columns from measured K_eq and ΔG."""

    temperature_col = 'Т.син., °С'
    if temperature_col not in df.columns or 'K_equilibrium' not in df.columns:
        return

    temperature_c = pd.to_numeric(df[temperature_col], errors='coerce').to_numpy(dtype=np.float64)
    temperature_k = temperature_c + 273.15
    k_measured = pd.to_numeric(df['K_equilibrium'], errors='coerce').to_numpy(dtype=np.float64)

    delta_g_from_k = np.full(len(df), np.nan, dtype=np.float64)
    valid_k = np.isfinite(temperature_k) & np.isfinite(k_measured) & (k_measured > 0)
    if np.any(valid_k):
        delta_g_from_k[valid_k] = -(
            R_GAS_CONSTANT * temperature_k[valid_k] * np.log(k_measured[valid_k])
        ) / 1000.0
    df['Delta_G_equilibrium'] = delta_g_from_k

    if 'Delta_G' in df.columns:
        delta_g = pd.to_numeric(df['Delta_G'], errors='coerce').to_numpy(dtype=np.float64)
        k_from_delta = np.full(len(df), np.nan, dtype=np.float64)
        valid_delta = np.isfinite(temperature_k) & np.isfinite(delta_g)
        if np.any(valid_delta):
            k_from_delta[valid_delta] = np.exp(
                -(delta_g[valid_delta] * 1000.0) / (R_GAS_CONSTANT * temperature_k[valid_delta])
            )
        df['K_equilibrium_from_delta_G'] = k_from_delta

        residual = np.full(len(df), np.nan, dtype=np.float64)
        valid_both = valid_k & valid_delta
        if np.any(valid_both):
            residual[valid_both] = delta_g[valid_both] - delta_g_from_k[valid_both]
        df['Delta_G_residual'] = residual
        df['K_equilibrium_ratio'] = np.divide(
            k_measured,
            k_from_delta,
            out=np.full(len(df), np.nan, dtype=np.float64),
            where=np.isfinite(k_from_delta) & (k_from_delta != 0),
        )


def add_temperature_categories(df: pd.DataFrame) -> None:
    """Add categorical temperature buckets used for classification models."""

    for category_name, spec in TEMPERATURE_CATEGORIES.items():
        column = spec["column"]
        bins = spec["bins"]
        labels = spec["labels"]
        if column not in df.columns:
            continue
        df[category_name] = pd.cut(df[column], bins=bins, labels=labels, right=False)


def build_lookup_tables(df: pd.DataFrame) -> LookupTables:
    """Create lookup tables for descriptor features used during inference."""

    missing_cols: Dict[str, Iterable[str]] = {
        "metal": _missing_columns(df, METAL_DESCRIPTOR_FEATURES + ["Металл"]),
        "ligand": _missing_columns(df, LIGAND_DESCRIPTOR_FEATURES + ["Лиганд"]),
        "solvent": _missing_columns(df, SOLVENT_DESCRIPTOR_FEATURES + ["Растворитель"]),
    }
    for entity, cols in missing_cols.items():
        if cols:
            raise ValueError(f"Dataset is missing required columns {sorted(cols)} for {entity} lookup")

    metal_table = (
        df[["Металл", *METAL_DESCRIPTOR_FEATURES]].drop_duplicates().set_index("Металл").sort_index()
    )
    ligand_table = (
        df[["Лиганд", *LIGAND_DESCRIPTOR_FEATURES]].drop_duplicates().set_index("Лиганд").sort_index()
    )
    solvent_table = (
        df[["Растворитель", *SOLVENT_DESCRIPTOR_FEATURES]].drop_duplicates().set_index("Растворитель").sort_index()
    )

    return LookupTables(metal=metal_table, ligand=ligand_table, solvent=solvent_table)


def _ensure_adsorption_features(df: pd.DataFrame) -> None:
    """Compute engineered adsorption descriptors when they are missing."""

    required_base = {
        'E, кДж/моль',
        'Ws, см3/г',
        'а0, ммоль/г',
        'SБЭТ, м2/г',
        'W0, см3/г',
        'E0, кДж/моль',
        'х0, нм',
    }
    if not required_base.issubset(df.columns):
        missing = required_base.difference(df.columns)
        raise ValueError(
            "Dataset is missing base adsorption columns required for feature engineering: "
            f"{missing}"
        )

    if 'Adsorption_Potential' not in df.columns:
        df['Adsorption_Potential'] = df['E, кДж/моль'] * df['Ws, см3/г']
    if 'Capacity_Density' not in df.columns:
        df['Capacity_Density'] = df['а0, ммоль/г'] / (df['SБЭТ, м2/г'] + EPSILON)
    if 'SurfaceArea_MicroVol_Ratio' not in df.columns:
        df['SurfaceArea_MicroVol_Ratio'] = df['SБЭТ, м2/г'] / (df['W0, см3/г'] + EPSILON)
    if 'Adsorption_Energy_Ratio' not in df.columns:
        df['Adsorption_Energy_Ratio'] = df['E, кДж/моль'] / (df['E0, кДж/моль'] + EPSILON)
    if 'S_BET_E' not in df.columns:
        df['S_BET_E'] = df['SБЭТ, м2/г'] * df['E, кДж/моль']
    if 'x0_W0' not in df.columns:
        df['x0_W0'] = df['х0, нм'] * df['W0, см3/г']

    if 'K_equilibrium' not in df.columns or 'Delta_G' not in df.columns:
        R_kj = R_GAS_CONSTANT / 1000
        df['K_equilibrium'] = np.exp(df['E, кДж/моль'] / (R_kj * TEMPERATURE_DEFAULT_K))
        df['Delta_G'] = -R_kj * TEMPERATURE_DEFAULT_K * np.log(df['K_equilibrium'] + EPSILON)

    if 'B_micropore' not in df.columns:
        e_j_per_mol = df['E, кДж/моль'] * 1000
        df['B_micropore'] = ((2.303 * R_GAS_CONSTANT) / (e_j_per_mol + EPSILON)) ** 2

    expected = set(ADSORPTION_FEATURES)
    missing_after = expected.difference(df.columns)
    if missing_after:
        raise ValueError(
            "Missing engineered adsorption features after processing: "
            f"{sorted(missing_after)}"
        )

def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> Iterable[str]:
    return [col for col in columns if col not in df.columns]
