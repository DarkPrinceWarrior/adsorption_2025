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
    SOLVENT_BOILING_POINTS_C,
    TEMPERATURE_CATEGORIES,
)
from .data_validation import (
    DEFAULT_VALIDATION_MODE,
    validate_SEH_data,
    validate_synthesis_data,
)
from .molar_masses import add_molar_mass_columns

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


def load_dataset(
    csv_path: str,
    *,
    add_categories: bool = True,
    add_salt_features: bool = True,
    validation_mode: str = DEFAULT_VALIDATION_MODE,
) -> pd.DataFrame:
    """Load the prepared dataset and ensure derived features exist."""

    df = pd.read_csv(csv_path)
    df = df.copy()

    add_molar_mass_columns(df)

    _ensure_adsorption_features(df)

    if add_categories:
        add_temperature_categories(df)
    
    if add_salt_features:
        add_salt_mass_features(df)
    
    add_thermodynamic_features(df)

    validate_SEH_data(df, mode=validation_mode)
    validate_synthesis_data(df, boiling_points=SOLVENT_BOILING_POINTS_C, mode=validation_mode)

    return df


def add_salt_mass_features(df: pd.DataFrame) -> None:
    """Add engineered features derived from synthesis parameters and adsorption checks."""

    metal_col = 'Металл'
    ligand_col = 'Лиганд'
    salt_mass_col = 'm (соли), г'
    acid_mass_col = 'm(кис-ты), г'
    solvent_vol_col = 'Vсин. (р-ля), мл'
    syn_temp_col = 'Т.син., °С'
    dry_temp_col = 'Т суш., °С'
    reg_temp_col = 'Tрег, ᵒС'

    # Metal-Ligand interaction combinations
    if metal_col in df.columns and ligand_col in df.columns:
        df['Metal_Ligand_Combo'] = df[metal_col].astype(str) + '_' + df[ligand_col].astype(str)

    if 'Total molecular weight (metal)' in df.columns:
        df['Log_Metal_MW'] = np.log1p(df['Total molecular weight (metal)'])

    if metal_col in df.columns:
        df['Is_Cu'] = (df[metal_col] == 'Cu').astype(int)
        df['Is_Zn'] = (df[metal_col] == 'Zn').astype(int)

    if salt_mass_col in df.columns:
        df['log_salt_mass'] = np.log1p(df[salt_mass_col])

    # Concentrations
    if solvent_vol_col in df.columns:
        solvent_vol = pd.to_numeric(df[solvent_vol_col], errors='coerce').replace(0, np.nan)
        if salt_mass_col in df.columns:
            df['C_metal'] = pd.to_numeric(df[salt_mass_col], errors='coerce') / solvent_vol
            df['log_C_metal'] = np.log1p(df['C_metal'].clip(lower=0))
        if acid_mass_col in df.columns:
            df['C_ligand'] = pd.to_numeric(df[acid_mass_col], errors='coerce') / solvent_vol
            df['log_C_ligand'] = np.log1p(df['C_ligand'].clip(lower=0))

    # Ratios
    if salt_mass_col in df.columns and acid_mass_col in df.columns:
        acid_mass = pd.to_numeric(df[acid_mass_col], errors='coerce')
        with np.errstate(divide='ignore', invalid='ignore'):
            df['R_mass'] = pd.to_numeric(df[salt_mass_col], errors='coerce') / acid_mass.replace(0, np.nan)

    molar_cols = {'Молярка_соли', 'Молярка_кислоты'}
    if {salt_mass_col, acid_mass_col}.issubset(df.columns) and molar_cols.issubset(df.columns):
        salt_mass = pd.to_numeric(df[salt_mass_col], errors='coerce')
        acid_mass = pd.to_numeric(df[acid_mass_col], errors='coerce')
        molar_salt = pd.to_numeric(df['Молярка_соли'], errors='coerce').replace(0, np.nan)
        molar_acid = pd.to_numeric(df['Молярка_кислоты'], errors='coerce').replace(0, np.nan)
        salt_moles = salt_mass / molar_salt
        ligand_moles = acid_mass / molar_acid
        df['R_molar'] = np.divide(
            salt_moles,
            ligand_moles,
            out=np.full_like(salt_moles, np.nan),
            where=ligand_moles != 0,
        )

    # Temperature-derived features
    if syn_temp_col in df.columns and reg_temp_col in df.columns:
        syn_temp = pd.to_numeric(df[syn_temp_col], errors='coerce')
        reg_temp = pd.to_numeric(df[reg_temp_col], errors='coerce')
        df['T_range'] = reg_temp - syn_temp
        df['T_activation'] = reg_temp - 100.0

        if dry_temp_col in df.columns:
            dry_temp = pd.to_numeric(df[dry_temp_col], errors='coerce')
            denom = (reg_temp - syn_temp).replace(0, np.nan)
            df['T_dry_norm'] = (dry_temp - syn_temp) / denom

    # Adsorption sanity-check features
    required_ads_cols = {'W0, см3/г', 'а0, ммоль/г', 'E0, кДж/моль', 'E, кДж/моль', 'Ws, см3/г'}
    if required_ads_cols.issubset(df.columns):
        W0 = pd.to_numeric(df['W0, см3/г'], errors='coerce')
        a0 = pd.to_numeric(df['а0, ммоль/г'], errors='coerce')
        E0 = pd.to_numeric(df['E0, кДж/моль'], errors='coerce')
        E = pd.to_numeric(df['E, кДж/моль'], errors='coerce')
        Ws = pd.to_numeric(df['Ws, см3/г'], errors='coerce')
        df['a0_calc'] = 28.86 * W0
        df['E_calc'] = E0 / 3.0
        df['Ws_W0_ratio'] = np.divide(Ws, W0, out=np.full_like(Ws, np.nan), where=W0 != 0)
        df['delta_a0'] = a0 - df['a0_calc']
        df['delta_E'] = E - df['E_calc']
        df['delta_Ws'] = Ws - W0
        df['E_E0_ratio'] = np.divide(E, E0, out=np.full_like(E, np.nan), where=E0 != 0)

    if {'W0, см3/г', 'SБЭТ, м2/г'}.issubset(df.columns):
        SBET = pd.to_numeric(df['SБЭТ, м2/г'], errors='coerce')
        W0 = pd.to_numeric(df['W0, см3/г'], errors='coerce')
        df['W0_per_SBET'] = np.divide(W0, SBET, out=np.full_like(W0, np.nan), where=SBET != 0)


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
