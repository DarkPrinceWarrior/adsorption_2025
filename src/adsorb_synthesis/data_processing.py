"""Utilities for loading and enriching the adsorption synthesis dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union, overload
from typing_extensions import Literal

import numpy as np
import pandas as pd

from .constants import (
    ADSORPTION_FEATURES,
    LIGAND_DESCRIPTOR_FEATURES,
    METAL_DESCRIPTOR_FEATURES,
    SOLVENT_DESCRIPTOR_FEATURES,
    SOLVENT_BOILING_POINTS_C,
    TEMPERATURE_CATEGORIES,
    FORWARD_MODEL_INPUTS,
    FORWARD_MODEL_TARGETS,
    FORWARD_MODEL_ENGINEERED_FEATURES,
    METAL_COORD_FEATURES,
    LIGAND_3D_FEATURES,
    LIGAND_2D_FEATURES,
    INTERACTION_FEATURES,
    HYDRATION_MAP,
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


def prepare_forward_dataset(
    df: pd.DataFrame,
    lookup_tables: Optional[LookupTables] = None,
    drop_missing_targets: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features (X) and targets (y) for the Forward Model (Recipe -> Properties).
    
    This function implements 'Stage 1: Data Flip' from the Bayesian Optimization plan.
    
    Args:
        df: The loaded dataset containing synthesis parameters and properties.
        lookup_tables: Optional tables to enrich X with physical descriptors of ingredients.
        drop_missing_targets: If True, rows with missing values in ANY target column are dropped.
        
    Returns:
        X: DataFrame of input features (Recipe parameters + optional descriptors).
        y: DataFrame of target properties (W0, E0, S_BET, x0).
    """
    df = df.copy()
    
    # 1. Basic Input Features
    # Ensure we have all required inputs
    missing_inputs = [col for col in FORWARD_MODEL_INPUTS if col not in df.columns]
    if missing_inputs:
        raise ValueError(f"Dataset missing required forward model inputs: {missing_inputs}")
        
    X = df[FORWARD_MODEL_INPUTS].copy()
    
    # 2. Feature Engineering (Log transforms for skewed continuous variables)
    # As suggested in the plan: masses and volumes usually follow log-normal distributions
    continuous_vars_to_log = [
        'm (соли), г',
        'm(кис-ты), г',
        'Vсин. (р-ля), мл'
    ]
    
    for col in continuous_vars_to_log:
        if col in X.columns:
            # log1p is safer for zeros (though mass shouldn't be zero)
            X[f'log_{col}'] = np.log1p(pd.to_numeric(X[col], errors='coerce').clip(lower=0))
            
    # 3. Enrich with Descriptors (if provided)
    # This adds 'Total molecular weight', 'polarizability', etc. to X
    if lookup_tables:
        # Merge Metal Descriptors
        if 'Металл' in X.columns:
            X = X.merge(lookup_tables.metal, on='Металл', how='left')
            
        # Merge Ligand Descriptors
        if 'Лиганд' in X.columns:
            X = X.merge(lookup_tables.ligand, on='Лиганд', how='left')
            
        # Merge Solvent Descriptors
        if 'Растворитель' in X.columns:
            X = X.merge(lookup_tables.solvent, on='Растворитель', how='left')
    
    # 4. Copy Engineered Features from df (already computed by add_salt_mass_features)
    # These are critical for W0 prediction - CatBoost struggles to learn ratios
    for col in FORWARD_MODEL_ENGINEERED_FEATURES:
        if col in df.columns:
            X[col] = df.loc[X.index, col]
    
    # 4b. Copy NEW physicochemical descriptors if present in df
    # These are computed by scripts/enrich_descriptors.py
    new_descriptor_cols = METAL_COORD_FEATURES + LIGAND_3D_FEATURES + LIGAND_2D_FEATURES + INTERACTION_FEATURES
    for col in new_descriptor_cols:
        if col in df.columns and col not in X.columns:
            X[col] = df.loc[X.index, col]
            
    # 5. Prepare Targets
    missing_targets = [col for col in FORWARD_MODEL_TARGETS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Dataset missing required targets: {missing_targets}")
        
    y = df[FORWARD_MODEL_TARGETS].copy()
    
    # Convert targets to numeric, coercing errors to NaN
    for col in y.columns:
        y[col] = pd.to_numeric(y[col], errors='coerce')
    
    # 5. Clean up
    if drop_missing_targets:
        # Align X and y indices after dropping NaNs in y
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
    return X, y


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
    
    # Add thermodynamic features BEFORE validation (creates K_equilibrium, Delta_G)
    add_thermodynamic_features(df)

    _ensure_adsorption_features(df)

    if add_categories:
        add_temperature_categories(df)
    
    if add_salt_features:
        add_salt_mass_features(df)
    
    add_physicochemical_descriptors(df)

    validate_SEH_data(df, mode=validation_mode)
    validate_synthesis_data(df, boiling_points=SOLVENT_BOILING_POINTS_C, mode=validation_mode)

    return df


@overload
def add_salt_mass_features(df: pd.DataFrame, *, inplace: Literal[True] = ...) -> None: ...
@overload
def add_salt_mass_features(df: pd.DataFrame, *, inplace: Literal[False]) -> pd.DataFrame: ...

def add_salt_mass_features(df: pd.DataFrame, *, inplace: bool = True) -> Optional[pd.DataFrame]:
    """Add engineered features derived from synthesis parameters and adsorption checks.
    
    Args:
        df: Input DataFrame.
        inplace: If True (default), mutate df in place. If False, return a copy.
        
    Returns:
        None if inplace=True, otherwise a new DataFrame with added features.
    """
    if not inplace:
        df = df.copy()

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
        salt_mass = pd.to_numeric(df[salt_mass_col], errors='coerce')
        df['log_salt_mass'] = np.log1p(salt_mass.clip(lower=0))

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

    # Temperature-derived features (Tрег removed - not available in merged dataset)
    # Only compute if reg_temp_col exists (legacy datasets)
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
    numeric_candidates = required_ads_cols | {'SБЭТ, м2/г'}
    available_numeric = [col for col in numeric_candidates if col in df.columns]
    numeric_adsorption: Dict[str, np.ndarray] = {}
    if available_numeric:
        numeric_block = df[available_numeric].apply(pd.to_numeric, errors='coerce')
        for column in available_numeric:
            numeric_adsorption[column] = numeric_block[column].to_numpy(dtype=np.float64, copy=False)

    if required_ads_cols.issubset(df.columns):
        W0 = numeric_adsorption['W0, см3/г']
        a0 = numeric_adsorption['а0, ммоль/г']
        E0 = numeric_adsorption['E0, кДж/моль']
        E = numeric_adsorption['E, кДж/моль']
        Ws = numeric_adsorption['Ws, см3/г']
        df['a0_calc'] = 28.86 * W0
        df['E_calc'] = E0 / 3.0
        df['Ws_W0_ratio'] = np.divide(Ws, W0, out=np.full_like(Ws, np.nan), where=W0 != 0)
        df['delta_a0'] = a0 - df['a0_calc']
        df['delta_E'] = E - df['E_calc']
        df['delta_Ws'] = Ws - W0
        df['E_E0_ratio'] = np.divide(E, E0, out=np.full_like(E, np.nan), where=E0 != 0)

    if {'W0, см3/г', 'SБЭТ, м2/г'}.issubset(df.columns):
        SBET = numeric_adsorption['SБЭТ, м2/г']
        W0 = numeric_adsorption['W0, см3/г']
        df['W0_per_SBET'] = np.divide(W0, SBET, out=np.full_like(W0, np.nan), where=SBET != 0)
    
    if not inplace:
        return df
    return None


@overload
def add_thermodynamic_features(df: pd.DataFrame, *, inplace: Literal[True] = ...) -> None: ...
@overload
def add_thermodynamic_features(df: pd.DataFrame, *, inplace: Literal[False]) -> pd.DataFrame: ...

def add_thermodynamic_features(df: pd.DataFrame, *, inplace: bool = True) -> Optional[pd.DataFrame]:
    """Derive thermodynamic helper columns from measured K_eq and ΔG.
    
    Args:
        df: Input DataFrame.
        inplace: If True (default), mutate df in place. If False, return a copy.
        
    Returns:
        None if inplace=True, otherwise a new DataFrame with added features.
    """
    if not inplace:
        df = df.copy()

    temperature_col = 'Т.син., °С'
    if temperature_col not in df.columns:
        return df if not inplace else None

    temperature_c = pd.to_numeric(df[temperature_col], errors='coerce').to_numpy(dtype=np.float64)
    temperature_k = temperature_c + 273.15

    # Fill missing K_equilibrium using E (if available) and actual temperature
    if 'K_equilibrium' not in df.columns and 'E, кДж/моль' in df.columns:
        E = pd.to_numeric(df['E, кДж/моль'], errors='coerce').to_numpy(dtype=np.float64)
        fallback_k = np.full(len(df), np.nan, dtype=np.float64)
        valid = np.isfinite(E) & np.isfinite(temperature_k)
        if np.any(valid):
            R_kj = R_GAS_CONSTANT / 1000.0
            fallback_k[valid] = np.exp(E[valid] / (R_kj * temperature_k[valid]))
        df['K_equilibrium'] = fallback_k
    k_measured = pd.to_numeric(df.get('K_equilibrium'), errors='coerce').to_numpy(dtype=np.float64)

    delta_g_from_k = np.full(len(df), np.nan, dtype=np.float64)
    valid_k = np.isfinite(temperature_k) & np.isfinite(k_measured) & (k_measured > 0)
    if np.any(valid_k):
        delta_g_from_k[valid_k] = -(
            R_GAS_CONSTANT * temperature_k[valid_k] * np.log(k_measured[valid_k])
        ) / 1000.0
    df['Delta_G_equilibrium'] = delta_g_from_k

    delta_g_raw = None
    if 'Delta_G' in df.columns:
        delta_g_raw = pd.to_numeric(df['Delta_G'], errors='coerce').to_numpy(dtype=np.float64)
    elif 'E, кДж/моль' in df.columns:
        # fallback Delta_G estimate using adsorption energy (negative sign for spontaneous adsorption)
        delta_g_raw = -pd.to_numeric(df['E, кДж/моль'], errors='coerce').to_numpy(dtype=np.float64)
        df['Delta_G'] = delta_g_raw  # Create the column for downstream use

    if delta_g_raw is not None:
        k_from_delta = np.full(len(df), np.nan, dtype=np.float64)
        valid_delta = np.isfinite(temperature_k) & np.isfinite(delta_g_raw)
        if np.any(valid_delta):
            k_from_delta[valid_delta] = np.exp(
                -(delta_g_raw[valid_delta] * 1000.0) / (R_GAS_CONSTANT * temperature_k[valid_delta])
            )
        df['K_equilibrium_from_delta_G'] = k_from_delta

        residual = np.full(len(df), np.nan, dtype=np.float64)
        valid_both = valid_k & valid_delta
        if np.any(valid_both):
            residual[valid_both] = delta_g_raw[valid_both] - delta_g_from_k[valid_both]
        df['Delta_G_residual'] = residual
        df['K_equilibrium_ratio'] = np.divide(
            k_measured,
            k_from_delta,
            out=np.full(len(df), np.nan, dtype=np.float64),
            where=np.isfinite(k_from_delta) & (k_from_delta != 0),
        )
    
    if not inplace:
        return df
    return None


@overload
def add_temperature_categories(df: pd.DataFrame, *, inplace: Literal[True] = ...) -> None: ...
@overload
def add_temperature_categories(df: pd.DataFrame, *, inplace: Literal[False]) -> pd.DataFrame: ...

def add_temperature_categories(df: pd.DataFrame, *, inplace: bool = True) -> Optional[pd.DataFrame]:
    """Add categorical temperature buckets used for classification models.
    
    Args:
        df: Input DataFrame.
        inplace: If True (default), mutate df in place. If False, return a copy.
        
    Returns:
        None if inplace=True, otherwise a new DataFrame with added features.
    """
    if not inplace:
        df = df.copy()

    for category_name, spec in TEMPERATURE_CATEGORIES.items():
        column = spec["column"]
        bins = spec["bins"]
        labels = spec["labels"]
        if column not in df.columns:
            continue
        df[category_name] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    
    if not inplace:
        return df
    return None


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


@overload
def _ensure_adsorption_features(df: pd.DataFrame, *, inplace: Literal[True] = ...) -> None: ...
@overload
def _ensure_adsorption_features(df: pd.DataFrame, *, inplace: Literal[False]) -> pd.DataFrame: ...

def _ensure_adsorption_features(df: pd.DataFrame, *, inplace: bool = True) -> Optional[pd.DataFrame]:
    """Compute engineered adsorption descriptors when they are missing.
    
    Args:
        df: Input DataFrame.
        inplace: If True (default), mutate df in place. If False, return a copy.
        
    Returns:
        None if inplace=True, otherwise a new DataFrame with added features.
    """
    if not inplace:
        df = df.copy()

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

    if 'B_micropore' not in df.columns:
        e_j_per_mol = df['E, кДж/моль'] * 1000
        df['B_micropore'] = ((2.303 * R_GAS_CONSTANT) / (e_j_per_mol + EPSILON)) ** 2

    expected = set(ADSORPTION_FEATURES)
    missing_after = expected.difference(df.columns)
    if missing_after:
        raise ValueError(
            f"Missing engineered adsorption features after processing: "
            f"{sorted(missing_after)}"
        )
    
    if not inplace:
        return df
    return None


@overload
def add_physicochemical_descriptors(df: pd.DataFrame, *, inplace: Literal[True] = ...) -> None: ...
@overload
def add_physicochemical_descriptors(df: pd.DataFrame, *, inplace: Literal[False]) -> pd.DataFrame: ...

def add_physicochemical_descriptors(df: pd.DataFrame, *, inplace: bool = True) -> Optional[pd.DataFrame]:
    """
    Add physicochemical descriptors based on 'Hidden Water' and True Molarity.
    
    Algorithms:
    1. Calculate Moles (n) if missing.
    2. Calculate Hidden Water using HYDRATION_MAP.
    3. Calculate True Molarity (Molarity_Metal, Molarity_Ligand, Molarity_H2O_Hidden).
    4. Calculate Supersaturation Index and Reactor Loading.
    
    Args:
        df: Input DataFrame.
        inplace: If True (default), mutate df in place. If False, return a copy.
        
    Returns:
        None if inplace=True, otherwise a new DataFrame with added features.
    """
    if not inplace:
        df = df.copy()
    
    # Constants
    EPSILON = 1e-9
    
    # 1. Calculate Moles (n)
    # Note: 'm (соли), г' is hydrate mass
    if 'n_соли' not in df.columns and {'m (соли), г', 'Молярка_соли'}.issubset(df.columns):
        df['n_соли'] = pd.to_numeric(df['m (соли), г'], errors='coerce') / \
                       pd.to_numeric(df['Молярка_соли'], errors='coerce').replace(0, np.nan)
                       
    if 'n_кислоты' not in df.columns and {'m(кис-ты), г', 'Молярка_кислоты'}.issubset(df.columns):
        df['n_кислоты'] = pd.to_numeric(df['m(кис-ты), г'], errors='coerce') / \
                          pd.to_numeric(df['Молярка_кислоты'], errors='coerce').replace(0, np.nan)

    # 2. Hidden Water Calculations
    if 'Металл' in df.columns and 'n_соли' in df.columns:
        # Map hydration coefficient
        # Use 0.0 for unknown metals to avoid NaNs, or keep NaNs if critical? 
        # Prompt says "fill NaNs... with zeros" later, so we can use map and fillna(0)
        df['hydration_coeff'] = df['Металл'].map(HYDRATION_MAP).fillna(0.0)
        
        n_salt = pd.to_numeric(df['n_соли'], errors='coerce').fillna(0.0)
        
        # n_water_hidden = n_salt * hydration_coeff
        df['n_water_hidden'] = n_salt * df['hydration_coeff']
        
        # Ratio_H2O_Metal
        df['Ratio_H2O_Metal'] = df['hydration_coeff']
        
    # 3. True Molarity Calculations
    if 'Vсин. (р-ля), мл' in df.columns:
        vol_ml = pd.to_numeric(df['Vсин. (р-ля), мл'], errors='coerce').fillna(0.0)
        vol_L = vol_ml / 1000.0
        # Avoid division by zero
        vol_L_safe = vol_L.replace(0, np.nan)
        
        if 'n_соли' in df.columns:
            n_salt = pd.to_numeric(df['n_соли'], errors='coerce').fillna(0.0)
            df['Molarity_Metal'] = n_salt / vol_L_safe
            
        if 'n_кислоты' in df.columns:
            n_acid = pd.to_numeric(df['n_кислоты'], errors='coerce').fillna(0.0)
            df['Molarity_Ligand'] = n_acid / vol_L_safe
            
        if 'n_water_hidden' in df.columns:
            n_water = pd.to_numeric(df['n_water_hidden'], errors='coerce').fillna(0.0)
            df['Molarity_H2O_Hidden'] = n_water / vol_L_safe
            
    # 4. Supersaturation & Reactor Loading
    if 'Molarity_Metal' in df.columns and 'Molarity_Ligand' in df.columns:
        df['Supersaturation_Index'] = df['Molarity_Metal'].fillna(0.0) * df['Molarity_Ligand'].fillna(0.0)
        
    if {'m (соли), г', 'm(кис-ты), г', 'Vсин. (р-ля), мл'}.issubset(df.columns):
        m_salt = pd.to_numeric(df['m (соли), г'], errors='coerce').fillna(0.0)
        m_acid = pd.to_numeric(df['m(кис-ты), г'], errors='coerce').fillna(0.0)
        vol_ml = pd.to_numeric(df['Vсин. (р-ля), мл'], errors='coerce').replace(0, np.nan)
        
        df['Reactor_Loading_g_mL'] = (m_salt + m_acid) / vol_ml

    # Fill NaNs with 0 for the new columns as requested
    new_cols = [
        'n_water_hidden', 'Ratio_H2O_Metal', 
        'Molarity_Metal', 'Molarity_Ligand', 'Molarity_H2O_Hidden',
        'Supersaturation_Index', 'Reactor_Loading_g_mL'
    ]
    
    for col in new_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    if not inplace:
        return df
    return None


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> Iterable[str]:
    return [col for col in columns if col not in df.columns]
