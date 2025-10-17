"""Physics-informed loss functions for adsorption synthesis models."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .constants import (
    ADSORPTION_ENERGY_RATIO_BOUNDS,
    E0_BOUNDS_KJ_MOL,
    R_GAS_J_MOL_K,
    THERMODYNAMIC_TOLERANCE,
)


def thermodynamic_consistency_loss(
    X: np.ndarray,
    feature_names: list[str],
    *,
    temperature_column: str = "Т.син., °С",
    delta_g_column: str = "Delta_G",
    k_eq_column: str = "K_equilibrium",
    tolerance: float = THERMODYNAMIC_TOLERANCE,
) -> float:
    """
    Calculate physics-informed loss based on Gibbs-K_equilibrium relationship.
    
    Thermodynamic constraint:
        K_eq = exp(-ΔG / (R*T))
    
    where:
        R = 8.314 J/(mol·K)
        T = synthesis temperature in Kelvin
        ΔG = Gibbs free energy in kJ/mol (converted to J/mol)
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    feature_names : list[str]
        Names of features corresponding to X columns
    temperature_column : str
        Name of temperature column in Celsius
    delta_g_column : str
        Name of Delta_G column in kJ/mol
    k_eq_column : str
        Name of equilibrium constant column
    tolerance : float
        Relative tolerance for violation (0.1 = 10%)
    
    Returns
    -------
    float
        Mean relative error with tolerance applied
    """
    # Convert feature names to indices
    try:
        t_idx = feature_names.index(temperature_column)
        dg_idx = feature_names.index(delta_g_column)
        k_idx = feature_names.index(k_eq_column)
    except ValueError as exc:
        # If required columns are missing, return zero loss
        return 0.0
    
    # Extract columns
    T_celsius = X[:, t_idx]
    Delta_G_kj = X[:, dg_idx]
    K_measured = X[:, k_idx]
    
    # Convert temperature to Kelvin
    T_kelvin = T_celsius + 273.15
    
    # Convert Delta_G from kJ/mol to J/mol
    Delta_G_j = Delta_G_kj * 1000.0
    
    # Calculate theoretical K from Gibbs equation
    # K_theoretical = exp(-ΔG / (R*T))
    exponent = -Delta_G_j / (R_GAS_J_MOL_K * T_kelvin)
    # Clip exponent to prevent overflow
    exponent = np.clip(exponent, -100, 100)
    K_theoretical = np.exp(exponent)
    
    # Calculate relative error with tolerance
    # Only penalize if error exceeds tolerance
    relative_error = np.abs(K_theoretical - K_measured) / (np.abs(K_measured) + 1e-6)
    violation = np.maximum(0.0, relative_error - tolerance)
    
    return float(np.mean(violation))


def energy_bounds_loss(
    X: np.ndarray,
    feature_names: list[str],
    *,
    e0_column: str = "E0, кДж/моль",
    energy_ratio_column: str = "Adsorption_Energy_Ratio",
    e0_bounds: tuple[float, float] = E0_BOUNDS_KJ_MOL,
    ratio_bounds: tuple[float, float] = ADSORPTION_ENERGY_RATIO_BOUNDS,
) -> float:
    """
    Calculate loss for violations of physical energy bounds.
    
    Physical constraints:
        1. E0 (characteristic energy) should be in [15, 40] kJ/mol for CO2 physisorption
        2. E/E0 ratio should be in [0.3, 0.8] (micropore vs mesopore energies)
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    feature_names : list[str]
        Names of features corresponding to X columns
    e0_column : str
        Name of E0 column
    energy_ratio_column : str
        Name of Adsorption_Energy_Ratio column
    e0_bounds : tuple[float, float]
        (min, max) bounds for E0 in kJ/mol
    ratio_bounds : tuple[float, float]
        (min, max) bounds for E/E0 ratio
    
    Returns
    -------
    float
        Mean penalty for bound violations
    """
    penalties = []
    
    # E0 bounds constraint
    if e0_column in feature_names:
        e0_idx = feature_names.index(e0_column)
        E0 = X[:, e0_idx]
        
        # Penalty for values below minimum
        below_min = np.maximum(0.0, e0_bounds[0] - E0)
        # Penalty for values above maximum
        above_max = np.maximum(0.0, E0 - e0_bounds[1])
        
        # Normalize by bounds range for consistent scale
        bound_range = e0_bounds[1] - e0_bounds[0]
        e0_penalty = (below_min + above_max) / bound_range
        penalties.append(e0_penalty)
    
    # Energy ratio bounds constraint
    if energy_ratio_column in feature_names:
        ratio_idx = feature_names.index(energy_ratio_column)
        ratio = X[:, ratio_idx]
        
        # Penalty for values below minimum
        below_min = np.maximum(0.0, ratio_bounds[0] - ratio)
        # Penalty for values above maximum
        above_max = np.maximum(0.0, ratio - ratio_bounds[1])
        
        # Normalize by bounds range
        bound_range = ratio_bounds[1] - ratio_bounds[0]
        ratio_penalty = (below_min + above_max) / bound_range
        penalties.append(ratio_penalty)
    
    if not penalties:
        return 0.0
    
    # Average across all constraints
    total_penalty = np.sum(penalties, axis=0)
    return float(np.mean(total_penalty))


def combined_physics_loss(
    X: np.ndarray,
    feature_names: list[str],
    *,
    w_thermo: float = 0.05,
    w_energy: float = 0.02,
    **kwargs,
) -> float:
    """
    Combine multiple physics-informed losses with weights.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list[str]
        Feature names
    w_thermo : float
        Weight for thermodynamic consistency loss
    w_energy : float
        Weight for energy bounds loss
    **kwargs : dict
        Additional parameters passed to individual loss functions
    
    Returns
    -------
    float
        Weighted sum of physics losses
    """
    loss_thermo = thermodynamic_consistency_loss(X, feature_names, **kwargs)
    loss_energy = energy_bounds_loss(X, feature_names, **kwargs)
    
    total_loss = w_thermo * loss_thermo + w_energy * loss_energy
    return total_loss


def validate_physics_constraints(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Validate physics constraints on a dataset and return violation statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with physics features
    verbose : bool
        Whether to print statistics
    
    Returns
    -------
    Dict[str, float]
        Statistics about constraint violations
    """
    stats = {}
    
    # Thermodynamic consistency check
    if all(col in df.columns for col in ["Т.син., °С", "Delta_G", "K_equilibrium"]):
        T_kelvin = df["Т.син., °С"] + 273.15
        Delta_G_j = df["Delta_G"] * 1000.0
        K_theoretical = np.exp(-Delta_G_j / (R_GAS_J_MOL_K * T_kelvin))
        K_measured = df["K_equilibrium"]
        
        relative_error = np.abs(K_theoretical - K_measured) / (np.abs(K_measured) + 1e-6)
        consistent = (relative_error < THERMODYNAMIC_TOLERANCE).sum() / len(df)
        stats["thermodynamic_consistency_rate"] = float(consistent)
        stats["thermodynamic_mean_error"] = float(np.mean(relative_error))
        
        if verbose:
            print(f"Thermodynamic consistency: {consistent*100:.1f}% within {THERMODYNAMIC_TOLERANCE*100:.0f}% tolerance")
            print(f"  Mean relative error: {np.mean(relative_error):.4f}")
    
    # E0 bounds check
    if "E0, кДж/моль" in df.columns:
        E0 = df["E0, кДж/моль"]
        within_bounds = ((E0 >= E0_BOUNDS_KJ_MOL[0]) & (E0 <= E0_BOUNDS_KJ_MOL[1])).sum() / len(df)
        stats["e0_within_bounds_rate"] = float(within_bounds)
        stats["e0_mean"] = float(E0.mean())
        stats["e0_std"] = float(E0.std())
        
        if verbose:
            print(f"E0 within [{E0_BOUNDS_KJ_MOL[0]}, {E0_BOUNDS_KJ_MOL[1]}] kJ/mol: {within_bounds*100:.1f}%")
            print(f"  E0 range: [{E0.min():.2f}, {E0.max():.2f}], mean: {E0.mean():.2f} kJ/mol")
    
    # Energy ratio bounds check
    if "Adsorption_Energy_Ratio" in df.columns:
        ratio = df["Adsorption_Energy_Ratio"]
        within_bounds = ((ratio >= ADSORPTION_ENERGY_RATIO_BOUNDS[0]) & 
                        (ratio <= ADSORPTION_ENERGY_RATIO_BOUNDS[1])).sum() / len(df)
        stats["energy_ratio_within_bounds_rate"] = float(within_bounds)
        stats["energy_ratio_mean"] = float(ratio.mean())
        
        if verbose:
            print(f"Energy ratio within [{ADSORPTION_ENERGY_RATIO_BOUNDS[0]}, {ADSORPTION_ENERGY_RATIO_BOUNDS[1]}]: {within_bounds*100:.1f}%")
            print(f"  Ratio range: [{ratio.min():.2f}, {ratio.max():.2f}], mean: {ratio.mean():.2f}")
    
    return stats
