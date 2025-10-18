"""Physics-informed loss utilities with structured constraint configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import (
    ADSORPTION_ENERGY_RATIO_BOUNDS,
    E0_BOUNDS_KJ_MOL,
    R_GAS_J_MOL_K,
    THERMODYNAMIC_TOLERANCE,
)


@dataclass(frozen=True)
class BoundConstraint:
    """Bounded range constraint applied to a numeric column."""

    column: str
    lower: float
    upper: float
    normaliser: Optional[float] = None

    @property
    def span(self) -> float:
        width = self.normaliser if self.normaliser is not None else self.upper - self.lower
        return width if width > 0 else 1e-6


@dataclass(frozen=True)
class ThermodynamicConstraint:
    """Thermodynamic relationship between ΔG, temperature and K_eq."""

    temperature_column: str = "Т.син., °С"
    delta_g_column: str = "Delta_G"
    equilibrium_column: str = "K_equilibrium"
    tolerance: float = THERMODYNAMIC_TOLERANCE


@dataclass
class PhysicsConstraintEvaluator:
    """Evaluator that computes physics-based penalties and summary statistics."""

    energy_bounds: Sequence[BoundConstraint] = field(default_factory=tuple)
    thermodynamic: Optional[ThermodynamicConstraint] = None
    gas_constant: float = R_GAS_J_MOL_K

    def _ensure_length(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=np.float64)

    def energy_penalty(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty or not self.energy_bounds:
            return np.zeros(len(df), dtype=np.float64)
        penalties = np.zeros(len(df), dtype=np.float64)
        for constraint in self.energy_bounds:
            if constraint.column not in df.columns:
                continue
            values = pd.to_numeric(df[constraint.column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            below = np.maximum(0.0, constraint.lower - values)
            above = np.maximum(0.0, values - constraint.upper)
            penalties += np.nan_to_num((below + above) / constraint.span, nan=0.0)
        return penalties

    def thermodynamic_penalty(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty or self.thermodynamic is None:
            return np.zeros(len(df), dtype=np.float64)

        cfg = self.thermodynamic
        required = {cfg.temperature_column, cfg.delta_g_column, cfg.equilibrium_column}
        if not required.issubset(df.columns):
            return np.zeros(len(df), dtype=np.float64)

        temperature = pd.to_numeric(df[cfg.temperature_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        delta_g = pd.to_numeric(df[cfg.delta_g_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        equilibrium = pd.to_numeric(df[cfg.equilibrium_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        mask = np.isfinite(temperature) & np.isfinite(delta_g) & np.isfinite(equilibrium)
        penalties = np.zeros(len(df), dtype=np.float64)
        if not np.any(mask):
            return penalties

        T_kelvin = temperature[mask] + 273.15
        T_kelvin = np.clip(T_kelvin, 1e-3, None)
        delta_g_j = delta_g[mask] * 1000.0

        exponent = -delta_g_j / (self.gas_constant * T_kelvin)
        exponent = np.clip(exponent, -100.0, 100.0)
        k_theoretical = np.exp(exponent)
        denom = np.abs(equilibrium[mask]) + 1e-6
        relative_error = np.abs(k_theoretical - equilibrium[mask]) / denom
        violation = np.maximum(0.0, relative_error - cfg.tolerance)

        penalties[mask] = violation
        return penalties

    def penalties(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.zeros(0, dtype=np.float64)
        penalty = self.energy_penalty(df)
        penalty += self.thermodynamic_penalty(df)
        return np.nan_to_num(penalty, nan=0.0, neginf=0.0, posinf=0.0)

    def summary(self, df: pd.DataFrame, *, verbose: bool = False) -> Dict[str, float]:
        stats: Dict[str, float] = {}

        for constraint in self.energy_bounds:
            if constraint.column not in df.columns or df[constraint.column].empty:
                continue
            values = pd.to_numeric(df[constraint.column], errors="coerce")
            mask = np.isfinite(values)
            if not mask.any():
                continue
            within = ((values >= constraint.lower) & (values <= constraint.upper))[mask]
            key = f"{constraint.column}_within_bounds_rate"
            stats[key] = float(within.mean())
            stats[f"{constraint.column}_mean"] = float(values[mask].mean())
            stats[f"{constraint.column}_std"] = float(values[mask].std())
            if verbose:
                print(
                    f"{constraint.column}: {within.mean()*100:.1f}% within "
                    f"[{constraint.lower}, {constraint.upper}] "
                    f"(mean={values[mask].mean():.3f}, std={values[mask].std():.3f})"
                )

        if self.thermodynamic is not None:
            cfg = self.thermodynamic
            required = {cfg.temperature_column, cfg.delta_g_column, cfg.equilibrium_column}
            if required.issubset(df.columns):
                penalties = self.thermodynamic_penalty(df)
                rate = float(np.mean(penalties <= cfg.tolerance))
                stats["thermodynamic_consistency_rate"] = rate
                stats["thermodynamic_mean_penalty"] = float(np.mean(penalties))
                if verbose:
                    print(
                        f"Thermodynamic consistency: {rate*100:.1f}% within tolerance "
                        f"(mean penalty={np.mean(penalties):.4f})"
                    )

        return stats


DEFAULT_PHYSICS_EVALUATOR = PhysicsConstraintEvaluator(
    energy_bounds=(
        BoundConstraint(
            column="E0, кДж/моль",
            lower=E0_BOUNDS_KJ_MOL[0],
            upper=E0_BOUNDS_KJ_MOL[1],
        ),
        BoundConstraint(
            column="Adsorption_Energy_Ratio",
            lower=ADSORPTION_ENERGY_RATIO_BOUNDS[0],
            upper=ADSORPTION_ENERGY_RATIO_BOUNDS[1],
        ),
    ),
    thermodynamic=ThermodynamicConstraint(
        temperature_column="Т.син., °С",
        delta_g_column="Delta_G_equilibrium",
        equilibrium_column="K_equilibrium",
        tolerance=THERMODYNAMIC_TOLERANCE,
    ),
)


def thermodynamic_consistency_loss(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    evaluator: PhysicsConstraintEvaluator = DEFAULT_PHYSICS_EVALUATOR,
) -> float:
    """Mean thermodynamic violation for the provided samples."""
    if evaluator.thermodynamic is None:
        return 0.0
    df = pd.DataFrame(X, columns=feature_names)
    penalties = evaluator.thermodynamic_penalty(df)
    return float(np.mean(penalties))


def energy_bounds_loss(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    evaluator: PhysicsConstraintEvaluator = DEFAULT_PHYSICS_EVALUATOR,
) -> float:
    """Mean energy bound violation for the provided samples."""
    df = pd.DataFrame(X, columns=feature_names)
    penalties = evaluator.energy_penalty(df)
    return float(np.mean(penalties))


def combined_physics_loss(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    evaluator: PhysicsConstraintEvaluator = DEFAULT_PHYSICS_EVALUATOR,
    w_thermo: float = 0.05,
    w_energy: float = 0.02,
) -> float:
    """Weighted combination of thermodynamic and energy penalties."""
    df = pd.DataFrame(X, columns=feature_names)
    thermo = evaluator.thermodynamic_penalty(df)
    energy = evaluator.energy_penalty(df)
    total = w_thermo * thermo + w_energy * energy
    return float(np.mean(total))


def physics_violation_scores(
    df: pd.DataFrame,
    *,
    evaluator: PhysicsConstraintEvaluator = DEFAULT_PHYSICS_EVALUATOR,
) -> np.ndarray:
    """Per-sample aggregate penalty across all configured constraints."""
    return evaluator.penalties(df)


def validate_physics_constraints(
    df: pd.DataFrame,
    *,
    evaluator: PhysicsConstraintEvaluator = DEFAULT_PHYSICS_EVALUATOR,
    verbose: bool = True,
) -> Dict[str, float]:
    """Return summary statistics for configured physics constraints."""
    return evaluator.summary(df, verbose=verbose)


def project_thermodynamics(
    df: pd.DataFrame,
    *,
    evaluator: PhysicsConstraintEvaluator = DEFAULT_PHYSICS_EVALUATOR,
    overwrite: bool = False,
    residual_column: Optional[str] = "K_equilibrium_residual",
) -> pd.DataFrame:
    """
    Project thermodynamic variables onto the Gibbs equilibrium manifold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe expected to contain temperature, Delta_G and K_equilibrium.
    evaluator : PhysicsConstraintEvaluator
        Evaluator providing thermodynamic configuration.
    overwrite : bool
        When True, replace the original ``K_equilibrium`` column with the projected values.
        Otherwise a new ``K_equilibrium_projected`` column is added.
    residual_column : Optional[str]
        Name of the column that will store the difference between original and projected values.
        Set to None to skip recording residuals.

    Returns
    -------
    pd.DataFrame
        Dataframe with enforced thermodynamic consistency.
    """
    if evaluator.thermodynamic is None or df.empty:
        return df if overwrite else df.copy()

    cfg = evaluator.thermodynamic
    required = {cfg.temperature_column, cfg.delta_g_column, cfg.equilibrium_column}
    if not required.issubset(df.columns):
        return df if overwrite else df.copy()

    frame = df if overwrite else df.copy()
    temperature = pd.to_numeric(frame[cfg.temperature_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    delta_g = pd.to_numeric(frame[cfg.delta_g_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    k_eq = pd.to_numeric(frame[cfg.equilibrium_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    mask = np.isfinite(temperature) & np.isfinite(delta_g)
    projected = np.full_like(temperature, np.nan, dtype=np.float64)
    temperature_k = np.full_like(temperature, np.nan, dtype=np.float64)

    if np.any(mask):
        T_kelvin = np.clip(temperature[mask] + 273.15, 1e-3, None)
        temperature_k[mask] = T_kelvin
        delta_g_j = delta_g[mask] * 1000.0
        exponent = -delta_g_j / (evaluator.gas_constant * T_kelvin)
        exponent = np.clip(exponent, -100.0, 100.0)
        projected[mask] = np.exp(exponent)

    if residual_column is not None:
        residual = np.full_like(projected, np.nan, dtype=np.float64)
        valid = np.isfinite(projected) & np.isfinite(k_eq)
        residual[valid] = k_eq[valid] - projected[valid]
        frame[residual_column] = residual

    delta_projected = np.full_like(projected, np.nan, dtype=np.float64)
    valid_projected = np.isfinite(projected) & np.isfinite(temperature_k) & (projected > 0)
    if np.any(valid_projected):
        delta_projected[valid_projected] = -(
            evaluator.gas_constant * temperature_k[valid_projected] * np.log(projected[valid_projected])
        ) / 1000.0
    frame[cfg.delta_g_column] = delta_projected

    if overwrite:
        frame[cfg.equilibrium_column] = projected
    else:
        frame[f"{cfg.equilibrium_column}_projected"] = projected

    return frame
