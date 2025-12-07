"""Physics-informed penalties and validation for adsorption descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .constants import (
    E_E0_RATIO_TARGET,
    E_E0_RATIO_TOLERANCE,
    E0_BOUNDS_KJ_MOL,
    WS_W0_TOLERANCE,
)


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def energy_ratio_penalty(E: np.ndarray, E0: np.ndarray) -> np.ndarray:
    """Penalty for |E/E0 - target| beyond tolerance."""
    ratio = np.full_like(E, np.nan, dtype=float)
    valid = np.isfinite(E) & np.isfinite(E0) & (E0 != 0)
    ratio[valid] = E[valid] / E0[valid]
    target = E_E0_RATIO_TARGET
    tol = E_E0_RATIO_TOLERANCE
    penalty = np.zeros_like(ratio)
    mask = np.isfinite(ratio)
    excess = np.abs(ratio[mask] - target) - tol
    penalty[mask] = np.where(excess > 0, excess, 0.0)
    return penalty


def e0_bounds_penalty(E0: np.ndarray) -> np.ndarray:
    """Penalty for E0 outside physical bounds."""
    lo, hi = E0_BOUNDS_KJ_MOL
    penalty = np.zeros_like(E0, dtype=float)
    with np.errstate(invalid="ignore"):
        penalty = np.where(E0 < lo, lo - E0, penalty)
        penalty = np.where(E0 > hi, E0 - hi, penalty)
    penalty[~np.isfinite(penalty)] = 0.0
    return penalty


def ws_w0_penalty(Ws: np.ndarray, W0: np.ndarray) -> np.ndarray:
    """Penalty when Ws < W0 (violates pore volume hierarchy)."""
    penalty = np.zeros_like(Ws, dtype=float)
    valid = np.isfinite(Ws) & np.isfinite(W0)
    mask = valid & (Ws < W0 - WS_W0_TOLERANCE)
    penalty[mask] = (W0[mask] - Ws[mask])
    return penalty


@dataclass
class PhysicsPenaltyReport:
    energy_ratio_mean: float
    e0_bounds_mean: float
    ws_w0_mean: float
    n_rows: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "energy_ratio_mean": self.energy_ratio_mean,
            "e0_bounds_mean": self.e0_bounds_mean,
            "ws_w0_mean": self.ws_w0_mean,
            "n_rows": self.n_rows,
        }


def compute_physics_penalty(df: pd.DataFrame) -> pd.Series:
    """Aggregate physics penalties per row."""
    e = _safe_numeric(df["E, кДж/моль"]) if "E, кДж/моль" in df.columns else np.array([])
    e0 = _safe_numeric(df["E0, кДж/моль"]) if "E0, кДж/моль" in df.columns else np.array([])
    ws = _safe_numeric(df["Ws, см3/г"]) if "Ws, см3/г" in df.columns else np.array([])
    w0 = _safe_numeric(df["W0, см3/г"]) if "W0, см3/г" in df.columns else np.array([])

    n = len(df)
    penalty = np.zeros(n, dtype=float)

    if len(e) == n and len(e0) == n:
        penalty += energy_ratio_penalty(e, e0)
        penalty += e0_bounds_penalty(e0)
    if len(ws) == n and len(w0) == n:
        penalty += ws_w0_penalty(ws, w0)

    return pd.Series(penalty, index=df.index, name="physics_penalty")


def validate_physics_constraints(df: pd.DataFrame, verbose: bool = False) -> PhysicsPenaltyReport:
    """Compute mean penalties for monitoring."""
    penalties = compute_physics_penalty(df)

    # Break down components
    e = _safe_numeric(df["E, кДж/моль"]) if "E, кДж/моль" in df.columns else np.array([])
    e0 = _safe_numeric(df["E0, кДж/моль"]) if "E0, кДж/моль" in df.columns else np.array([])
    ws = _safe_numeric(df["Ws, см3/г"]) if "Ws, см3/г" in df.columns else np.array([])
    w0 = _safe_numeric(df["W0, см3/г"]) if "W0, см3/г" in df.columns else np.array([])

    er_pen = energy_ratio_penalty(e, e0) if len(e) else np.array([])
    e0_pen = e0_bounds_penalty(e0) if len(e0) else np.array([])
    ws_pen = ws_w0_penalty(ws, w0) if len(ws) else np.array([])

    report = PhysicsPenaltyReport(
        energy_ratio_mean=float(np.nanmean(er_pen)) if er_pen.size else 0.0,
        e0_bounds_mean=float(np.nanmean(e0_pen)) if e0_pen.size else 0.0,
        ws_w0_mean=float(np.nanmean(ws_pen)) if ws_pen.size else 0.0,
        n_rows=len(df),
    )

    if verbose:
        print("Physics penalty report:")
        print(f"  Energy ratio (|E/E0 - {E_E0_RATIO_TARGET:.3f}| - tol)+ : {report.energy_ratio_mean:.4f}")
        print(f"  E0 bounds [{E0_BOUNDS_KJ_MOL[0]}, {E0_BOUNDS_KJ_MOL[1]}] mean penalty: {report.e0_bounds_mean:.4f}")
        print(f"  Ws >= W0 penalty mean: {report.ws_w0_mean:.4f}")
        print(f"  Rows: {report.n_rows}")

    return report
