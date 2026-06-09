"""Data validation helpers for adsorption descriptors and synthesis parameters."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .constants import (
    STOICHIOMETRY_GLOBAL_BOUNDS,
    STOICHIOMETRY_GROUP_FACTOR,
    STOICHIOMETRY_GROUP_MIN_N,
)

DEFAULT_VALIDATION_MODE = "warn"
VALIDATION_MODES = {"warn", "strict"}

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single validation warning or error."""

    row: object
    column: str
    severity: str
    message: str
    actual: Optional[float] = None
    expected: Optional[float] = None
    delta: Optional[float] = None


@dataclass
class ValidationReport:
    """Collection of validation issues materialised during dataset checks."""

    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def __bool__(self) -> bool:  # pragma: no cover - convenience helper
        return bool(self.issues)


def validate_SEH_data(
    df: pd.DataFrame,
    *,
    mode: str = DEFAULT_VALIDATION_MODE,
    a0_tolerance: float = 0.01,
    e_ratio_tolerance: float = 0.10,
) -> ValidationReport:
    """Check adsorption descriptors for hard-equality relationships."""

    _ensure_mode(mode)
    issues: list[ValidationIssue] = []

    w0_col = 'W0, см3/г'
    a0_col = 'а0, ммоль/г'
    e0_col = 'E0, кДж/моль'
    e_col = 'E, кДж/моль'
    ws_col = 'Ws, см3/г'

    if {w0_col, a0_col}.issubset(df.columns):
        w0 = pd.to_numeric(df[w0_col], errors='coerce')
        a0 = pd.to_numeric(df[a0_col], errors='coerce')
        expected_a0 = 28.86 * w0
        delta_a0 = a0 - expected_a0
        denom = np.where(np.abs(a0) > 0, np.abs(a0), 1.0)
        rel_err = np.abs(delta_a0) / denom
        mask = (rel_err > a0_tolerance) & np.isfinite(rel_err)
        for pos in np.where(mask)[0]:
            row = df.index[pos]
            issues.append(
                ValidationIssue(
                    row=row,
                    column=a0_col,
                    severity="error",
                    message=(
                        f"a0 deviates from 28.86·W0 by {rel_err[pos] * 100:.2f}% "
                        f"(tol={a0_tolerance * 100:.1f}%)"
                    ),
                    actual=_safe_float(a0.iloc[pos]),
                    expected=_safe_float(expected_a0.iloc[pos]),
                    delta=_safe_float(delta_a0.iloc[pos]),
                )
            )

    if {e0_col, e_col}.issubset(df.columns):
        e0 = pd.to_numeric(df[e0_col], errors='coerce')
        e_vals = pd.to_numeric(df[e_col], errors='coerce')
        valid = (e0.abs() > 0) & e0.notna() & e_vals.notna()
        ratios = pd.Series(np.nan, index=df.index, dtype=np.float64)
        ratios.loc[valid] = e_vals.loc[valid] / e0.loc[valid]
        expected_ratio = 1.0 / 3.0
        delta_ratio = ratios - expected_ratio
        mask = delta_ratio.abs() > e_ratio_tolerance
        violating_rows = mask[mask].index
        for row in violating_rows:
            ratio_value = ratios.loc[row]
            if not np.isfinite(ratio_value):
                continue
            issues.append(
                ValidationIssue(
                    row=row,
                    column=e_col,
                    severity="warning",
                    message=(
                        f"E/E0 ratio = {ratio_value:.3f}; expected ~{expected_ratio:.2f} "
                        f"(Δ={delta_ratio.loc[row]:+.3f})"
                    ),
                    actual=_safe_float(ratio_value),
                    expected=expected_ratio,
                    delta=_safe_float(delta_ratio.loc[row]),
                )
            )

    if {w0_col, ws_col}.issubset(df.columns):
        w0 = pd.to_numeric(df[w0_col], errors='coerce')
        ws = pd.to_numeric(df[ws_col], errors='coerce')
        mask = (ws < w0) & w0.notna() & ws.notna()
        for pos in np.where(mask)[0]:
            row = df.index[pos]
            delta = ws.iloc[pos] - w0.iloc[pos]
            issues.append(
                ValidationIssue(
                    row=row,
                    column=ws_col,
                    severity="error",
                    message=(
                        f"Ws ({ws.iloc[pos]:.3f}) is smaller than W0 ({w0.iloc[pos]:.3f})"
                    ),
                    actual=_safe_float(ws.iloc[pos]),
                    expected=_safe_float(w0.iloc[pos]),
                    delta=_safe_float(delta),
                )
            )

    report = ValidationReport(issues=issues)
    _log_issues(report, context="SEH data")
    if mode == "strict" and report.errors:
        raise ValueError(
            f"SEH validation failed: {len(report.errors)} error(s). "
            "Fix adsorption descriptors or run with --validation-mode warn."
        )
    return report


def validate_synthesis_data(
    df: pd.DataFrame,
    *,
    boiling_points: Mapping[str, float],
    mode: str = DEFAULT_VALIDATION_MODE,
) -> ValidationReport:
    """Check synthesis-related columns for physical plausibility."""

    _ensure_mode(mode)
    issues: list[ValidationIssue] = []

    metal_mass_col = 'm (соли), г'
    ligand_mass_col = 'm(кис-ты), г'
    solvent_vol_col = 'Vсин. (р-ля), мл'
    syn_temp_col = 'Т.син., °С'
    dry_temp_col = 'Т суш., °С'
    reg_temp_col = 'Tрег, ᵒС'
    solvent_col = 'Растворитель'

    for column in (metal_mass_col, ligand_mass_col, solvent_vol_col):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors='coerce')
        mask = ~(values > 0)
        for pos in np.where(mask)[0]:
            row = df.index[pos]
            issues.append(
                ValidationIssue(
                    row=row,
                    column=column,
                    severity="error",
                    message=f"{column} must be > 0",
                    actual=_safe_float(values.iloc[pos]),
                    expected=0.0,
                    delta=_safe_float(values.iloc[pos]),
                )
            )

    if {syn_temp_col, dry_temp_col, reg_temp_col}.issubset(df.columns):
        syn = pd.to_numeric(df[syn_temp_col], errors='coerce')
        dry = pd.to_numeric(df[dry_temp_col], errors='coerce')
        reg = pd.to_numeric(df[reg_temp_col], errors='coerce')
        valid = syn.notna() & dry.notna() & reg.notna()
        mask = valid & ((dry < syn) | (reg < dry))
        for pos in np.where(mask)[0]:
            row = df.index[pos]
            issues.append(
                ValidationIssue(
                    row=row,
                    column=syn_temp_col,
                    severity="warning",
                    message=(
                        f"Temperature order unexpected: T_syn={syn.iloc[pos]:.1f}, "
                        f"T_dry={dry.iloc[pos]:.1f}, T_reg={reg.iloc[pos]:.1f}"
                    ),
                    actual=_safe_float(syn.iloc[pos]),
                    expected=None,
                    delta=None,
                )
            )

    if {syn_temp_col, solvent_col}.issubset(df.columns):
        syn = pd.to_numeric(df[syn_temp_col], errors='coerce')
        solvent_series = df[solvent_col].astype(str).str.strip()
        lookup = {str(k).strip().lower(): v for k, v in boiling_points.items()}

        def _mixture_boiling(name: str) -> float:
            # Audit #9: for a solvent mixture (e.g. "ДМФА/Вода") use the LOWEST
            # component boiling point — the most volatile component caps the
            # achievable reflux temperature.
            comps = [c.strip().lower() for c in str(name).split('/') if c.strip()]
            vals = [lookup[c] for c in comps if c in lookup]
            return float(min(vals)) if vals else np.nan

        boiling = solvent_series.apply(_mixture_boiling)
        mask = syn.notna() & boiling.notna() & (syn >= boiling)
        for pos in np.where(mask)[0]:
            row = df.index[pos]
            issues.append(
                ValidationIssue(
                    row=row,
                    column=syn_temp_col,
                    # Audit #7/#9: exceeding the ATMOSPHERIC boiling point is normal
                    # for sealed solvothermal synthesis (autogenous pressure), so this
                    # is a WARNING (confirm sealed vessel), not a hard error.
                    severity="warning",
                    message=(
                        f"T_syn ({syn.iloc[pos]:.1f}°C) exceeds atmospheric boiling point "
                        f"of {df.at[row, solvent_col]} ({boiling.iloc[pos]:.1f}°C) — "
                        f"expected only if synthesis is sealed/solvothermal"
                    ),
                    actual=_safe_float(syn.iloc[pos]),
                    expected=_safe_float(boiling.iloc[pos]),
                    delta=_safe_float(syn.iloc[pos] - boiling.iloc[pos]),
                )
            )

    # Stoichiometry checks (Metal/Ligand feed molar ratio) — audit #2.
    # Policy: hard ERROR only for physically impossible ratios (global bounds);
    # data-driven WARNING for per-group statistical outliers (Tukey fences).
    # Framework formula ratios (STOICHIOMETRY_REFERENCE in constants) are
    # documentation only — the lab's feed ratios legitimately differ (excess
    # reagent / modulated synthesis), so we never gate against them.
    metal_col = 'Металл'
    ligand_col = 'Лиганд'
    ratio_col = 'R_molar'
    n_salt_col = 'n_соли'
    n_acid_col = 'n_кислоты'
    salt_mass_col = 'm (соли), г'
    acid_mass_col = 'm(кис-ты), г'
    molar_salt_col = 'Молярка_соли'
    molar_acid_col = 'Молярка_кислоты'

    def _row_ratio(idx) -> float:
        if ratio_col in df.columns:
            r = pd.to_numeric(df.at[idx, ratio_col], errors='coerce')
            if np.isfinite(r):
                return float(r)
        n_salt = pd.to_numeric(df.at[idx, n_salt_col], errors='coerce') if n_salt_col in df.columns else np.nan
        n_acid = pd.to_numeric(df.at[idx, n_acid_col], errors='coerce') if n_acid_col in df.columns else np.nan
        if np.isfinite(n_salt) and np.isfinite(n_acid) and n_acid != 0:
            return float(n_salt / n_acid)
        if {salt_mass_col, acid_mass_col, molar_salt_col, molar_acid_col}.issubset(df.columns):
            m_salt = pd.to_numeric(df.at[idx, salt_mass_col], errors='coerce')
            m_acid = pd.to_numeric(df.at[idx, acid_mass_col], errors='coerce')
            mw_salt = pd.to_numeric(df.at[idx, molar_salt_col], errors='coerce')
            mw_acid = pd.to_numeric(df.at[idx, molar_acid_col], errors='coerce')
            if (np.isfinite(m_salt) and np.isfinite(m_acid) and np.isfinite(mw_salt)
                    and np.isfinite(mw_acid) and mw_acid != 0 and mw_salt != 0):
                n_s = m_salt / mw_salt
                n_a = m_acid / mw_acid
                if n_a != 0:
                    return float(n_s / n_a)
        return float('nan')

    if metal_col in df.columns and ligand_col in df.columns:
        ratios = {idx: _row_ratio(idx) for idx in df.index}
        group_keys = df[metal_col].astype(str) + '|' + df[ligand_col].astype(str)
        lo_global, hi_global = STOICHIOMETRY_GLOBAL_BOUNDS

        # Per-group fences: wide multiplicative band around the group median, so
        # only gross per-group deviations are flagged (intentional DOE sweeps pass).
        fences: dict[str, tuple[float, float]] = {}
        finite_by_group: dict[str, list[float]] = {}
        for idx, value in ratios.items():
            if np.isfinite(value):
                finite_by_group.setdefault(group_keys.loc[idx], []).append(value)
        for gkey, values in finite_by_group.items():
            if len(values) >= STOICHIOMETRY_GROUP_MIN_N:
                med = float(np.median(values))
                if med > 0:
                    fences[gkey] = (
                        med / STOICHIOMETRY_GROUP_FACTOR,
                        med * STOICHIOMETRY_GROUP_FACTOR,
                    )

        for idx in df.index:
            ratio_val = ratios[idx]
            if not np.isfinite(ratio_val):
                continue
            gkey = group_keys.loc[idx]
            if ratio_val < lo_global or ratio_val > hi_global:
                issues.append(
                    ValidationIssue(
                        row=idx,
                        column=ratio_col,
                        severity="error",
                        message=(
                            f"R_molar={ratio_val:.3f} outside physical bounds "
                            f"[{lo_global}, {hi_global}] for ({df.at[idx, metal_col]}, "
                            f"{df.at[idx, ligand_col]})"
                        ),
                        actual=_safe_float(ratio_val),
                        expected=None,
                        delta=None,
                    )
                )
                continue
            fence = fences.get(gkey)
            if fence is not None and (ratio_val < fence[0] or ratio_val > fence[1]):
                issues.append(
                    ValidationIssue(
                        row=idx,
                        column=ratio_col,
                        severity="warning",
                        message=(
                            f"R_molar={ratio_val:.3f} is a feed-ratio outlier for {gkey} "
                            f"(group fence [{fence[0]:.3f}, {fence[1]:.3f}])"
                        ),
                        actual=_safe_float(ratio_val),
                        expected=None,
                        delta=None,
                    )
                )

    # Moles consistency (audit #11): if precomputed moles exist in the CSV, verify
    # they match the canonical mass / molar-mass derivation within tolerance. This
    # surfaces any divergence between the pre-computed and recomputed moles instead
    # of letting the two sources silently disagree downstream.
    for n_col, m_col, mw_col in (
        ('n_соли', 'm (соли), г', 'Молярка_соли'),
        ('n_кислоты', 'm(кис-ты), г', 'Молярка_кислоты'),
    ):
        if {n_col, m_col, mw_col}.issubset(df.columns):
            n_pre = pd.to_numeric(df[n_col], errors='coerce')
            m_vals = pd.to_numeric(df[m_col], errors='coerce')
            mw_vals = pd.to_numeric(df[mw_col], errors='coerce').replace(0, np.nan)
            n_derived = m_vals / mw_vals
            denom = n_derived.abs().replace(0, np.nan)
            rel = (n_pre - n_derived).abs() / denom
            mask = (rel > 0.02) & np.isfinite(rel)
            for pos in np.where(mask.to_numpy())[0]:
                row = df.index[pos]
                issues.append(
                    ValidationIssue(
                        row=row,
                        column=n_col,
                        severity="warning",
                        message=(
                            f"{n_col}={n_pre.iloc[pos]:.4g} diverges from mass/MW "
                            f"derivation {n_derived.iloc[pos]:.4g} by {rel.iloc[pos] * 100:.1f}%"
                        ),
                        actual=_safe_float(n_pre.iloc[pos]),
                        expected=_safe_float(n_derived.iloc[pos]),
                        delta=_safe_float(n_pre.iloc[pos] - n_derived.iloc[pos]),
                    )
                )

    report = ValidationReport(issues=issues)
    _log_issues(report, context="synthesis data")
    if mode == "strict" and report.errors:
        raise ValueError(
            f"Synthesis validation failed: {len(report.errors)} error(s). "
            "Fix dataset rows or run with --validation-mode warn."
        )
    return report


def _log_issues(report: ValidationReport, *, context: str) -> None:
    if not report.issues:
        return
    for issue in report.issues:
        log_fn = logger.error if issue.severity == "error" else logger.warning
        log_fn(
            "%s validation (%s) row=%s: %s [actual=%s, expected=%s, delta=%s]",
            context,
            issue.column,
            issue.row,
            issue.message,
            issue.actual,
            issue.expected,
            issue.delta,
        )


def _ensure_mode(mode: str) -> None:
    if mode not in VALIDATION_MODES:
        raise ValueError(f"Unknown validation mode '{mode}'. Available: {sorted(VALIDATION_MODES)}")


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:  # pragma: no cover - defensive fallback
        return None
