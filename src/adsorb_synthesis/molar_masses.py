"""Reference data and helpers for molar masses used in the pipeline."""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

METAL_SALTS_MOLAR_MASSES = {
    # Copper
    "Cu(NO3)2·3H2O": 241.60,
    "Cu(NO3)2·6H2O": 295.65,
    "Cu(NO3)2": 187.56,
    # Zinc
    "Zn(NO3)2·6H2O": 297.49,
    "Zn(NO3)2": 189.40,
    # Aluminium
    "Al(NO3)3·9H2O": 375.13,
    "Al(NO3)3": 213.00,
    # Iron
    "Fe(NO3)3·9H2O": 404.00,
    "Fe(NO3)3·6H2O": 349.95,
    "Fe(NO3)3": 241.86,
    # Zirconium / zirconyl
    "ZrO(NO3)2·H2O": 249.26,
    "Zr(NO3)4·5H2O": 429.33,
    # Cerium
    "Ce(NO3)3·6H2O": 434.22,
    "Ce(NO3)3": 326.13,
    # Lanthanum
    "La(NO3)3·6H2O": 433.01,
    "La(NO3)3": 324.92,
    # Yttrium
    "Y(NO3)3·6H2O": 383.01,
    "Y(NO3)3": 274.92,
}

TYPICAL_SALTS = {
    "Cu": "Cu(NO3)2·3H2O",
    "Zn": "Zn(NO3)2·6H2O",
    "Al": "Al(NO3)3·9H2O",
    "Fe": "Fe(NO3)3·9H2O",
    "Zr": "ZrO(NO3)2·H2O",
    "Ce": "Ce(NO3)3·6H2O",
    "La": "La(NO3)3·6H2O",
    "Y": "Y(NO3)3·6H2O",
}

ANHYDROUS_FALLBACKS = {
    "Cu": "Cu(NO3)2",
    "Zn": "Zn(NO3)2",
    "Al": "Al(NO3)3",
    "Fe": "Fe(NO3)3",
    "Ce": "Ce(NO3)3",
    "La": "La(NO3)3",
    "Y": "Y(NO3)3",
}

METAL_ALIASES = {
    "Cu2+": "Cu",
    "Cu(II)": "Cu",
    "Zn2+": "Zn",
    "Zn(II)": "Zn",
    "Fe3+": "Fe",
    "Fe(III)": "Fe",
    "Al3+": "Al",
    "Al(III)": "Al",
    "La3+": "La",
    "La(III)": "La",
    "Ce3+": "Ce",
    "Ce(III)": "Ce",
    "Y3+": "Y",
    "Y(III)": "Y",
}

LIGANDS_MOLAR_MASSES = {
    "H3BTC": 210.14,
    "BTC": 210.14,
    "trimesic_acid": 210.14,
    "H2BDC": 166.13,
    "BDC": 166.13,
    "terephthalic_acid": 166.13,
    "H3BTB": 446.46,
    "BTB": 446.46,
}

LIGAND_ALIASES = {
    "Trimesic Acid": "trimesic_acid",
    "Terephthalic Acid": "terephthalic_acid",
}


def _normalise_key(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    stripped = str(name).strip()
    return stripped if stripped else None


def _lookup_metal_symbol(metal: Optional[str]) -> Optional[str]:
    key = _normalise_key(metal)
    if key is None:
        return None
    if key in TYPICAL_SALTS:
        return key
    if key in METAL_ALIASES:
        return METAL_ALIASES[key]
    token = key.split()[0]
    if token in TYPICAL_SALTS:
        return token
    if token in METAL_ALIASES:
        return METAL_ALIASES[token]
    letters = "".join(ch for ch in token if ch.isalpha())
    if not letters:
        return None
    if len(letters) == 1:
        symbol = letters.upper()
    else:
        symbol = letters[0].upper() + letters[1].lower()
    if symbol in TYPICAL_SALTS or symbol in ANHYDROUS_FALLBACKS:
        return symbol
    return None


def _lookup_ligand_key(ligand: Optional[str]) -> Optional[str]:
    key = _normalise_key(ligand)
    if key is None:
        return None
    key_lower = key.lower()
    if key in LIGANDS_MOLAR_MASSES:
        return key
    if key_lower in LIGANDS_MOLAR_MASSES:
        return key_lower
    if key in LIGAND_ALIASES:
        return LIGAND_ALIASES[key]
    if key_lower in LIGAND_ALIASES:
        return LIGAND_ALIASES[key_lower]
    return None


def metal_salt_molar_mass(
    metal: Optional[str],
    *,
    warn_on_fallback: bool = True,
) -> tuple[Optional[float], Optional[str]]:
    """Return molar mass (g/mol) and salt name for a metal entry."""
    symbol = _lookup_metal_symbol(metal)
    if symbol is None:
        return None, None

    salt = TYPICAL_SALTS.get(symbol)
    if salt in METAL_SALTS_MOLAR_MASSES:
        return METAL_SALTS_MOLAR_MASSES[salt], salt

    fallback_salt = ANHYDROUS_FALLBACKS.get(symbol)
    if fallback_salt and fallback_salt in METAL_SALTS_MOLAR_MASSES:
        if warn_on_fallback:
            warnings.warn(
                f"Using fallback salt '{fallback_salt}' for metal '{metal}'.",
                RuntimeWarning,
                stacklevel=3,
            )
        return METAL_SALTS_MOLAR_MASSES[fallback_salt], fallback_salt

    return None, None


def ligand_molar_mass(ligand: Optional[str]) -> Optional[float]:
    key = _lookup_ligand_key(ligand)
    if key is None:
        return None
    return LIGANDS_MOLAR_MASSES.get(key)


def add_molar_mass_columns(df: pd.DataFrame, *, warn_on_fallback: bool = True) -> None:
    """Ensure molar mass columns for salt and ligand are present."""
    if "Металл" in df.columns:
        masses = []
        for metal in df["Металл"]:
            mass, _ = metal_salt_molar_mass(metal, warn_on_fallback=warn_on_fallback)
            masses.append(mass)
        mass_series = pd.Series(masses, index=df.index, dtype="float64")
        if "Молярка_соли" in df.columns:
            mask = df["Молярка_соли"].isna() & mass_series.notna()
            df.loc[mask, "Молярка_соли"] = mass_series[mask]
        else:
            df["Молярка_соли"] = mass_series

    if "Лиганд" in df.columns:
        lig_masses = df["Лиганд"].map(ligand_molar_mass)
        if "Молярка_кислоты" in df.columns:
            mask = df["Молярка_кислоты"].isna() & lig_masses.notna()
            df.loc[mask, "Молярка_кислоты"] = lig_masses[mask]
        else:
            df["Молярка_кислоты"] = lig_masses
