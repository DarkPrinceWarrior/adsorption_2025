"""Shared constants for the adsorbent inverse design pipeline."""
from __future__ import annotations

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.25
CV_FOLDS_CLASSIFICATION: int = 5
CV_FOLDS_REGRESSION: int = 5
OUTLIER_CONTAMINATION: float = 0.05

# Physics-informed loss constants
R_GAS_J_MOL_K: float = 8.314  # Gas constant in J/(mol·K)
E0_BOUNDS_KJ_MOL: tuple[float, float] = (10.0, 50.0)  # Bounds for E0 in kJ/mol (physisorption range)
ADSORPTION_ENERGY_RATIO_BOUNDS: tuple[float, float] = (0.2, 1.0)  # Bounds for E/E0 ratio
THERMODYNAMIC_TOLERANCE: float = 0.15  # 15% tolerance for K_eq vs theoretical

ADSORPTION_FEATURES = [
    'W0, см3/г',
    'E0, кДж/моль',
    'х0, нм',
    'а0, ммоль/г',
    'E, кДж/моль',
    'SБЭТ, м2/г',
    'Ws, см3/г',
    'Sme, м2/г',
    'Wme, см3/г',
    'Adsorption_Potential',
    'Capacity_Density',
    'K_equilibrium',
    'Delta_G',
    'SurfaceArea_MicroVol_Ratio',
    'Adsorption_Energy_Ratio',
    'S_BET_E',
    'x0_W0',
    'B_micropore',
]

METAL_DESCRIPTOR_FEATURES = [
    'Total molecular weight (metal)',
    'Average ionic radius (metal)',
    'Average electronegativity (metal)',
    'Молярка_соли',
]

LIGAND_DESCRIPTOR_FEATURES = [
    'carboxyl_groups (ligand)',
    'aromatic_rings (ligand)',
    'carbon_atoms (ligand)',
    'oxygen_atoms (ligand)',
    'nitrogen_atoms (ligand)',
    'molecular_weight (ligand)',
    'amino_groups (ligand)',
    'logP (ligand)',
    'TPSA (ligand)',
    'h_bond_acceptors (ligand)',
    'h_bond_donors (ligand)',
    'Молярка_кислоты',
]

SOLVENT_DESCRIPTOR_FEATURES = [
    'Solvent_MolWt',
    'Solvent_LogP',
    'Solvent_NumHDonors',
    'Solvent_NumHAcceptors',
]

PROCESS_CONTEXT_FEATURES = [
    'Mix_solv_ratio',
]

TEMPERATURE_CATEGORIES = {
    'Tsyn_Category': {
        'column': 'Т.син., °С',
        'bins': [0, 115, 135, 300],
        'labels': ['Низкая (<115°C)', 'Средняя (115-135°C)', 'Высокая (>135°C)'],
    },
    'Tdry_Category': {
        'column': 'Т суш., °С',
        'bins': [0, 115, 135, 300],
        'labels': ['Низкая (<115°C)', 'Средняя (115-135°C)', 'Высокая (>135°C)'],
    },
    'Treg_Category': {
        'column': 'Tрег, ᵒС',
        'bins': [0, 155, 265, 500],
        'labels': ['Низкая (<155°C)', 'Средняя (155-265°C)', 'Высокая (>265°C)'],
    },
}
