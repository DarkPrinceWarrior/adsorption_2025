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
A0_W0_COEFFICIENT: float = 28.86
A0_W0_REL_TOLERANCE: float = 0.05  # 5% deviation tolerance for a0 vs W0 relation
E_E0_RATIO_TARGET: float = 1.0 / 3.0
E_E0_RATIO_TOLERANCE: float = 0.05  # absolute tolerance in ratio units
WS_W0_TOLERANCE: float = 0.0

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

SOLVENT_BOILING_POINTS_C = {
    'ДМФА': 153.0,
    'DMF': 153.0,
    'N,N-dimethylformamide': 153.0,
    'Этанол': 78.0,
    'Ethanol': 78.0,
    'Вода': 100.0,
    'Water': 100.0,
    'Ацетонитрил': 82.0,
    'Acetonitrile': 82.0,
    'Метанол': 65.0,
    'Methanol': 65.0,
}

STOICHIOMETRY_TARGETS = {
    ('Cu', 'BTC'): {'ratio': 1.5, 'tolerance': 0.1},
    ('Zn', 'BDC'): {'ratio': 2.0, 'tolerance': 0.1},
    ('Zr', 'BDC'): {'ratio': 1.5, 'tolerance': 0.1},
    ('Al', 'BTC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('Fe', 'BTC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('Fe', 'BDC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('Al', 'BDC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('La', 'BTC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('Zn', 'BTB'): {'ratio': 1.33, 'tolerance': 0.15},
    ('Fe', 'NH2-BDC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('Zr', 'BTC'): {'ratio': 1.0, 'tolerance': 0.15},
    ('Al', 'BTB'): {'ratio': 1.5, 'tolerance': 0.2},
    ('Al', 'NH2-BDC'): {'ratio': 1.0, 'tolerance': 0.2},
    ('Fe', 'BTB'): {'ratio': 1.5, 'tolerance': 0.2},
}

DEFAULT_STOICHIOMETRY_BOUNDS: tuple[float, float] = (0.45, 2.3)
