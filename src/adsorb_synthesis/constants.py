"""Shared constants for the adsorbent inverse design pipeline."""
from __future__ import annotations

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.25
CV_FOLDS_CLASSIFICATION: int = 5
CV_FOLDS_REGRESSION: int = 5
OUTLIER_CONTAMINATION: float = 0.05
HUBER_DELTA_DEFAULT: float = 5.0  # Tuned to match salt-mass scale (std ~29 g)

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

# Advanced metal coordination chemistry descriptors (from enrich_descriptors.py)
# Full list for reference
METAL_COORD_FEATURES_ALL = [
    'oxidation_state (metal_coord)',
    'coordination_number (metal_coord)',
    'hsab_hardness_numeric (metal_coord)',
    'd_electrons (metal_coord)',
    'ionic_radius_pm (metal_coord)',
    'electronegativity_pauling (metal_coord)',
    'first_ionization_kj (metal_coord)',
    'electron_affinity_kj (metal_coord)',
]

# TOP-5 metal features (expert-selected, low multicollinearity)
# - Jahn_Teller_Active: marker for Cu2+ open metal sites (HKUST-1)
# - hsab_hardness: explains bond strength (Zr4+ hard -> UiO stable)
# - electron_affinity: correlates with E0, x0
# - ionic_radius: size matching with ligand
# - oxidation_state: charge density
METAL_COORD_FEATURES = [
    'oxidation_state (metal_coord)',
    'hsab_hardness_numeric (metal_coord)',
    'ionic_radius_pm (metal_coord)',
    'electron_affinity_kj (metal_coord)',
    # Note: Jahn_Teller_Active is in INTERACTION_FEATURES
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

# Advanced ligand 3D geometry descriptors (from RDKit)
# Full list for reference
LIGAND_3D_FEATURES_ALL = [
    'PMI1 (ligand_3d)',
    'PMI2 (ligand_3d)',
    'PMI3 (ligand_3d)',
    'NPR1 (ligand_3d)',
    'NPR2 (ligand_3d)',
    'Asphericity (ligand_3d)',
    'Eccentricity (ligand_3d)',
    'InertialShapeFactor (ligand_3d)',
    'RadiusOfGyration (ligand_3d)',
    'SpherocityIndex (ligand_3d)',
]

# TOP-3 ligand geometry features (expert-selected)
# - RadiusOfGyration: effective molecular size (relates to pore size)
# - PMI1: smallest principal moment (molecular "thickness")
# - Asphericity: shape deviation from sphere (rod vs disk vs sphere)
# Note: With only 4 ligands, more features = more noise
LIGAND_3D_FEATURES = [
    'RadiusOfGyration (ligand_3d)',
    'PMI1 (ligand_3d)',
    'Asphericity (ligand_3d)',
]

# Advanced ligand 2D chemical descriptors (from RDKit)
# Full list for reference - mostly noise with only 4 ligands
LIGAND_2D_FEATURES_ALL = [
    'NumRotatableBonds (ligand_2d)',
    'NumHeavyAtoms (ligand_2d)',
    'NumRings (ligand_2d)',
    'NumAromaticRings (ligand_2d)',
    'FractionCSP3 (ligand_2d)',
    'NumHeteroatoms (ligand_2d)',
    'NumAliphaticRings (ligand_2d)',
    'NumSaturatedRings (ligand_2d)',
    'RingCount (ligand_2d)',
    'LabuteASA (ligand_2d)',
    'BalabanJ (ligand_2d)',
    'BertzCT (ligand_2d)',
]

# Only LabuteASA (Approximate Surface Area) - physically meaningful
# Other 2D features are noisy with 4 ligands
LIGAND_2D_FEATURES = [
    'LabuteASA (ligand_2d)',  # Van der Waals surface area proxy
]

# Engineered metal-ligand interaction features
# Full list for reference
INTERACTION_FEATURES_ALL = [
    'Metal_Ligand_Size_Ratio',
    'Metal_O_Electronegativity_Diff',
    'Coordination_Saturation',
    'HSAB_Match_Score',
    'Jahn_Teller_Active',
    'Ligand_Flexibility_Index',
    'Ligand_Shape_Anisotropy',
]

# TOP-3 interaction features (expert-selected, physically meaningful)
# - Metal_Ligand_Size_Ratio: ionic radius / ligand size (pore geometry)
# - Metal_O_Electronegativity_Diff: bond polarity (adsorption energy)
# - Jahn_Teller_Active: Cu2+ marker (open metal sites in HKUST-1)
INTERACTION_FEATURES = [
    'Metal_Ligand_Size_Ratio',
    'Metal_O_Electronegativity_Diff',
    'Jahn_Teller_Active',
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


# --- Forward Model (Bayesian Optimization) Constants ---

FORWARD_MODEL_INPUTS = [
    # Categorical Inputs (The "Ingredients")
    'Металл',
    'Лиганд',
    'Растворитель',
    
    # Continuous Inputs (The "Recipe")
    'm (соли), г',
    'm(кис-ты), г',
    'Vсин. (р-ля), мл',
    'Т.син., °С',
    'Т суш., °С',
    'Tрег, ᵒС',  # T activation
]

FORWARD_MODEL_TARGETS = [
    'W0, см3/г',
    'E0, кДж/моль',
    'SБЭТ, м2/г',
    'х0, нм',
    'Sme, м2/г',
]

# Features derived from inputs that are safe to use in Forward Model
# (Physical descriptors of ingredients known BEFORE synthesis)
FORWARD_MODEL_AUGMENTED_FEATURES = (
    METAL_DESCRIPTOR_FEATURES + 
    METAL_COORD_FEATURES +
    LIGAND_DESCRIPTOR_FEATURES + 
    LIGAND_3D_FEATURES +
    LIGAND_2D_FEATURES +
    SOLVENT_DESCRIPTOR_FEATURES +
    INTERACTION_FEATURES
)

# Engineered features computed from synthesis parameters (known BEFORE synthesis)
# These capture stoichiometry and concentration - critical for W0 prediction
FORWARD_MODEL_ENGINEERED_FEATURES = [
    # Stoichiometry (from add_salt_mass_features)
    'R_molar',          # Molar ratio: n_salt / n_acid
    'R_mass',           # Mass ratio: m_salt / m_acid
    
    # Concentrations
    'C_metal',          # Metal concentration: m_salt / V_solvent
    'C_ligand',         # Ligand concentration: m_acid / V_solvent
    'log_C_metal',      # Log-transformed concentration
    'log_C_ligand',     # Log-transformed concentration
    
    # From CSV (pre-computed)
    'n_соли',           # Moles of salt
    'n_кислоты',        # Moles of acid
    'Vsyn_m',           # V_synthesis / m_salt (inverse concentration)
    
    # Temperature features
    'T_range',          # T_activation - T_synthesis
    'T_activation',     # T_reg - 100
    'T_dry_norm',       # Normalized drying temperature
    
    # Interaction
    'Metal_Ligand_Combo',  # Categorical: metal-ligand pair
]
