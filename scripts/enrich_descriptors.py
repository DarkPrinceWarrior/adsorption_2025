#!/usr/bin/env python3
"""
Enrich MOF synthesis dataset with advanced physicochemical descriptors.

This script adds:
1. Ligand 3D geometry descriptors (PMI, NPR, Asphericity, Radius of Gyration)
2. Ligand 2D descriptors (rotatable bonds, heavy atoms, ring count)
3. Metal coordination chemistry descriptors (oxidation state, coordination number, HSAB hardness)
4. Engineered interaction features (metal-ligand size ratio, electronegativity difference)

Usage:
    python scripts/enrich_descriptors.py \
        --input data/SEC_SYN_with_features_DMFA_only.csv \
        --output data/SEC_SYN_with_features_DMFA_only_enriched.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors

warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# LIGAND SMILES DICTIONARY (verified by user)
# =============================================================================
LIGAND_SMILES = {
    'BTC': 'OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1',
    'BDC': 'OC(=O)c1ccc(cc1)C(=O)O',
    'NH2-BDC': 'Nc1cc(C(=O)O)ccc1C(=O)O',
    'BTB': 'OC(=O)c1ccc(cc1)c2cc(cc(c2)c3ccc(cc3)C(=O)O)c4ccc(cc4)C(=O)O',
}

# =============================================================================
# METAL COORDINATION CHEMISTRY DICTIONARY
# Literature-based values for common MOF metals
# =============================================================================
METAL_PROPERTIES = {
    'Cu': {
        'oxidation_state': 2,
        'coordination_number': 4,  # Square planar in paddlewheel
        'hsab_hardness': 'borderline',  # Borderline acid
        'hsab_hardness_numeric': 2,  # 1=soft, 2=borderline, 3=hard
        'd_electrons': 9,
        'ionic_radius_pm': 73,  # pm, for Cu2+
        'electronegativity_pauling': 1.90,
        'first_ionization_kj': 745.5,
        'electron_affinity_kj': 118.4,
    },
    'Zn': {
        'oxidation_state': 2,
        'coordination_number': 4,  # Tetrahedral in ZIF
        'hsab_hardness': 'borderline',
        'hsab_hardness_numeric': 2,
        'd_electrons': 10,
        'ionic_radius_pm': 74,  # pm, for Zn2+
        'electronegativity_pauling': 1.65,
        'first_ionization_kj': 906.4,
        'electron_affinity_kj': 0,
    },
    'Al': {
        'oxidation_state': 3,
        'coordination_number': 6,  # Octahedral in MIL
        'hsab_hardness': 'hard',
        'hsab_hardness_numeric': 3,
        'd_electrons': 0,
        'ionic_radius_pm': 53.5,  # pm, for Al3+
        'electronegativity_pauling': 1.61,
        'first_ionization_kj': 577.5,
        'electron_affinity_kj': 42.5,
    },
    'Fe': {
        'oxidation_state': 3,
        'coordination_number': 6,  # Octahedral in MIL
        'hsab_hardness': 'hard',  # Fe3+ is hard
        'hsab_hardness_numeric': 3,
        'd_electrons': 5,  # Fe3+ high spin
        'ionic_radius_pm': 64.5,  # pm, for Fe3+
        'electronegativity_pauling': 1.83,
        'first_ionization_kj': 762.5,
        'electron_affinity_kj': 15.7,
    },
    'Zr': {
        'oxidation_state': 4,
        'coordination_number': 8,  # Square antiprismatic in UiO
        'hsab_hardness': 'hard',
        'hsab_hardness_numeric': 3,
        'd_electrons': 0,  # Zr4+ has no d electrons
        'ionic_radius_pm': 84,  # pm, for Zr4+
        'electronegativity_pauling': 1.33,
        'first_ionization_kj': 640.1,
        'electron_affinity_kj': 41.1,
    },
    'La': {
        'oxidation_state': 3,
        'coordination_number': 9,  # High coordination typical for lanthanides
        'hsab_hardness': 'hard',
        'hsab_hardness_numeric': 3,
        'd_electrons': 0,  # La3+ has no d/f electrons
        'ionic_radius_pm': 103.2,  # pm, for La3+
        'electronegativity_pauling': 1.10,
        'first_ionization_kj': 538.1,
        'electron_affinity_kj': 48,
    },
    'Y': {
        'oxidation_state': 3,
        'coordination_number': 8,  # Similar to lanthanides
        'hsab_hardness': 'hard',
        'hsab_hardness_numeric': 3,
        'd_electrons': 0,  # Y3+ has no d electrons
        'ionic_radius_pm': 90,  # pm, for Y3+
        'electronegativity_pauling': 1.22,
        'first_ionization_kj': 600,
        'electron_affinity_kj': 29.6,
    },
}


def compute_ligand_3d_descriptors(smiles: str, n_conformers: int = 10) -> dict:
    """
    Compute 3D geometry descriptors for a ligand using RDKit.
    
    Args:
        smiles: SMILES string of the ligand
        n_conformers: Number of conformers to generate for averaging
        
    Returns:
        Dictionary of 3D descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in [
            'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
            'Asphericity', 'Eccentricity', 'InertialShapeFactor',
            'RadiusOfGyration', 'SpherocityIndex'
        ]}
    
    # Add hydrogens for proper 3D geometry
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)
    
    if mol.GetNumConformers() == 0:
        # Fallback: try with different parameters
        AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() == 0:
            return {k: np.nan for k in [
                'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
                'Asphericity', 'Eccentricity', 'InertialShapeFactor',
                'RadiusOfGyration', 'SpherocityIndex'
            ]}
    
    # Optimize geometry
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    
    # Compute descriptors for each conformer and average
    descriptors_list = []
    for conf_id in range(mol.GetNumConformers()):
        try:
            desc = {
                'PMI1': Descriptors3D.PMI1(mol, confId=conf_id),
                'PMI2': Descriptors3D.PMI2(mol, confId=conf_id),
                'PMI3': Descriptors3D.PMI3(mol, confId=conf_id),
                'NPR1': Descriptors3D.NPR1(mol, confId=conf_id),
                'NPR2': Descriptors3D.NPR2(mol, confId=conf_id),
                'Asphericity': Descriptors3D.Asphericity(mol, confId=conf_id),
                'Eccentricity': Descriptors3D.Eccentricity(mol, confId=conf_id),
                'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol, confId=conf_id),
                'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol, confId=conf_id),
                'SpherocityIndex': Descriptors3D.SpherocityIndex(mol, confId=conf_id),
            }
            descriptors_list.append(desc)
        except Exception:
            continue
    
    if not descriptors_list:
        return {k: np.nan for k in [
            'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
            'Asphericity', 'Eccentricity', 'InertialShapeFactor',
            'RadiusOfGyration', 'SpherocityIndex'
        ]}
    
    # Average over conformers
    avg_desc = {}
    for key in descriptors_list[0].keys():
        values = [d[key] for d in descriptors_list if not np.isnan(d[key])]
        avg_desc[key] = np.mean(values) if values else np.nan
    
    return avg_desc


def compute_ligand_2d_descriptors(smiles: str) -> dict:
    """
    Compute 2D chemical descriptors for a ligand using RDKit.
    
    Args:
        smiles: SMILES string of the ligand
        
    Returns:
        Dictionary of 2D descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in [
            'NumRotatableBonds', 'NumHeavyAtoms', 'NumRings',
            'NumAromaticRings', 'FractionCSP3', 'NumHeteroatoms',
            'NumAliphaticRings', 'NumSaturatedRings', 'RingCount',
            'LabuteASA', 'BalabanJ', 'BertzCT'
        ]}
    
    return {
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'NumHeavyAtoms': rdMolDescriptors.CalcNumHeavyAtoms(mol),
        'NumRings': rdMolDescriptors.CalcNumRings(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'RingCount': Descriptors.RingCount(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),  # Labute's Approximate Surface Area
        'BalabanJ': Descriptors.BalabanJ(mol),  # Balaban's J index (topological)
        'BertzCT': Descriptors.BertzCT(mol),  # Bertz complexity index
    }


def build_ligand_descriptor_table(ligand_smiles: dict) -> pd.DataFrame:
    """
    Build a lookup table with all ligand descriptors.
    
    Args:
        ligand_smiles: Dictionary mapping ligand names to SMILES
        
    Returns:
        DataFrame with ligand descriptors
    """
    records = []
    
    for ligand_name, smiles in ligand_smiles.items():
        print(f"  Computing descriptors for {ligand_name}...")
        
        # Get 3D descriptors
        desc_3d = compute_ligand_3d_descriptors(smiles)
        
        # Get 2D descriptors
        desc_2d = compute_ligand_2d_descriptors(smiles)
        
        # Combine
        record = {'Лиганд': ligand_name}
        
        # Add 3D descriptors with prefix
        for key, value in desc_3d.items():
            record[f'{key} (ligand_3d)'] = value
        
        # Add 2D descriptors with prefix
        for key, value in desc_2d.items():
            record[f'{key} (ligand_2d)'] = value
        
        records.append(record)
    
    return pd.DataFrame(records)


def build_metal_descriptor_table(metal_properties: dict) -> pd.DataFrame:
    """
    Build a lookup table with all metal descriptors.
    
    Args:
        metal_properties: Dictionary of metal properties
        
    Returns:
        DataFrame with metal descriptors
    """
    records = []
    
    for metal_name, props in metal_properties.items():
        record = {'Металл': metal_name}
        
        # Add all properties with prefix
        for key, value in props.items():
            if key != 'hsab_hardness':  # Skip string version
                record[f'{key} (metal_coord)'] = value
        
        records.append(record)
    
    return pd.DataFrame(records)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered interaction features between metals and ligands.
    
    Args:
        df: DataFrame with metal and ligand descriptors
        
    Returns:
        DataFrame with additional interaction features
    """
    df = df.copy()
    
    # Metal-Ligand size ratio (ionic radius / radius of gyration)
    if 'ionic_radius_pm (metal_coord)' in df.columns and 'RadiusOfGyration (ligand_3d)' in df.columns:
        # Convert pm to Angstrom for radius of gyration comparison
        df['Metal_Ligand_Size_Ratio'] = (
            df['ionic_radius_pm (metal_coord)'] / 100  # pm to Angstrom
        ) / df['RadiusOfGyration (ligand_3d)'].replace(0, np.nan)
    
    # Electronegativity difference (metal - oxygen in carboxylate)
    # Oxygen electronegativity ~ 3.44
    if 'electronegativity_pauling (metal_coord)' in df.columns:
        df['Metal_O_Electronegativity_Diff'] = 3.44 - df['electronegativity_pauling (metal_coord)']
    
    # Coordination saturation proxy
    # (coordination number * oxidation state) / number of carboxyl groups
    if all(col in df.columns for col in [
        'coordination_number (metal_coord)', 
        'oxidation_state (metal_coord)',
        'carboxyl_groups (ligand)'
    ]):
        df['Coordination_Saturation'] = (
            df['coordination_number (metal_coord)'] * df['oxidation_state (metal_coord)']
        ) / df['carboxyl_groups (ligand)'].replace(0, np.nan)
    
    # HSAB matching score (hardness compatibility)
    # Carboxylates are borderline-hard bases
    if 'hsab_hardness_numeric (metal_coord)' in df.columns:
        # Higher score = better match for carboxylate (borderline-hard)
        df['HSAB_Match_Score'] = 3 - abs(df['hsab_hardness_numeric (metal_coord)'] - 2.5)
    
    # d-electron influence on geometry
    if 'd_electrons (metal_coord)' in df.columns:
        # Jahn-Teller active: d4, d9 (high spin) or d7 (low spin)
        df['Jahn_Teller_Active'] = df['d_electrons (metal_coord)'].isin([4, 9]).astype(int)
    
    # Ligand flexibility index
    if all(col in df.columns for col in ['NumRotatableBonds (ligand_2d)', 'NumHeavyAtoms (ligand_2d)']):
        df['Ligand_Flexibility_Index'] = (
            df['NumRotatableBonds (ligand_2d)'] / df['NumHeavyAtoms (ligand_2d)'].replace(0, np.nan)
        )
    
    # Ligand shape descriptors ratio
    if 'NPR1 (ligand_3d)' in df.columns and 'NPR2 (ligand_3d)' in df.columns:
        df['Ligand_Shape_Anisotropy'] = df['NPR2 (ligand_3d)'] - df['NPR1 (ligand_3d)']
    
    return df


def enrich_dataset(
    input_path: str,
    output_path: str,
    ligand_smiles: dict = LIGAND_SMILES,
    metal_properties: dict = METAL_PROPERTIES,
) -> pd.DataFrame:
    """
    Main function to enrich the dataset with physicochemical descriptors.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        ligand_smiles: Dictionary of ligand SMILES
        metal_properties: Dictionary of metal properties
        
    Returns:
        Enriched DataFrame
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check for required columns
    if 'Лиганд' not in df.columns:
        raise ValueError("Dataset must contain 'Лиганд' column")
    if 'Металл' not in df.columns:
        raise ValueError("Dataset must contain 'Металл' column")
    
    # Check ligand coverage
    dataset_ligands = set(df['Лиганд'].unique())
    known_ligands = set(ligand_smiles.keys())
    missing_ligands = dataset_ligands - known_ligands
    if missing_ligands:
        print(f"  WARNING: Unknown ligands in dataset: {missing_ligands}")
    
    # Check metal coverage
    dataset_metals = set(df['Металл'].unique())
    known_metals = set(metal_properties.keys())
    missing_metals = dataset_metals - known_metals
    if missing_metals:
        print(f"  WARNING: Unknown metals in dataset: {missing_metals}")
    
    # Build descriptor tables
    print("\nComputing ligand descriptors...")
    ligand_table = build_ligand_descriptor_table(ligand_smiles)
    print(f"  Generated {len(ligand_table.columns) - 1} ligand descriptors")
    
    print("\nBuilding metal descriptor table...")
    metal_table = build_metal_descriptor_table(metal_properties)
    print(f"  Generated {len(metal_table.columns) - 1} metal descriptors")
    
    # Merge descriptors into dataset
    print("\nMerging descriptors into dataset...")
    df_enriched = df.merge(ligand_table, on='Лиганд', how='left')
    df_enriched = df_enriched.merge(metal_table, on='Металл', how='left')
    
    # Add interaction features
    print("Adding interaction features...")
    df_enriched = add_interaction_features(df_enriched)
    
    # Count new columns
    new_cols = len(df_enriched.columns) - len(df.columns)
    print(f"  Added {new_cols} new columns")
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset to {output_path}...")
    df_enriched.to_csv(output_path, index=False)
    print(f"  Saved {len(df_enriched)} rows, {len(df_enriched.columns)} columns")
    
    # Print summary of new columns
    print("\n" + "=" * 60)
    print("NEW DESCRIPTOR COLUMNS:")
    print("=" * 60)
    
    new_columns = [c for c in df_enriched.columns if c not in df.columns]
    
    ligand_3d_cols = [c for c in new_columns if '(ligand_3d)' in c]
    ligand_2d_cols = [c for c in new_columns if '(ligand_2d)' in c]
    metal_cols = [c for c in new_columns if '(metal_coord)' in c]
    interaction_cols = [c for c in new_columns if c not in ligand_3d_cols + ligand_2d_cols + metal_cols]
    
    print(f"\nLigand 3D Geometry ({len(ligand_3d_cols)}):")
    for col in ligand_3d_cols:
        print(f"  - {col}")
    
    print(f"\nLigand 2D Chemical ({len(ligand_2d_cols)}):")
    for col in ligand_2d_cols:
        print(f"  - {col}")
    
    print(f"\nMetal Coordination ({len(metal_cols)}):")
    for col in metal_cols:
        print(f"  - {col}")
    
    print(f"\nInteraction Features ({len(interaction_cols)}):")
    for col in interaction_cols:
        print(f"  - {col}")
    
    return df_enriched


def main():
    parser = argparse.ArgumentParser(
        description='Enrich MOF synthesis dataset with physicochemical descriptors'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    enrich_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
