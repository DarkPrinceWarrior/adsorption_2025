"""Validate physics constraints on the dataset before training."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import pandas as pd
from adsorb_synthesis.data_processing import load_dataset
from adsorb_synthesis.physics_losses import validate_physics_constraints


def main():
    parser = argparse.ArgumentParser(
        description="Validate physics constraints (thermodynamic consistency, energy bounds) on dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/SEC_SYN_with_features.csv",
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed statistics",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("warn", "strict"),
        default="warn",
        help="Strict mode raises on validation errors; warn mode logs them.",
    )
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.data}")
    df = load_dataset(
        args.data,
        add_categories=True,
        add_salt_features=True,
        validation_mode=args.validation_mode,
    )
    print(f"Dataset shape: {df.shape}")
    print()
    
    print("=" * 70)
    print("PHYSICS CONSTRAINTS VALIDATION")
    print("=" * 70)
    print()
    
    stats = validate_physics_constraints(df, verbose=args.verbose)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print()
    print("Recommendations:")
    if stats.get("thermodynamic_consistency_rate", 0) < 0.8:
        print("  ⚠ Low thermodynamic consistency - consider using soft constraints with tolerance")
    else:
        print("  ✓ Good thermodynamic consistency")
    
    if stats.get("e0_within_bounds_rate", 0) < 0.9:
        print("  ⚠ Some E0 values outside physical range - energy bounds loss will help")
    else:
        print("  ✓ E0 values mostly within physical bounds")
    
    if stats.get("energy_ratio_within_bounds_rate", 0) < 0.9:
        print("  ⚠ Some E/E0 ratios outside physical range - energy ratio constraints recommended")
    else:
        print("  ✓ Energy ratios mostly within physical bounds")
    
    print()
    print("Next steps:")
    print("  1. Train models with physics-informed loss: python scripts/train_inverse_design.py --data data/SEC_SYN_with_features.csv --output artifacts")
    print("  2. Physics loss is now enabled by default for metal and ligand stages")
    print("  3. Monitor training metrics to see impact of physics constraints")


if __name__ == "__main__":
    main()
