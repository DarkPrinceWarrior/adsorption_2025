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
        default="data/SEC_SYN_with_features_enriched.csv",
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
    
    report = validate_physics_constraints(df, verbose=args.verbose)
    report_dict = report.to_dict() if hasattr(report, "to_dict") else {}
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if report_dict:
        for key, value in report_dict.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("  (no summary values available)")
    
    print()
    print("Recommendations:")
    er_mean = report_dict.get("energy_ratio_mean", None)
    if er_mean is not None:
        if er_mean > 0.05:
            print("  ⚠ Energy ratio penalty is high — consider enforcing E/E0 ≈ 1/3.")
        else:
            print("  ✓ Energy ratio penalty is low.")

    e0_bounds_mean = report_dict.get("e0_bounds_mean", None)
    if e0_bounds_mean is not None:
        if e0_bounds_mean > 0:
            print("  ⚠ Some E0 values outside physical range — tighten bounds or clean data.")
        else:
            print("  ✓ E0 values within configured bounds.")

    ws_w0_mean = report_dict.get("ws_w0_mean", None)
    if ws_w0_mean is not None:
        if ws_w0_mean > 0:
            print("  ⚠ Ws < W0 detected — check pore volume hierarchy.")
        else:
            print("  ✓ Ws ≥ W0 holds for all rows.")
    
    print()
    print("Next steps:")
    print("  1. Train models with physics-informed loss: python scripts/train_inverse_design.py --data data/SEC_SYN_with_features.csv --output artifacts")
    print("  2. Physics loss is now enabled by default for metal and ligand stages")
    print("  3. Monitor training metrics to see impact of physics constraints")


if __name__ == "__main__":
    main()
