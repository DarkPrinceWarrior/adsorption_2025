#!/usr/bin/env python3
"""Historical BoFire-domain + Optuna-TPE search backend."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adsorb_synthesis.inverse_optimization import LegacyOptunaBofireOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical BoFire domain models with Optuna-TPE search.")
    parser.add_argument("--E0", type=float, help="Target E0 (kJ/mol)")
    parser.add_argument("--x0", type=float, help="Target x0 (nm)")
    parser.add_argument("--Sme", type=float, help="Target Sme (m2/g)")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--max-per-chemistry", type=int, default=4)
    parser.add_argument("--max-per-process", type=int, default=3)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models-dir", "--models", dest="models_dir", type=str, default="artifacts/forward_models")
    parser.add_argument("--output", type=str, default="artifacts/predictions_bofire_optuna_legacy.csv")
    parser.add_argument("--all-output", type=str, help="Optional path to save the full searched candidate pool.")
    parser.add_argument("--strict-validation", action="store_true")
    args = parser.parse_args()

    targets: Dict[str, float] = {}
    if args.E0 is not None:
        targets["E0, кДж/моль"] = args.E0
    if args.x0 is not None:
        targets["х0, нм"] = args.x0
    if args.Sme is not None:
        targets["Sme, м2/г"] = args.Sme
    if not targets:
        raise SystemExit("Specify at least one target via --E0, --x0, or --Sme.")

    optimizer = LegacyOptunaBofireOptimizer(
        models_dir=args.models_dir,
        data_path=args.data,
        n_trials=args.trials,
        strict_validation=args.strict_validation,
    )
    results, all_results = optimizer.optimize(
        targets,
        shortlist_size=args.shortlist_size,
        max_per_chemistry=args.max_per_chemistry,
        max_per_process=args.max_per_process,
        min_distance=args.min_distance,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    if args.all_output:
        os.makedirs(os.path.dirname(args.all_output) or ".", exist_ok=True)
        all_results.to_csv(args.all_output, index=False)
        print(f"Saved full search pool ({len(all_results)} candidates) to {args.all_output}")
    print(f"Saved shortlist ({len(results)} candidates) to {args.output}")
    if not results.empty:
        print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
