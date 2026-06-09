#!/usr/bin/env python3
"""Feature-selection stability analysis (audit #8).

Runs bootstrap stability selection per target and reports how often each feature is
selected, plus the stable feature set (selected in >= threshold of bootstraps). Use
this to understand and stabilize the otherwise fold-dependent feature sets on this
small dataset. Does not modify the model or the production training path.

Usage:
    PYTHONPATH=src python scripts/feature_stability.py \
        --data data/SEC_SYN_with_features_enriched.csv --bootstraps 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adsorb_synthesis.config import FORWARD_MODEL_CONFIG
from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS
from adsorb_synthesis.data_processing import (
    build_lookup_tables,
    load_dataset,
    prepare_forward_dataset,
)
from adsorb_synthesis.feature_selection import stability_select_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap feature-selection stability analysis.")
    parser.add_argument("--data", default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--output-dir", default="artifacts/feature_stability")
    parser.add_argument("--bootstraps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--validation-mode", choices=["warn", "strict"], default="warn")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, validation_mode=args.validation_mode)
    lookup_tables = build_lookup_tables(df)
    X, y = prepare_forward_dataset(df, lookup_tables=lookup_tables)
    cat = [c for c in X.columns if X[c].dtype.name in ("object", "category")]

    summary = {}
    for target in FORWARD_MODEL_TARGETS:
        if target not in y.columns:
            continue
        print(f"\n=== feature stability: {target} ===")
        stable, freq = stability_select_features(
            X,
            y[target],
            cat,
            n_bootstraps=args.bootstraps,
            selection_threshold=args.threshold,
            corr_threshold=FORWARD_MODEL_CONFIG.feature_selection_corr_threshold,
            vif_threshold=FORWARD_MODEL_CONFIG.feature_selection_vif_threshold,
            max_features=FORWARD_MODEL_CONFIG.feature_selection_max_features,
        )
        for feat, frac in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:15]:
            print(f"  {feat:<40} {frac:.2f}")
        print(f"  stable (>= {args.threshold}): {len(stable)} features incl. {len(cat)} categorical")
        summary[target] = {
            "stable_features": stable,
            "selection_frequency": freq,
            "threshold": args.threshold,
            "bootstraps": args.bootstraps,
        }

    with open(out / "feature_stability.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nFeature stability analysis saved to {out}")


if __name__ == "__main__":
    main()
