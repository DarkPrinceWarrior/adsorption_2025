"""Train the inverse design pipeline and persist trained artefacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adsorb_synthesis.data_processing import build_lookup_tables, load_dataset
from adsorb_synthesis.pipeline import InverseDesignPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inverse design models for adsorption synthesis.")
    parser.add_argument(
        "--data",
        default="data/SEC_SYN_with_features_DMFA_only_no_Y.csv",
        help="Path to the prepared dataset with engineered features (ДМФА only, Y metal removed).",
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory where trained models and metadata will be stored.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("warn", "strict"),
        default="warn",
        help="Strict mode raises on validation errors; warn mode logs them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_dataset(str(data_path), validation_mode=args.validation_mode)
    lookups = build_lookup_tables(df)

    pipeline = InverseDesignPipeline()
    pipeline.fit(df, lookup_tables=lookups)
    pipeline.save(args.output)

    print("Training completed. Metrics per stage:")
    for stage_name, stage_result in pipeline.stage_results.items():
        metrics_str = ", ".join(f"{metric}: {value:.3f}" for metric, value in stage_result.metrics.items())
        print(f" - {stage_name}: {metrics_str} | cv_mean={stage_result.cv_mean:.3f}, cv_std={stage_result.cv_std:.3f}")

    print(f"Artifacts saved to: {args.output}")


if __name__ == "__main__":
    main()
