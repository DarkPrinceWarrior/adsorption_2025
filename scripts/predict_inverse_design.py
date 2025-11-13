"""Run inference with a trained inverse design pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from adsorb_synthesis.data_validation import DEFAULT_VALIDATION_MODE, validate_SEH_data
from adsorb_synthesis.pipeline import InverseDesignPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict synthesis parameters from adsorption descriptors.")
    parser.add_argument(
        "--model",
        default="artifacts",
        help="Directory containing saved pipeline artifacts (created by train_inverse_design.py).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file with adsorption descriptors to run inference on.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save predictions as CSV.",
    )
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help="Return only the predicted synthesis targets without intermediate columns.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("warn", "strict"),
        default=DEFAULT_VALIDATION_MODE,
        help="Validate adsorption descriptors before running inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pipeline = InverseDesignPipeline.load(model_dir)

    df_input = pd.read_csv(args.input)
    validate_SEH_data(df_input, mode=args.validation_mode)
    predictions = pipeline.predict(df_input, return_intermediate=not args.targets_only)

    if args.output:
        output_path = Path(args.output)
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print(predictions)


if __name__ == "__main__":
    main()
