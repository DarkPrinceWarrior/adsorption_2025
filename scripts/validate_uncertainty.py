#!/usr/bin/env python3
"""Internal CV-calibration diagnostic for interval-based uncertainty artifacts."""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS


def validate_uncertainty(models_dir: str, output_dir: str) -> None:
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    calibrators = {}
    cal_path = os.path.join(models_dir, "uncertainty_calibrators.joblib")
    if os.path.exists(cal_path):
        calibrators = joblib.load(cal_path)

    for target in FORWARD_MODEL_TARGETS:
        safe_target = target.replace('/', '_').replace(' ', '_')
        pred_path = os.path.join(models_dir, f"predictions_{safe_target}.csv")
        if not os.path.exists(pred_path):
            print(f"Skipping {target}: missing {pred_path}")
            continue

        df = pd.read_csv(pred_path)
        required_columns = {"y_actual", "y_oof", "y_lo", "y_hi", "interval_width"}
        if not required_columns.issubset(df.columns):
            print(f"Skipping {target}: predictions file has legacy schema.")
            continue
        y_true = df["y_actual"].to_numpy(dtype=float)
        y_oof = df["y_oof"].to_numpy(dtype=float)
        y_lo = df["y_lo"].to_numpy(dtype=float)
        y_hi = df["y_hi"].to_numpy(dtype=float)
        interval_width = df["interval_width"].to_numpy(dtype=float)

        empirical_coverage = float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))
        mean_width = float(np.mean(interval_width))
        print(f"\nValidating {target} (internal CV calibration diagnostic)...")
        print(f"  Coverage: {empirical_coverage:.1%}")
        print(f"  Mean interval width: {mean_width:.4f}")

        df_res = pd.DataFrame({
            "Error": np.abs(y_true - y_oof),
            "IntervalWidth": interval_width,
            "True": y_true,
            "Pred": y_oof,
        }).sort_values("IntervalWidth")

        rejection_rates = np.linspace(0, 0.9, 19)
        maes = []
        r2s = []
        widths = []
        for rejection_rate in rejection_rates:
            n_keep = int(len(df_res) * (1.0 - rejection_rate))
            if n_keep < 10:
                break
            subset = df_res.head(n_keep)
            maes.append(mean_absolute_error(subset["True"], subset["Pred"]))
            widths.append(float(subset["IntervalWidth"].mean()))
            if len(subset) > 10 and subset["True"].std() > 1e-6:
                r2s.append(r2_score(subset["True"], subset["Pred"]))
            else:
                r2s.append(np.nan)

        confidence_level = 0.9
        calibrator = calibrators.get(target)
        if calibrator is not None and hasattr(calibrator, "confidence_level"):
            confidence_level = float(calibrator.confidence_level)

        results[target] = {
            "rejection_rates": rejection_rates[:len(maes)],
            "maes": maes,
            "r2s": r2s,
            "widths": widths,
            "coverage": empirical_coverage,
            "nominal_coverage": confidence_level,
            "mean_width": mean_width,
        }

    valid_targets = [target for target in FORWARD_MODEL_TARGETS if target in results]
    if not valid_targets:
        print("No uncertainty artifacts found.")
        return

    fig, axes = plt.subplots(3, len(valid_targets), figsize=(5 * len(valid_targets), 11), squeeze=False)
    for idx, target in enumerate(valid_targets):
        result = results[target]

        ax = axes[0, idx]
        ax.plot(result["rejection_rates"] * 100, result["maes"], "b-o", markersize=3)
        ax.set_title(f"{target}\nMAE vs rejection")
        ax.set_xlabel("Rejection rate (%)")
        ax.set_ylabel("OOF MAE")
        ax.grid(True, alpha=0.3)

        ax = axes[1, idx]
        ax.plot(result["rejection_rates"] * 100, result["widths"], "g-o", markersize=3)
        ax.set_title("Interval width vs rejection")
        ax.set_xlabel("Rejection rate (%)")
        ax.set_ylabel("Mean interval width")
        ax.grid(True, alpha=0.3)

        ax = axes[2, idx]
        nominal = result["nominal_coverage"]
        empirical = result["coverage"]
        ax.bar(["Empirical"], [empirical], color="#2ecc71" if empirical >= nominal else "#e67e22", width=0.4)
        ax.axhline(nominal, color="black", linestyle="--", linewidth=1.5, label=f"Nominal {nominal:.0%}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage")
        ax.set_title("Interval coverage")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "uncertainty_rejection_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlots saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate interval-based uncertainty artifacts as an internal CV-calibration diagnostic.",
    )
    parser.add_argument("--models-dir", "--models", dest="models_dir", default="artifacts/forward_models")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default="artifacts/plots")
    args = parser.parse_args()
    validate_uncertainty(models_dir=args.models_dir, output_dir=args.output_dir)
