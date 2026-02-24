#!/usr/bin/env python3
"""
Uncertainty Validation Script.

Uses saved OOF (out-of-fold) predictions for honest evaluation — each point
was predicted by a model that never saw it during training.

Generates:
1. Rejection Plots (MAE vs Rejection Rate based on ensemble sigma)
2. Conformal Coverage Check (empirical vs nominal coverage)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS


def validate_uncertainty(models_dir: str, output_dir: str):
    """Validate UQ using saved OOF predictions from train_forward_model.py."""

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # Load conformal calibrators
    cal_path = os.path.join(models_dir, "uncertainty_calibrators.joblib")
    calibrators = {}
    if os.path.exists(cal_path):
        try:
            calibrators = joblib.load(cal_path)
        except Exception as e:
            print(f"Warning: could not load calibrators: {e}")

    for target in FORWARD_MODEL_TARGETS:
        safe_target = target.replace('/', '_').replace(' ', '_')
        pred_path = os.path.join(models_dir, f"predictions_{safe_target}.csv")

        if not os.path.exists(pred_path):
            print(f"Skipping {target}: no predictions file at {pred_path}")
            continue

        print(f"\nValidating {target}...")
        df = pd.read_csv(pred_path)

        y_true = df['y_actual'].values
        y_oof = df['y_oof'].values
        sigma = df['y_prod_sigma'].values

        # OOF errors (honest — model never saw these points)
        oof_errors = np.abs(y_true - y_oof)

        # --- Rejection Curve (based on production ensemble sigma) ---
        df_res = pd.DataFrame({
            'Error': oof_errors,
            'Uncertainty': sigma,
            'True': y_true,
            'Pred': y_oof,
        }).sort_values('Uncertainty')

        rejection_rates = np.linspace(0, 0.9, 19)
        maes = []
        r2s = []

        for r in rejection_rates:
            n_keep = int(len(df_res) * (1.0 - r))
            if n_keep < 10:
                break
            subset = df_res.head(n_keep)
            maes.append(mean_absolute_error(subset['True'], subset['Pred']))
            if len(subset) > 10 and subset['True'].std() > 1e-6:
                r2s.append(r2_score(subset['True'], subset['Pred']))
            else:
                r2s.append(np.nan)

        # --- Conformal Coverage Check ---
        cal = calibrators.get(target, {})
        conformal_q = cal.get("conformal_q") if isinstance(cal, dict) else None
        nominal_alpha = cal.get("alpha", 0.10) if isinstance(cal, dict) else 0.10

        coverage = None
        if conformal_q is not None:
            # Prediction interval: y_prod_mean ± conformal_q * sigma
            prod_mean = df['y_prod_mean'].values
            half_width = conformal_q * sigma
            covered = np.abs(y_true - prod_mean) <= half_width
            coverage = float(np.mean(covered))
            print(f"  Conformal coverage: {coverage:.1%} "
                  f"(nominal: {1 - nominal_alpha:.0%}, q={conformal_q:.3f})")

        results[target] = {
            'rejection_rates': rejection_rates[:len(maes)],
            'maes': maes,
            'r2s': r2s,
            'coverage': coverage,
            'nominal_coverage': 1 - nominal_alpha,
        }

        print(f"  OOF MAE (full): {maes[0]:.4f}")
        if len(maes) > len(maes) // 2:
            print(f"  OOF MAE (top-50% confident): {maes[len(maes)//2]:.4f}")

    # ---- Plot ----
    valid_targets = [t for t in FORWARD_MODEL_TARGETS if t in results]
    n_targets = len(valid_targets)
    if n_targets == 0:
        print("No targets to plot.")
        return

    fig, axes = plt.subplots(2, n_targets, figsize=(5 * n_targets, 8),
                             squeeze=False)

    for i, target in enumerate(valid_targets):
        res = results[target]

        # Row 0: Rejection curve
        ax = axes[0, i]
        ax.plot(res['rejection_rates'] * 100, res['maes'], 'b-o', markersize=3)
        ax.set_title(f"{target}\n(OOF-based)")
        ax.set_xlabel('Rejection Rate (%)')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)
        if len(res['maes']) >= 2:
            imp = (res['maes'][0] - res['maes'][-1]) / res['maes'][0] * 100
            ax.text(0.05, 0.95, f"Imp: {imp:.1f}%", transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Row 1: Conformal coverage bar
        ax2 = axes[1, i]
        cov = res.get('coverage')
        nom = res.get('nominal_coverage', 0.9)
        if cov is not None:
            colors = ['#2ecc71' if cov >= nom else '#e74c3c']
            ax2.bar(['Empirical'], [cov], color=colors, width=0.4)
            ax2.axhline(nom, color='black', ls='--', lw=1.5,
                        label=f'Nominal {nom:.0%}')
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Coverage')
            ax2.set_title('Conformal Coverage')
            ax2.legend(fontsize=8)
            ax2.text(0, cov + 0.02, f"{cov:.1%}", ha='center', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No calibrator', transform=ax2.transAxes,
                     ha='center', va='center')
            ax2.set_title('Conformal Coverage')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'uncertainty_rejection_plots.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlots saved to {plot_path}")


if __name__ == "__main__":
    validate_uncertainty(
        models_dir="artifacts/forward_models",
        output_dir="artifacts/plots"
    )
