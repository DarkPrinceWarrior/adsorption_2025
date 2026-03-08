#!/usr/bin/env python3
"""Generate publication-style figures from current training artifacts."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
}


def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_model_performance(metrics: dict, output_dir: str) -> None:
    targets = list(metrics.keys())
    r2_oof = [float(metrics[target].get("R2_oof", metrics[target].get("R2", np.nan))) for target in targets]
    r2_prod = [float(metrics[target].get("R2_production", np.nan)) for target in targets]
    labels = [target.replace(", ", "\n") for target in targets]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(targets))
    width = 0.35
    ax.bar(x - width / 2, r2_oof, width, label="OOF", color=COLORS["primary"], alpha=0.85)
    ax.bar(x + width / 2, r2_prod, width, label="Production", color=COLORS["secondary"], alpha=0.85)
    ax.set_ylabel("R²")
    ax.set_xlabel("Target")
    ax.set_title("Forward Model Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_model_performance.png"))
    fig.savefig(os.path.join(output_dir, "fig1_model_performance.pdf"))
    plt.close(fig)


def plot_parity_plots(models_dir: str, metrics: dict, output_dir: str) -> None:
    n_targets = len(metrics)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, target in enumerate(metrics.keys()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        safe_target = target.replace('/', '_').replace(' ', '_')
        predictions_path = os.path.join(models_dir, f"predictions_{safe_target}.csv")
        if not os.path.exists(predictions_path):
            ax.set_visible(False)
            continue

        preds_df = pd.read_csv(predictions_path)
        y_true = preds_df["y_actual"].to_numpy(dtype=float)
        y_pred = preds_df["y_oof"].to_numpy(dtype=float)
        center = preds_df.get("y_interval_center", preds_df["y_oof"]).to_numpy(dtype=float)
        yerr = (preds_df["interval_width"].to_numpy(dtype=float) / 2.0)

        ax.scatter(y_true, y_pred, alpha=0.65, s=30, c=COLORS["primary"], edgecolors="white", linewidth=0.5)
        ax.errorbar(y_true, center, yerr=yerr, fmt="none", alpha=0.2, color=COLORS["secondary"])

        lim_min = min(np.nanmin(y_true), np.nanmin(y_pred))
        lim_max = max(np.nanmax(y_true), np.nanmax(y_pred))
        pad = (lim_max - lim_min) * 0.05 if lim_max > lim_min else 1.0
        lims = [lim_min - pad, lim_max + pad]
        ax.plot(lims, lims, "k--", alpha=0.75)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual")
        ax.set_ylabel("OOF prediction")
        ax.set_title(f"{target}\nR²={metrics[target].get('R2_oof', np.nan):.3f}")

    for idx in range(n_targets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_parity_plots.png"))
    fig.savefig(os.path.join(output_dir, "fig2_parity_plots.pdf"))
    plt.close(fig)


def plot_feature_importance(models_dir: str, metrics: dict, output_dir: str) -> None:
    n_targets = len(metrics)
    n_cols = min(2, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, target in enumerate(metrics.keys()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        safe_target = target.replace('/', '_').replace(' ', '_')
        model_path = os.path.join(models_dir, f"catboost_{safe_target}_ens0.cbm")
        if not os.path.exists(model_path):
            ax.set_visible(False)
            continue
        model = CatBoostRegressor()
        model.load_model(model_path)
        importances = model.get_feature_importance()
        feature_names = np.asarray(model.feature_names_)
        top_idx = np.argsort(importances)[-15:]
        top_importances = importances[top_idx]
        top_features = feature_names[top_idx]
        ax.barh(range(len(top_idx)), top_importances, color=COLORS["accent"], alpha=0.8)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(target)

    for idx in range(n_targets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_feature_importance.png"))
    fig.savefig(os.path.join(output_dir, "fig3_feature_importance.pdf"))
    plt.close(fig)


def plot_target_distributions(data_path: str, metrics: dict, output_dir: str) -> None:
    df = pd.read_csv(data_path)
    targets = [target for target in metrics.keys() if target in df.columns]
    if not targets:
        return

    fig, axes = plt.subplots(2, len(targets), figsize=(3.5 * len(targets), 6))
    axes = np.atleast_2d(axes)
    for idx, target in enumerate(targets):
        data = pd.to_numeric(df[target], errors="coerce").dropna()
        axes[0, idx].hist(data, bins=30, color=COLORS["primary"], alpha=0.75, edgecolor="white")
        axes[0, idx].axvline(data.mean(), color=COLORS["secondary"], linestyle="--")
        axes[0, idx].set_xlabel(target)
        axes[0, idx].set_ylabel("Count")

        bp = axes[1, idx].boxplot(data, patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS["primary"])
        bp["boxes"][0].set_alpha(0.75)
        axes[1, idx].set_ylabel(target)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_target_distributions.png"))
    fig.savefig(os.path.join(output_dir, "fig4_target_distributions.pdf"))
    plt.close(fig)


def plot_uncertainty_analysis(metrics: dict, output_dir: str) -> None:
    targets = list(metrics.keys())
    widths = [float(metrics[target].get("interval_width_mean", np.nan)) for target in targets]
    coverages = [float(metrics[target].get("interval_coverage", np.nan)) for target in targets]
    labels = [target.replace(", ", "\n") for target in targets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(targets))
    axes[0].bar(x, widths, color=COLORS["primary"], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean interval width")
    axes[0].set_title("Prediction interval width")

    axes[1].bar(x, coverages, color=COLORS["secondary"], alpha=0.8)
    axes[1].axhline(0.9, linestyle="--", color="black", linewidth=1.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("Interval coverage")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_uncertainty_analysis.png"))
    fig.savefig(os.path.join(output_dir, "fig6_uncertainty_analysis.pdf"))
    plt.close(fig)


def plot_correlation_heatmap(data_path: str, metrics: dict, output_dir: str) -> None:
    df = pd.read_csv(data_path)
    all_features = []
    for payload in metrics.values():
        all_features.extend(payload.get("selected_features", []))
    common_features = pd.Series(all_features).value_counts()
    numeric_features = [
        feature for feature in common_features.index
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
    ][:12]
    if len(numeric_features) < 3:
        return

    corr_matrix = df[numeric_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7_correlation_heatmap.png"))
    fig.savefig(os.path.join(output_dir, "fig7_correlation_heatmap.pdf"))
    plt.close(fig)


def generate_all_figures(data_path: str, models_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    metrics = load_metrics(os.path.join(models_dir, "metrics.json"))
    plot_model_performance(metrics, output_dir)
    plot_parity_plots(models_dir, metrics, output_dir)
    plot_feature_importance(models_dir, metrics, output_dir)
    plot_target_distributions(data_path, metrics, output_dir)
    plot_uncertainty_analysis(metrics, output_dir)
    plot_correlation_heatmap(data_path, metrics, output_dir)
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models", type=str, default="artifacts/forward_models")
    parser.add_argument("--output", type=str, default="artifacts/figures")
    args = parser.parse_args()
    generate_all_figures(args.data, args.models, args.output)
