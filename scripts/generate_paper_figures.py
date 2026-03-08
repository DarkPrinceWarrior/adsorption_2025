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
    "neutral": "#6C757D",
}


def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_backend_name(models_dir: str, metrics: dict) -> str:
    payloads = list(metrics.values())
    backend = payloads[0].get("backend") if payloads else None
    if backend:
        return str(backend)
    if any(name.startswith("catboost_") and name.endswith(".cbm") for name in os.listdir(models_dir)):
        return "catboost"
    return os.path.basename(os.path.normpath(models_dir)) or "model"


def collect_model_runs(model_paths: list[str]) -> list[dict]:
    runs: list[dict] = []
    seen: set[str] = set()
    for path in model_paths:
        candidate = os.path.abspath(path)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"Model path not found: {candidate}")
        metrics_path = os.path.join(candidate, "metrics.json")
        if os.path.exists(metrics_path):
            if candidate not in seen:
                metrics = load_metrics(metrics_path)
                runs.append({
                    "backend": infer_backend_name(candidate, metrics),
                    "models_dir": candidate,
                    "metrics": metrics,
                })
                seen.add(candidate)
            continue

        for entry in sorted(os.listdir(candidate)):
            subdir = os.path.join(candidate, entry)
            subdir_metrics = os.path.join(subdir, "metrics.json")
            if not os.path.isdir(subdir) or not os.path.exists(subdir_metrics):
                continue
            subdir = os.path.abspath(subdir)
            if subdir in seen:
                continue
            metrics = load_metrics(subdir_metrics)
            runs.append({
                "backend": infer_backend_name(subdir, metrics),
                "models_dir": subdir,
                "metrics": metrics,
            })
            seen.add(subdir)
    if not runs:
        raise FileNotFoundError("No model artifacts with metrics.json were found.")
    return runs


def collect_targets(runs: list[dict]) -> list[str]:
    targets: list[str] = []
    for run in runs:
        for target in run["metrics"].keys():
            if target not in targets:
                targets.append(target)
    return targets


def metric_as_float(payload: dict, key: str) -> float:
    value = payload.get(key, np.nan)
    return np.nan if value is None else float(value)


def plot_model_performance(runs: list[dict], output_dir: str) -> None:
    targets = collect_targets(runs)
    labels = [target.replace(", ", "\n") for target in targets]
    backends = [run["backend"] for run in runs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(targets))
    width = 0.8 / max(len(runs), 1)
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]

    for idx, run in enumerate(runs):
        color = palette[idx % len(palette)]
        offset = (idx - (len(runs) - 1) / 2.0) * width
        metrics = run["metrics"]
        r2_oof = [
            metric_as_float(metrics.get(target, {}), "R2_oof")
            if metrics.get(target, {}).get("R2_oof") is not None
            else metric_as_float(metrics.get(target, {}), "R2")
            for target in targets
        ]
        r2_prod = [metric_as_float(metrics.get(target, {}), "R2_production") for target in targets]
        axes[0].bar(x + offset, r2_oof, width, label=run["backend"], color=color, alpha=0.85)
        axes[1].bar(x + offset, r2_prod, width, label=run["backend"], color=color, alpha=0.85)

    axes[0].set_ylabel("R²")
    axes[0].set_xlabel("Target")
    axes[0].set_title("OOF Performance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_ylabel("R²")
    axes[1].set_xlabel("Target")
    axes[1].set_title("Production Performance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper right")

    if len(backends) == 1:
        axes[0].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_model_performance.png"))
    fig.savefig(os.path.join(output_dir, "fig1_model_performance.pdf"))
    plt.close(fig)


def plot_parity_plots(runs: list[dict], output_dir: str) -> None:
    targets = collect_targets(runs)
    n_targets = len(targets)
    n_cols = max(1, len(runs))
    n_rows = n_targets
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes.item()]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]

    for row, target in enumerate(targets):
        for col, run in enumerate(runs):
            ax = axes[row, col]
            safe_target = target.replace("/", "_").replace(" ", "_")
            predictions_path = os.path.join(run["models_dir"], f"predictions_{safe_target}.csv")
            if not os.path.exists(predictions_path):
                ax.set_visible(False)
                continue

            preds_df = pd.read_csv(predictions_path)
            y_true = preds_df["y_actual"].to_numpy(dtype=float)
            y_pred = preds_df["y_oof"].to_numpy(dtype=float)
            center_series = preds_df.get("y_interval_center", preds_df["y_oof"])
            center = center_series.to_numpy(dtype=float)
            yerr = None
            if "interval_width" in preds_df.columns:
                interval_width = preds_df["interval_width"].to_numpy(dtype=float)
                if np.isfinite(interval_width).any():
                    yerr = interval_width / 2.0

            color = palette[col % len(palette)]
            ax.scatter(y_true, y_pred, alpha=0.65, s=30, c=color, edgecolors="white", linewidth=0.5)
            if yerr is not None:
                finite_mask = np.isfinite(y_true) & np.isfinite(center) & np.isfinite(yerr)
                if finite_mask.any():
                    ax.errorbar(
                        y_true[finite_mask],
                        center[finite_mask],
                        yerr=yerr[finite_mask],
                        fmt="none",
                        alpha=0.18,
                        color=COLORS["neutral"],
                    )

            lim_min = min(np.nanmin(y_true), np.nanmin(y_pred))
            lim_max = max(np.nanmax(y_true), np.nanmax(y_pred))
            pad = (lim_max - lim_min) * 0.05 if lim_max > lim_min else 1.0
            lims = [lim_min - pad, lim_max + pad]
            ax.plot(lims, lims, "k--", alpha=0.75)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Actual")
            ax.set_ylabel("OOF prediction")
            score = run["metrics"].get(target, {}).get("R2_oof", np.nan)
            ax.set_title(f"{run['backend']} | {target}\nR²={score:.3f}")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_parity_plots.png"))
    fig.savefig(os.path.join(output_dir, "fig2_parity_plots.pdf"))
    plt.close(fig)


def plot_feature_importance(runs: list[dict], output_dir: str) -> None:
    catboost_runs = [run for run in runs if run["backend"] == "catboost"]
    if not catboost_runs:
        return
    run = catboost_runs[0]
    models_dir = run["models_dir"]
    metrics = run["metrics"]
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
        ax.set_title(f"{target} ({run['backend']})")

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


def plot_uncertainty_analysis(runs: list[dict], output_dir: str) -> None:
    targets = collect_targets(runs)
    labels = [target.replace(", ", "\n") for target in targets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(targets))
    width = 0.8 / max(len(runs), 1)
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]
    plotted = False

    for idx, run in enumerate(runs):
        metrics = run["metrics"]
        widths = [metric_as_float(metrics.get(target, {}), "interval_width_mean") for target in targets]
        coverages = [metric_as_float(metrics.get(target, {}), "interval_coverage") for target in targets]
        if not np.isfinite(widths).any() and not np.isfinite(coverages).any():
            continue
        plotted = True
        offset = (idx - (len(runs) - 1) / 2.0) * width
        color = palette[idx % len(palette)]
        axes[0].bar(x + offset, widths, width, color=color, alpha=0.8, label=run["backend"])
        axes[1].bar(x + offset, coverages, width, color=color, alpha=0.8, label=run["backend"])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean interval width")
    axes[0].set_title("Prediction interval width")

    axes[1].axhline(0.9, linestyle="--", color="black", linewidth=1.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("Interval coverage")
    if plotted:
        axes[1].legend(loc="upper right")
    else:
        axes[0].text(0.5, 0.5, "No interval UQ available", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No interval UQ available", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_uncertainty_analysis.png"))
    fig.savefig(os.path.join(output_dir, "fig6_uncertainty_analysis.pdf"))
    plt.close(fig)


def plot_correlation_heatmap(data_path: str, runs: list[dict], output_dir: str) -> None:
    df = pd.read_csv(data_path)
    all_features = []
    for run in runs:
        for payload in run["metrics"].values():
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


def generate_all_figures(data_path: str, model_paths: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    runs = collect_model_runs(model_paths)
    reference_metrics = runs[0]["metrics"]
    plot_model_performance(runs, output_dir)
    plot_parity_plots(runs, output_dir)
    plot_feature_importance(runs, output_dir)
    plot_target_distributions(data_path, reference_metrics, output_dir)
    plot_uncertainty_analysis(runs, output_dir)
    plot_correlation_heatmap(data_path, runs, output_dir)
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models", nargs="+", default=["artifacts/forward_models"])
    parser.add_argument("--output", type=str, default="artifacts/figures")
    args = parser.parse_args()
    generate_all_figures(args.data, args.models, args.output)
