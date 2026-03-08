#!/usr/bin/env python3
"""Generate a compact comparative report for wave 2 benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    "grid.alpha": 0.25,
})

COLORS = {
    "catboost": "#2E86AB",
    "tabpfn": "#A23B72",
    "bofire": "#2E86AB",
    "botorch": "#F18F01",
    "baybe": "#C73E1D",
    "inverse_direct": "#5C7AEA",
    "default": "#6C757D",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _backend_color(backend: str) -> str:
    return COLORS.get(str(backend), COLORS["default"])


def _infer_inverse_metrics_path(shortlist_path: str) -> Optional[Path]:
    path = Path(shortlist_path)
    candidate = path.parent / "metrics.json"
    if candidate.exists():
        return candidate
    return None


def _load_inverse_direct_metrics(inverse_direct_benchmark: pd.DataFrame) -> Optional[Dict[str, object]]:
    if inverse_direct_benchmark.empty:
        return None
    metrics_path = Path(str(inverse_direct_benchmark.iloc[0].get("metrics_path", "")))
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_forward_report(forward_benchmark: pd.DataFrame, output_dir: Path) -> None:
    if forward_benchmark.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pivot_oof = forward_benchmark.pivot(index="target", columns="backend", values="R2_oof")
    pivot_prod = forward_benchmark.pivot(index="target", columns="backend", values="R2_production")
    pivot_holdout = forward_benchmark.pivot(index="target", columns="backend", values="R2_holdout")

    sns.heatmap(pivot_oof, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Forward OOF R²")
    axes[0].set_xlabel("Backend")
    axes[0].set_ylabel("Target")

    sns.heatmap(pivot_prod, annot=True, fmt=".3f", cmap="Greens", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Forward Production R²")
    axes[1].set_xlabel("Backend")
    axes[1].set_ylabel("")

    sns.heatmap(pivot_holdout, annot=True, fmt=".3f", cmap="Oranges", vmin=0, vmax=1, ax=axes[2])
    axes[2].set_title("Forward External Holdout R²")
    axes[2].set_xlabel("Backend")
    axes[2].set_ylabel("")

    fig.savefig(output_dir / "wave2_forward_overview.png")
    fig.savefig(output_dir / "wave2_forward_overview.pdf")
    plt.close(fig)


def plot_inverse_optimizer_report(optimizer_df: pd.DataFrame, output_dir: Path) -> None:
    if optimizer_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ordered = optimizer_df.sort_values("best_score_pool", ascending=True)
    backends = ordered["backend"].tolist()
    x = np.arange(len(backends))
    colors = [_backend_color(backend) for backend in backends]

    axes[0, 0].bar(x, ordered["best_score_pool"], color=colors, alpha=0.9)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(backends)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Best inverse score")

    axes[0, 1].bar(x, ordered["mean_score_pool"], color=colors, alpha=0.9)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(backends)
    axes[0, 1].set_ylabel("Mean score")
    axes[0, 1].set_title("Mean pool score")

    width = 0.38
    axes[1, 0].bar(x - width / 2, ordered["unique_chemistries_shortlist"], width=width, color=colors, alpha=0.9, label="Chemistries")
    axes[1, 0].bar(x + width / 2, ordered["unique_processes_shortlist"], width=width, color="#8FA6B3", alpha=0.8, label="Processes")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(backends)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Shortlist diversity")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].bar(x - width / 2, ordered["pool_feasibility_rate"], width=width, color=colors, alpha=0.9, label="Pool feasibility")
    axes[1, 1].bar(x + width / 2, ordered["interval_width_total_mean_pool"], width=width, color="#BC6C25", alpha=0.8, label="Mean interval width")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(backends)
    axes[1, 1].set_title("Feasibility and uncertainty diagnostic")
    axes[1, 1].legend(loc="upper right")

    fig.savefig(output_dir / "wave2_inverse_optimizers.png")
    fig.savefig(output_dir / "wave2_inverse_optimizers.pdf")
    plt.close(fig)


def plot_direct_inverse_report(metrics: Dict[str, object], output_dir: Path) -> None:
    if not metrics:
        return

    target_mae = metrics.get("target_mae_recheck", {})
    categorical_accuracy = metrics.get("categorical_accuracy", {})
    if not target_mae and not categorical_accuracy:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    mae_items = list(target_mae.items())
    axes[0].bar(
        np.arange(len(mae_items)),
        [_safe_float(value) for _, value in mae_items],
        color=_backend_color("inverse_direct"),
        alpha=0.9,
    )
    axes[0].set_xticks(np.arange(len(mae_items)))
    axes[0].set_xticklabels([key.replace(", ", "\n") for key, _ in mae_items])
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Direct inverse recheck error")

    acc_items = list(categorical_accuracy.items())
    axes[1].bar(
        np.arange(len(acc_items)),
        [_safe_float(value) for _, value in acc_items],
        color="#7B2CBF",
        alpha=0.9,
    )
    axes[1].set_xticks(np.arange(len(acc_items)))
    axes[1].set_xticklabels(acc_items and [key.replace("Растворитель", "Solvent") for key, _ in acc_items])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Recipe category reconstruction")

    fig.savefig(output_dir / "wave2_direct_inverse.png")
    fig.savefig(output_dir / "wave2_direct_inverse.pdf")
    plt.close(fig)


def build_summary(
    forward_benchmark: pd.DataFrame,
    optimizer_benchmark: pd.DataFrame,
    inverse_direct_benchmark: pd.DataFrame,
    inverse_direct_metrics: Optional[Dict[str, object]],
) -> str:
    lines = [
        "# Wave 2 Comparative Report",
        "",
    ]

    if not forward_benchmark.empty:
        lines.extend(["## Forward", ""])
        for target, target_df in forward_benchmark.groupby("target", sort=False):
            score_column = "R2_holdout" if target_df["R2_holdout"].notna().any() else "R2_oof"
            ordered = target_df.sort_values(score_column, ascending=False).reset_index(drop=True)
            winner = ordered.iloc[0]
            loser = ordered.iloc[-1]
            delta = _safe_float(winner[score_column]) - _safe_float(loser[score_column])
            lines.append(
                f"- `{target}`: лучший backend `{winner['backend']}` по `{score_column}` = `{winner[score_column]:.4f}`; "
                f"разрыв до худшего `{delta:.4f}`."
            )
        lines.append("")

    if not optimizer_benchmark.empty:
        lines.extend(["## Inverse Optimizers", ""])
        best_inverse = optimizer_benchmark.sort_values("best_score_pool", ascending=True).iloc[0]
        diverse_inverse = optimizer_benchmark.sort_values("unique_chemistries_shortlist", ascending=False).iloc[0]
        lines.append(
            f"- Лучший optimizer по `best_score_pool`: `{best_inverse['backend']}` "
            f"(`{best_inverse['best_score_pool']:.4f}`)."
        )
        lines.append(
            f"- Самый разнообразный shortlist по chemistry coverage: `{diverse_inverse['backend']}` "
            f"(`{int(diverse_inverse['unique_chemistries_shortlist'])}` chemistry groups)."
        )
        lines.append("")

    if not inverse_direct_benchmark.empty:
        direct = inverse_direct_benchmark.iloc[0]
        lines.extend(["## Direct Inverse", ""])
        lines.append(
            f"- `inverse_direct` трактуется отдельно от optimizer-ов: "
            f"это benchmark baseline с feasibility `{direct['feasibility_rate']:.3f}`."
        )
        lines.append("")

    if inverse_direct_metrics:
        target_mae = inverse_direct_metrics.get("target_mae_recheck", {})
        categorical_accuracy = inverse_direct_metrics.get("categorical_accuracy", {})
        for target_name, value in target_mae.items():
            lines.append(f"- Recheck MAE для `{target_name}`: `{_safe_float(value):.4f}`.")
        if categorical_accuracy:
            summary_bits = ", ".join(
                f"{name}={_safe_float(value):.3f}"
                for name, value in categorical_accuracy.items()
            )
            lines.append(f"- Reconstruction accuracy по категориям: {summary_bits}.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_wave2_report(benchmark_dir: str, output_dir: str) -> None:
    benchmark_path = Path(benchmark_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    forward_benchmark = _load_csv(benchmark_path / "forward_benchmark.csv")
    optimizer_benchmark = _load_csv(benchmark_path / "inverse_optimizer_benchmark.csv") if (benchmark_path / "inverse_optimizer_benchmark.csv").exists() else _load_csv(benchmark_path / "inverse_benchmark.csv")
    inverse_direct_benchmark = _load_csv(benchmark_path / "inverse_direct_benchmark.csv") if (benchmark_path / "inverse_direct_benchmark.csv").exists() else pd.DataFrame()
    inverse_direct_metrics = _load_inverse_direct_metrics(inverse_direct_benchmark)

    plot_forward_report(forward_benchmark, output_path)
    plot_inverse_optimizer_report(optimizer_benchmark, output_path)
    if inverse_direct_metrics is not None:
        plot_direct_inverse_report(inverse_direct_metrics, output_path)

    summary_md = build_summary(forward_benchmark, optimizer_benchmark, inverse_direct_benchmark, inverse_direct_metrics)
    summary_path = output_path / "wave2_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    summary_json = {
        "forward_rows": int(len(forward_benchmark)),
        "inverse_optimizer_rows": int(len(optimizer_benchmark)),
        "inverse_direct_rows": int(len(inverse_direct_benchmark)),
        "has_inverse_direct_metrics": inverse_direct_metrics is not None,
    }
    with open(output_path / "wave2_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, ensure_ascii=False, indent=2)

    print(f"Saved wave 2 report to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparative wave 2 report from benchmark artifacts.")
    parser.add_argument("--benchmark-dir", type=str, default="artifacts/wave2_benchmark")
    parser.add_argument("--output-dir", type=str, default="artifacts/wave2_report")
    args = parser.parse_args()
    generate_wave2_report(args.benchmark_dir, args.output_dir)


if __name__ == "__main__":
    main()
