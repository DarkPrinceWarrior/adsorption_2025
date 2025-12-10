#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a mixed-type correlation matrix for key synthesis and adsorption features
in data/SEC_SYN_with_features.csv. Numeric pairs use Spearman correlation,
categorical-numeric pairs use correlation ratio (eta), and categorical pairs use
Cramer's V. Outputs a CSV with coefficients and a heatmap PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr

# Columns the user requested to analyze (keep the original naming)
COLUMNS_OF_INTEREST: Sequence[str] = (
    "Tрег, ᵒС",
    "W0, см3/г",
    "E0, кДж/моль",
    "х0, нм",
    "а0, ммоль/г",
    "E, кДж/моль",
    "SБЭТ, м2/г",
    "Ws, см3/г",
    "Sme, м2/г",
    "Wme, см3/г",
    "Металл",
    "Лиганд",
    "Растворитель",
    "m (соли), г",
    "m(кис-ты), г",
    "Т.син., °С",
    "Т суш., °С",
    "Vсин. (р-ля), мл",
)

# Columns to treat as categorical when computing mixed correlations
CATEGORICAL_COLUMNS = {"Металл", "Лиганд", "Растворитель"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "SEC_SYN_with_features.csv"
    default_output = repo_root / "analysis_results"

    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed-type correlation matrix for key synthesis and "
            "adsorption features. Saves both CSV and heatmap."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to the source CSV (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for outputs (default: {default_output})",
    )
    return parser.parse_args()


def correlation_ratio(categories: Iterable, values: Iterable) -> float:
    """
    Correlation ratio (eta) for categorical vs numeric features.
    Returns NaN when not enough data or zero variance.
    """
    df = pd.DataFrame({"cat": categories, "val": pd.to_numeric(values, errors="coerce")})
    df = df.dropna()
    if df.empty:
        return np.nan

    grouped = df.groupby("cat")["val"]
    grand_mean = df["val"].mean()

    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in grouped)
    ss_total = ((df["val"] - grand_mean) ** 2).sum()

    if ss_total == 0:
        return np.nan

    return float(np.sqrt(ss_between / ss_total))


def cramers_v(x: Iterable, y: Iterable) -> float:
    """
    Cramer's V for association between two categorical features.
    Uses chi-squared statistic without Yates correction.
    """
    table = pd.crosstab(x, y)
    if table.empty:
        return np.nan

    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    r, k = table.shape

    denom = n * (min(r - 1, k - 1))
    if denom == 0:
        return np.nan

    return float(np.sqrt(chi2 / denom))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation for two numeric series."""
    mask = ~(x.isna() | y.isna())
    if mask.sum() < 2:
        return np.nan
    coef, _ = spearmanr(x[mask], y[mask])
    return float(coef)


def compute_pairwise_correlation(
    series_a: pd.Series, series_b: pd.Series, is_a_categorical: bool, is_b_categorical: bool
) -> float:
    if is_a_categorical and is_b_categorical:
        return cramers_v(series_a, series_b)
    if is_a_categorical or is_b_categorical:
        cats, vals = (series_a, series_b) if is_a_categorical else (series_b, series_a)
        return correlation_ratio(cats, vals)
    return spearman_corr(series_a, series_b)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns to floats and categorical columns to strings so that
    downstream metrics get clean inputs.
    """
    processed = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in CATEGORICAL_COLUMNS:
            processed[col] = df[col].astype("string")
        else:
            processed[col] = pd.to_numeric(df[col], errors="coerce")
    return processed


def build_correlation_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
            val = compute_pairwise_correlation(
                df[col_i],
                df[col_j],
                col_i in CATEGORICAL_COLUMNS,
                col_j in CATEGORICAL_COLUMNS,
            )
            matrix.loc[col_i, col_j] = val
            matrix.loc[col_j, col_i] = val
    return matrix


def plot_heatmap(corr_matrix: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(14, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Correlation / Association"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input)
    selected_columns: List[str] = list(COLUMNS_OF_INTEREST)
    missing = [c for c in selected_columns if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Missing required columns in dataset: {missing}")

    df = prepare_dataframe(df_raw[selected_columns])
    corr_matrix = build_correlation_matrix(df, selected_columns)

    csv_path = args.output_dir / "mixed_correlation_matrix.csv"
    heatmap_path = args.output_dir / "mixed_correlation_heatmap.png"

    corr_matrix.to_csv(csv_path, float_format="%.4f")
    plot_heatmap(corr_matrix, heatmap_path)

    print(f"Saved correlation matrix to {csv_path}")
    print(f"Saved heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
