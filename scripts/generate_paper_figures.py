#!/usr/bin/env python3
"""
Generate publication-quality figures for the scientific paper.
Simplified version - loads data directly from CSV.
"""

import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from adsorb_synthesis.data_processing import load_dataset, build_lookup_tables, prepare_forward_dataset
from adsorb_synthesis.constants import RANDOM_SEED, FORWARD_MODEL_TARGETS

# Scientific plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
}


def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_model_performance(metrics: dict, output_dir: str):
    """Figure 1: Model Performance Comparison - R² bar chart."""
    targets = list(metrics.keys())
    r2_test = [metrics[t]['R2_test'] for t in targets]
    r2_train = [metrics[t]['R2_train'] for t in targets]
    
    target_labels = [t.replace(', ', '\n') for t in targets]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(targets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r2_train, width, label='Train', color=COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x + width/2, r2_test, width, label='Test', color=COLORS['secondary'], alpha=0.8)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('R² Score')
    ax.set_xlabel('Target Property')
    ax.set_title('Forward Model Performance: Train vs Test R²')
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_model_performance.png'))
    fig.savefig(os.path.join(output_dir, 'fig1_model_performance.pdf'))
    plt.close(fig)
    print("  Saved: fig1_model_performance.png/pdf")


def plot_parity_plots(data_path: str, models_dir: str, metrics: dict, output_dir: str):
    """Figure 2: Parity Plots (Predicted vs Actual).
    Loads pre-saved predictions from training (predictions_*.csv).
    """
    n_targets = len(metrics)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_targets == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, target in enumerate(metrics.keys()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        # Load pre-saved predictions
        safe_target = target.replace('/', '_').replace(' ', '_')
        predictions_path = os.path.join(models_dir, f"predictions_{safe_target}.csv")
        
        if not os.path.exists(predictions_path):
            ax.text(0.5, 0.5, f'{target}\nno predictions file\nRe-train model', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            continue
        
        # Load predictions
        preds_df = pd.read_csv(predictions_path)
        y_test = preds_df['y_actual'].values
        y_pred = preds_df['y_pred'].values
        y_std = preds_df['y_std'].values
        
        # Plot scatter
        ax.scatter(y_test, y_pred, alpha=0.6, s=30, c=COLORS['primary'], edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        
        # Uncertainty bars
        ax.errorbar(y_test, y_pred, yerr=y_std, fmt='none', alpha=0.3, color=COLORS['secondary'])
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        r2 = metrics[target]['R2_test']
        rmse = metrics[target]['RMSE']
        ax.set_title(f'{target}\nR²={r2:.3f}, RMSE={rmse:.2f}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
    
    for idx in range(n_targets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_parity_plots.png'))
    fig.savefig(os.path.join(output_dir, 'fig2_parity_plots.pdf'))
    plt.close(fig)
    print("  Saved: fig2_parity_plots.png/pdf")


def plot_feature_importance(models_dir: str, metrics: dict, output_dir: str):
    """Figure 3: Feature Importance Analysis."""
    n_targets = len(metrics)
    n_cols = min(2, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_targets == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, target in enumerate(metrics.keys()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        safe_target = target.replace('/', '_').replace(' ', '_')
        model_path = os.path.join(models_dir, f"catboost_{safe_target}_ens0.cbm")
        
        if os.path.exists(model_path):
            model = CatBoostRegressor()
            model.load_model(model_path)
            
            importances = model.get_feature_importance()
            feature_names = model.feature_names_
            
            sorted_idx = np.argsort(importances)[-15:]
            
            clean_names = []
            for f in np.array(feature_names)[sorted_idx]:
                f_clean = f.replace('(metal_coord)', '').replace('(ligand)', '').strip()
                if len(f_clean) > 25:
                    f_clean = f_clean[:22] + '...'
                clean_names.append(f_clean)
            
            colors = [COLORS['accent'] if 'metal' in str(f).lower() or 'ligand' in str(f).lower() else COLORS['primary'] 
                      for f in np.array(feature_names)[sorted_idx]]
            
            ax.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors, alpha=0.8)
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels(clean_names, fontsize=8)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{target}')
        else:
            ax.text(0.5, 0.5, f'{target}\nno model', ha='center', va='center', transform=ax.transAxes)
    
    for idx in range(n_targets, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_feature_importance.png'))
    fig.savefig(os.path.join(output_dir, 'fig3_feature_importance.pdf'))
    plt.close(fig)
    print("  Saved: fig3_feature_importance.png/pdf")


def plot_target_distributions(data_path: str, metrics: dict, output_dir: str):
    """Figure 4: Target Variable Distributions."""
    df = pd.read_csv(data_path)
    
    targets = [t for t in metrics.keys() if t in df.columns]
    n_targets = len(targets)
    
    if n_targets == 0:
        print("  Skipping target distributions: no targets found")
        return
    
    fig, axes = plt.subplots(2, n_targets, figsize=(3*n_targets, 6))
    if n_targets == 1:
        axes = axes.reshape(2, 1)
    
    for idx, target in enumerate(targets):
        data = df[target].dropna()
        
        axes[0, idx].hist(data, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
        axes[0, idx].set_xlabel(target)
        axes[0, idx].set_ylabel('Count')
        axes[0, idx].axvline(data.mean(), color=COLORS['secondary'], linestyle='--', label=f'μ={data.mean():.2f}')
        axes[0, idx].legend(fontsize=8)
        
        bp = axes[1, idx].boxplot(data, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['primary'])
        bp['boxes'][0].set_alpha(0.7)
        axes[1, idx].set_ylabel(target)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_target_distributions.png'))
    fig.savefig(os.path.join(output_dir, 'fig4_target_distributions.pdf'))
    plt.close(fig)
    print("  Saved: fig4_target_distributions.png/pdf")


def plot_metal_distribution(data_path: str, output_dir: str):
    """Figure 5: Distribution by Metal Type."""
    df = pd.read_csv(data_path)
    
    if 'Металл' not in df.columns:
        print("  Skipping metal distribution: column not found")
        return
    
    metal_counts = df['Металл'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metal_counts)))
    axes[0].pie(metal_counts.values, labels=metal_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0].set_title('Sample Distribution by Metal Type')
    
    if 'E0, кДж/моль' in df.columns:
        metal_means = df.groupby('Металл')['E0, кДж/моль'].mean().sort_values(ascending=False)
        axes[1].bar(metal_means.index, metal_means.values, color=COLORS['primary'], alpha=0.8)
        axes[1].set_xlabel('Metal')
        axes[1].set_ylabel('E₀, kJ/mol (mean)')
        axes[1].set_title('Characteristic Energy by Metal Type')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_metal_distribution.png'))
    fig.savefig(os.path.join(output_dir, 'fig5_metal_distribution.pdf'))
    plt.close(fig)
    print("  Saved: fig5_metal_distribution.png/pdf")


def plot_uncertainty_analysis(metrics: dict, output_dir: str):
    """Figure 6: Model Uncertainty Analysis."""
    targets = list(metrics.keys())
    uncertainties = [metrics[t]['Uncertainty_Mean'] for t in targets]
    r2_values = [metrics[t]['R2_test'] for t in targets]
    
    target_labels = [t.replace(', ', '\n') for t in targets]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(targets))
    colors = [COLORS['primary'] if r2 > 0.5 else COLORS['secondary'] for r2 in r2_values]
    
    bars = ax.bar(x, uncertainties, color=colors, alpha=0.8)
    
    ax.set_ylabel('Mean Prediction Uncertainty (σ)')
    ax.set_xlabel('Target Property')
    ax.set_title('Ensemble Prediction Uncertainty by Target')
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    
    for i, (bar, r2) in enumerate(zip(bars, r2_values)):
        ax.annotate(f'R²={r2:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_uncertainty_analysis.png'))
    fig.savefig(os.path.join(output_dir, 'fig6_uncertainty_analysis.pdf'))
    plt.close(fig)
    print("  Saved: fig6_uncertainty_analysis.png/pdf")


def plot_correlation_heatmap(data_path: str, metrics: dict, output_dir: str):
    """Figure 7: Feature Correlation Heatmap."""
    df = pd.read_csv(data_path)
    
    all_features = []
    for target_data in metrics.values():
        all_features.extend(target_data['selected_features'])
    
    feature_counts = pd.Series(all_features).value_counts()
    common_features = feature_counts[feature_counts >= 3].index.tolist()
    
    numeric_features = [f for f in common_features if f in df.columns and df[f].dtype in ['float64', 'int64']][:12]
    
    if len(numeric_features) < 3:
        print("  Skipping correlation heatmap: not enough numeric features")
        return
    
    corr_matrix = df[numeric_features].corr()
    
    clean_names = []
    for f in numeric_features:
        f_clean = f.replace('(metal_coord)', '').replace('(ligand)', '').strip()
        if len(f_clean) > 18:
            f_clean = f_clean[:15] + '...'
        clean_names.append(f_clean)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, xticklabels=clean_names, yticklabels=clean_names,
                ax=ax, annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig7_correlation_heatmap.png'))
    fig.savefig(os.path.join(output_dir, 'fig7_correlation_heatmap.pdf'))
    plt.close(fig)
    print("  Saved: fig7_correlation_heatmap.png/pdf")


def generate_all_figures(data_path: str, models_dir: str, output_dir: str):
    """Generate all figures for the paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Generating Paper Figures ===")
    print(f"  Data: {data_path}")
    print(f"  Models: {models_dir}")
    print(f"  Output: {output_dir}\n")
    
    metrics_path = os.path.join(models_dir, 'metrics.json')
    metrics = load_metrics(metrics_path)
    
    print("Generating figures:")
    
    plot_model_performance(metrics, output_dir)
    plot_parity_plots(data_path, models_dir, metrics, output_dir)
    plot_feature_importance(models_dir, metrics, output_dir)
    plot_target_distributions(data_path, metrics, output_dir)
    plot_metal_distribution(data_path, output_dir)
    plot_uncertainty_analysis(metrics, output_dir)
    plot_correlation_heatmap(data_path, metrics, output_dir)
    
    print(f"\n✓ All figures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--data", type=str, default="data/SEC_SYN_with_features_enriched.csv")
    parser.add_argument("--models", type=str, default="artifacts/forward_models")
    parser.add_argument("--output", type=str, default="artifacts/figures")
    
    args = parser.parse_args()
    generate_all_figures(args.data, args.models, args.output)
