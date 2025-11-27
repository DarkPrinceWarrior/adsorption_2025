#!/usr/bin/env python3
"""
Target Distribution Analysis Script.
Visualizes histograms and statistics for all target variables.
Helps decide on stratification strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.data_processing import load_dataset
from adsorb_synthesis.constants import FORWARD_MODEL_TARGETS

def analyze_targets(data_path: str, output_path: str):
    print(f"Loading dataset from {data_path}...")
    df = load_dataset(data_path)
    
    n_targets = len(FORWARD_MODEL_TARGETS)
    cols = 3
    rows = (n_targets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    print("\nTarget Statistics:")
    print("-" * 60)
    print(f"{'Target':<20} | {'Mean':<10} | {'Median':<10} | {'Std':<10} | {'Skew':<10}")
    print("-" * 60)
    
    for i, target in enumerate(FORWARD_MODEL_TARGETS):
        ax = axes[i]
        
        if target not in df.columns:
            ax.text(0.5, 0.5, f"{target} not found", ha='center')
            continue
            
        data = df[target].dropna()
        
        # Statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skew_val = data.skew()
        
        print(f"{target:<20} | {mean_val:<10.2f} | {median_val:<10.2f} | {std_val:<10.2f} | {skew_val:<10.2f}")
        
        # Histogram
        ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
        
        ax.set_title(target)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nDistributions plot saved to {output_path}")

if __name__ == "__main__":
    analyze_targets(
        data_path="data/SEC_SYN_with_features_DMFA_only.csv",
        output_path="artifacts/plots/target_distributions.png"
    )
