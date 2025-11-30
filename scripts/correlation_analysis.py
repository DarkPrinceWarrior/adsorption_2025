import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adsorb_synthesis.data_processing import load_dataset, build_lookup_tables, prepare_forward_dataset


def setup_plotting_style():
    """Sets up a professional and aesthetic plotting style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    # Use a nice color palette
    sns.set_palette("viridis")

def load_and_preprocess_data(filepath):
    """
    Loads the dataset using the SAME pipeline as training.
    This ensures we get ALL engineered features.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    
    # Use the same loading pipeline as training
    df_raw = load_dataset(filepath)
    lookup_tables = build_lookup_tables(df_raw)
    X, y = prepare_forward_dataset(df_raw, lookup_tables=lookup_tables)
    
    # Rename targets for cleaner plots
    y = y.rename(columns={
        'W0, см3/г': 'W0',
        'SБЭТ, м2/г': 'Sbet',
    })
    
    # Also add Solvent column for coloring scatter plots
    X['Solvent'] = df_raw.loc[X.index, 'Растворитель']
    
    return X, y

def get_feature_groups(X, y):
    """
    Get ALL numeric features from X (exactly as used in training).
    Targets: only W0 and Sbet.
    """
    targets = ['W0', 'Sbet']
    
    # Get ALL numeric features from X (excluding categorical)
    valid_features = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            valid_features.append(col)
        else:
            print(f"Skipping non-numeric feature: {col}")
    
    print(f"Total input features for correlation: {len(valid_features)}")
    return valid_features, targets

def plot_correlation_heatmap(X, y, features, targets, output_dir):
    """Plots a heatmap of correlations between Features and Targets."""
    
    # Combine X and y for correlation
    df_combined = pd.concat([X[features], y[targets]], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = df_combined.corr()
    
    # Extract only the Feature vs Target part
    target_corr = corr_matrix.loc[features, targets].sort_values(by='W0', ascending=False)
    
    plt.figure(figsize=(12, len(features) * 0.35 + 3))
    
    # Create a mask for weak correlations to keep the plot clean? No, show all for "powerful" analysis.
    # Use a diverging colormap centered at 0
    heatmap = sns.heatmap(
        target_corr, 
        annot=True, 
        fmt=".2f", 
        cmap="RdBu_r", 
        center=0,
        cbar_kws={'label': 'Pearson Correlation Coefficient'},
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    plt.title('All Input Features Correlation with W0 & Sbet', pad=20, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation heatmap to {save_path}")
    plt.close()
    
    return target_corr

def plot_top_features_scatter(X, y, target_corr, targets, output_dir):
    """
    Plots scatter plots for the top correlated features with each target,
    colored by Solvent.
    """
    n_top = 3  # Number of top features to plot per target
    
    # Combine X and y for plotting
    df = pd.concat([X, y], axis=1)
    
    # Ensure Solvent column is clean for plotting
    if 'Solvent' in df.columns:
        # Keep top 5 solvents, group others as 'Other' to avoid messy legends if too many
        top_solvents = df['Solvent'].value_counts().nlargest(5).index
        df['Solvent_Plot'] = df['Solvent'].apply(lambda x: x if x in top_solvents else 'Other')
    else:
        df['Solvent_Plot'] = 'Unknown'

    for target in targets:
        # Get top positive and negative correlations
        corrs = target_corr[target].abs().sort_values(ascending=False)
        top_features = corrs.head(n_top).index.tolist()
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            
            # Scatter with hue
            sns.scatterplot(
                data=df, 
                x=feature, 
                y=target, 
                hue='Solvent_Plot', 
                style='Solvent_Plot',
                s=100, 
                alpha=0.7,
                palette='deep'
            )
            
            # Add global trend line
            mask = ~df[[feature, target]].isna().any(axis=1)
            if mask.sum() > 1:
                z = np.polyfit(df.loc[mask, feature], df.loc[mask, target], 1)
                p = np.poly1d(z)
                x_range = np.linspace(df[feature].min(), df[feature].max(), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.5, label='Global Trend')
            
            plt.title(f'{target} vs {feature}\nCorrelation: {target_corr.loc[feature, target]:.2f}', fontweight='bold')
            plt.legend(title='Solvent', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            safe_feature_name = feature.replace('/', '_').replace('.', '').replace(' ', '_')
            save_path = os.path.join(output_dir, f'scatter_{target}_vs_{safe_feature_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved scatter plot to {save_path}")
            plt.close()

def plot_pairplot_analysis(X, y, top_features, targets, output_dir):
    """Creates a pairplot for a comprehensive view."""
    # Combine X and y
    df = pd.concat([X, y], axis=1)
    
    # Ensure Solvent_Plot exists
    if 'Solvent' in df.columns:
        top_solvents = df['Solvent'].value_counts().nlargest(5).index
        df['Solvent_Plot'] = df['Solvent'].apply(lambda x: x if x in top_solvents else 'Other')
    else:
        df['Solvent_Plot'] = 'Unknown'
    
    cols_to_plot = top_features + targets + ['Solvent_Plot']
    
    # Check if cols exist
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]
    
    plt.figure(figsize=(15, 15))
    pp = sns.pairplot(
        df[cols_to_plot], 
        hue='Solvent_Plot', 
        palette='deep', 
        corner=True,
        plot_kws={'alpha': 0.6, 's': 50}
    )
    pp.fig.suptitle('Pairwise Relationships: Top Features vs Targets', y=1.02, fontweight='bold')
    
    save_path = os.path.join(output_dir, 'pairplot_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved pairplot to {save_path}")
    plt.close()

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use enriched dataset with all solvents
    data_path = os.path.join(base_dir, 'data', 'SEC_SYN_with_features_enriched.csv')
    output_dir = os.path.join(base_dir, 'analysis_results')
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("Loading and preprocessing data (using training pipeline)...")
    X, y = load_and_preprocess_data(data_path)
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    
    print("\nIdentifying features and targets...")
    features, targets = get_feature_groups(X, y)
    print(f"Found {len(features)} input features and {len(targets)} targets.")
    
    setup_plotting_style()
    
    print("\nGenerating correlation heatmap...")
    target_corr = plot_correlation_heatmap(X, y, features, targets, output_dir)
    
    print("\nGenerating detailed scatter plots...")
    plot_top_features_scatter(X, y, target_corr, targets, output_dir)
    
    print("\nGenerating pairplot analysis...")
    # Select top 3 unique features across both targets for the pairplot
    top_cols = set()
    for t in targets:
        top_cols.update(target_corr[t].abs().nlargest(3).index.tolist())
    plot_pairplot_analysis(X, y, list(top_cols), targets, output_dir)
    
    print("\nAnalysis complete! Check the 'analysis_results' folder.")

if __name__ == "__main__":
    main()
