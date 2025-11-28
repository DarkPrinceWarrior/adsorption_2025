"""
Advanced Feature Selection with Multicollinearity Removal.

This module implements a robust feature selection pipeline:
1. Remove highly correlated features (keep most informative)
2. Calculate VIF and remove features with VIF > threshold
3. Use permutation importance for final selection
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def remove_highly_correlated(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.90,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features, keeping the one most correlated with target.
    
    Args:
        X: Feature DataFrame (numeric columns only)
        y: Target Series
        threshold: Correlation threshold above which to remove features
        verbose: Print removal details
        
    Returns:
        X_reduced: DataFrame with reduced features
        removed: List of removed feature names
    """
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Calculate correlation with target for each feature
    target_corr = X.corrwith(y).abs()
    
    # Track features to remove
    removed = []
    features_to_check = list(X.columns)
    
    while True:
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs above threshold
        high_corr_pairs = []
        for col in upper.columns:
            for idx in upper.index:
                if upper.loc[idx, col] > threshold:
                    high_corr_pairs.append((idx, col, upper.loc[idx, col]))
        
        if not high_corr_pairs:
            break
            
        # Sort by correlation (highest first)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Process the highest correlated pair
        feat1, feat2, corr_val = high_corr_pairs[0]
        
        # Keep the one with higher correlation to target
        corr1 = target_corr.get(feat1, 0)
        corr2 = target_corr.get(feat2, 0)
        
        if corr1 >= corr2:
            to_remove = feat2
            kept = feat1
        else:
            to_remove = feat1
            kept = feat2
            
        if verbose:
            print(f"  Removing '{to_remove}' (r={corr_val:.3f} with '{kept}', "
                  f"target_corr: {target_corr.get(to_remove, 0):.3f} vs {target_corr.get(kept, 0):.3f})")
        
        removed.append(to_remove)
        
        # Update matrices
        corr_matrix = corr_matrix.drop(columns=[to_remove], index=[to_remove])
        target_corr = target_corr.drop(to_remove, errors='ignore')
    
    X_reduced = X.drop(columns=removed)
    return X_reduced, removed


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for each feature."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    vif_data = []
    for i, col in enumerate(X_scaled.columns):
        try:
            vif = variance_inflation_factor(X_scaled.values, i)
            vif_data.append({'Feature': col, 'VIF': vif})
        except:
            vif_data.append({'Feature': col, 'VIF': np.inf})
    
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def remove_high_vif_iterative(
    X: pd.DataFrame,
    y: pd.Series,
    vif_threshold: float = 10.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove features with high VIF until all VIF < threshold.
    Removes feature with lowest target correlation when VIF is high.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        vif_threshold: Maximum allowed VIF
        verbose: Print removal details
        
    Returns:
        X_reduced: DataFrame with reduced features
        removed: List of removed feature names
    """
    removed = []
    X_current = X.copy()
    
    max_iterations = len(X.columns)  # Safety limit
    
    for iteration in range(max_iterations):
        if len(X_current.columns) <= 2:
            break
            
        vif_df = calculate_vif(X_current)
        max_vif = vif_df['VIF'].replace([np.inf], np.nan).max()
        
        if pd.isna(max_vif) or max_vif <= vif_threshold:
            break
        
        # Get features with high VIF
        high_vif = vif_df[vif_df['VIF'] > vif_threshold]['Feature'].tolist()
        
        if not high_vif:
            break
            
        # Remove the one with lowest target correlation
        target_corr = X_current[high_vif].corrwith(y).abs()
        to_remove = target_corr.idxmin()
        
        if verbose:
            vif_val = vif_df[vif_df['Feature'] == to_remove]['VIF'].values[0]
            print(f"  VIF iteration {iteration+1}: Removing '{to_remove}' "
                  f"(VIF={vif_val:.1f}, target_corr={target_corr[to_remove]:.3f})")
        
        removed.append(to_remove)
        X_current = X_current.drop(columns=[to_remove])
    
    return X_current, removed


def permutation_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance using RandomForest.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with feature importance scores
    """
    # Fit a quick RandomForest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Calculate permutation importance
    result = permutation_importance(
        rf, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    }).sort_values('Importance_Mean', ascending=False)
    
    return importance_df


def select_features_advanced(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str] = None,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    min_importance: float = 0.001,
    max_features: int = 20,
    verbose: bool = True
) -> Tuple[List[str], Dict]:
    """
    Advanced feature selection pipeline.
    
    Steps:
    1. Separate categorical and numeric features
    2. Remove highly correlated numeric features
    3. Remove features with high VIF
    4. Rank by permutation importance
    5. Select top features
    
    Args:
        X: Full feature DataFrame
        y: Target Series
        categorical_cols: List of categorical column names to preserve
        corr_threshold: Correlation threshold for removal
        vif_threshold: VIF threshold for removal
        min_importance: Minimum importance to keep feature
        max_features: Maximum number of features to select
        verbose: Print details
        
    Returns:
        selected_features: List of selected feature names
        report: Dictionary with selection report
    """
    if categorical_cols is None:
        categorical_cols = []
    
    report = {
        'initial_features': len(X.columns),
        'categorical_preserved': categorical_cols,
        'removed_correlation': [],
        'removed_vif': [],
        'importance_ranking': None
    }
    
    # Separate categorical and numeric
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    X_numeric = X[numeric_cols].copy()
    
    # Convert to numeric
    X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for processing
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    if verbose:
        print(f"\n=== Advanced Feature Selection ===")
        print(f"Initial: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical")
    
    # Step 1: Remove highly correlated features
    if verbose:
        print(f"\n[Step 1] Removing features with correlation > {corr_threshold}")
    
    X_reduced, removed_corr = remove_highly_correlated(
        X_numeric, y, threshold=corr_threshold, verbose=verbose
    )
    report['removed_correlation'] = removed_corr
    
    if verbose:
        print(f"  Removed {len(removed_corr)} features, {len(X_reduced.columns)} remaining")
    
    # Step 2: Remove high VIF features
    if verbose:
        print(f"\n[Step 2] Removing features with VIF > {vif_threshold}")
    
    X_reduced, removed_vif = remove_high_vif_iterative(
        X_reduced, y, vif_threshold=vif_threshold, verbose=verbose
    )
    report['removed_vif'] = removed_vif
    
    if verbose:
        print(f"  Removed {len(removed_vif)} features, {len(X_reduced.columns)} remaining")
    
    # Step 3: Permutation importance
    if verbose:
        print(f"\n[Step 3] Calculating permutation importance...")
    
    importance_df = permutation_feature_importance(X_reduced, y)
    report['importance_ranking'] = importance_df
    
    # Filter by minimum importance
    important_features = importance_df[
        importance_df['Importance_Mean'] > min_importance
    ]['Feature'].tolist()
    
    # Limit to max_features
    important_features = important_features[:max_features]
    
    if verbose:
        print(f"  Top {len(important_features)} features by importance:")
        for i, row in importance_df.head(max_features).iterrows():
            print(f"    {row['Feature']:<40} | {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}")
    
    # Final selection: categorical + important numeric
    selected_features = categorical_cols + important_features
    
    report['final_features'] = len(selected_features)
    report['selected_numeric'] = important_features
    
    if verbose:
        print(f"\n=== Final Selection ===")
        print(f"Selected {len(selected_features)} features: "
              f"{len(categorical_cols)} categorical + {len(important_features)} numeric")
    
    return selected_features, report


# Pre-defined feature groups for domain knowledge
FEATURE_GROUPS = {
    # Keep only one from each correlated group
    'metal_primary': [
        'oxidation_state (metal_coord)',      # Charge state
        'ionic_radius_pm (metal_coord)',      # Size
        'electron_affinity_kj (metal_coord)', # Reactivity
        'Jahn_Teller_Active',                 # Cu2+ marker
    ],
    'metal_drop': [
        'Total molecular weight (metal)',     # Correlated with ionic_radius
        'Average ionic radius (metal)',       # Duplicate
        'Average electronegativity (metal)',  # Correlated with electron_affinity
        'Молярка_соли',                       # Determined by metal
        'hsab_hardness_numeric (metal_coord)',# Correlated with oxidation_state
        'Metal_O_Electronegativity_Diff',     # = electronegativity - const
    ],
    'ligand_primary': [
        'carboxyl_groups (ligand)',           # Number of binding sites
        'molecular_weight (ligand)',          # Size proxy
    ],
    'ligand_drop': [
        'aromatic_rings (ligand)',            # Correlated with carboxyl_groups
        'TPSA (ligand)',                      # Correlated with carboxyl_groups
        'RadiusOfGyration (ligand_3d)',       # Correlated with molecular_weight
        'PMI1 (ligand_3d)',                   # Correlated with molecular_weight
        'Asphericity (ligand_3d)',            # Correlated with carboxyl_groups
        'LabuteASA (ligand_2d)',              # Correlated with molecular_weight
        'Молярка_кислоты',                    # = molecular_weight
    ],
    'recipe_primary': [
        'R_molar',                            # Metal/Ligand ratio (key!)
        'C_metal',                            # Metal concentration
        'Vсин. (р-ля), мл',                   # Volume
    ],
    'recipe_drop': [
        'm (соли), г',                        # = n_соли * MW
        'm(кис-ты), г',                       # = n_кислоты * MW
        'n_соли',                             # Correlated with m
        'n_кислоты',                          # Correlated with m
        'R_mass',                             # Correlated with R_molar
        'C_ligand',                           # Correlated with C_metal
    ],
    'temperature_primary': [
        'Т.син., °С',                         # Synthesis temperature
        'Tрег, ᵒС',                           # Regeneration temperature
        'T_range',                            # T_reg - T_syn
    ],
    'temperature_drop': [
        'Т суш., °С',                         # Correlated with T.син
        'T_activation',                       # = T_reg - T_dry
    ],
    'interaction_primary': [
        'Metal_Ligand_Size_Ratio',            # Geometric matching
    ],
}


def get_curated_features() -> Tuple[List[str], List[str]]:
    """
    Get curated feature list based on domain knowledge.
    
    Returns:
        keep_features: Features to keep
        drop_features: Features to drop (for reference)
    """
    keep = []
    drop = []
    
    for group_name, features in FEATURE_GROUPS.items():
        if 'primary' in group_name:
            keep.extend(features)
        elif 'drop' in group_name:
            drop.extend(features)
    
    return keep, drop
