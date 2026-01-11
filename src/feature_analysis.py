"""
feature_analysis.py - Feature Correlation and Selection Analysis

Provides utilities for analyzing feature multicollinearity and selecting
orthogonal features for cleaner model interpretability.

PHD-LEVEL RATIONALE:
--------------------
Multiple string distance metrics (jaro_winkler, token_jaccard, rapidfuzz_ratio)
are highly correlated. This module helps identify and select orthogonal features.
"""

from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_feature_correlations(
    features_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for features.
    
    Args:
        features_df: DataFrame with computed features
        feature_columns: Optional list of columns to include
    
    Returns:
        Correlation matrix DataFrame
    """
    if feature_columns is None:
        # Auto-detect numeric columns
        feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID columns
        feature_columns = [c for c in feature_columns if not c.endswith('_id')]
    
    corr_matrix = features_df[feature_columns].corr()
    
    logger.info(f"Computed correlation matrix for {len(feature_columns)} features")
    
    return corr_matrix


def find_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of features with correlation above threshold.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold (default 0.8)
    
    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    high_corr = []
    cols = corr_matrix.columns.tolist()
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                high_corr.append((cols[i], cols[j], corr))
    
    # Sort by absolute correlation descending
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    logger.info(f"Found {len(high_corr)} highly correlated pairs (|r| >= {threshold})")
    for f1, f2, r in high_corr[:10]:
        logger.info(f"  {f1} <-> {f2}: r={r:.3f}")
    
    return high_corr


# =============================================================================
# ORTHOGONAL FEATURE SELECTION
# =============================================================================

def select_orthogonal_features(
    features_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    threshold: float = 0.8,
    priority_order: Optional[List[str]] = None,
) -> List[str]:
    """
    Select orthogonal features by removing highly correlated redundant features.
    
    Algorithm:
    1. Compute correlation matrix
    2. For each pair with |r| >= threshold, keep the one with higher priority
       (or first alphabetically if no priority given)
    
    Args:
        features_df: DataFrame with features
        feature_columns: Columns to consider
        threshold: Correlation threshold for "redundant"
        priority_order: Optional list defining priority (first = keep)
    
    Returns:
        List of selected orthogonal feature names
    """
    corr_matrix = compute_feature_correlations(features_df, feature_columns)
    high_corr = find_highly_correlated_pairs(corr_matrix, threshold)
    
    # Start with all features
    selected = set(corr_matrix.columns.tolist())
    
    # Priority mapping
    if priority_order:
        priority = {f: i for i, f in enumerate(priority_order)}
    else:
        priority = {}
    
    # Remove redundant features
    for f1, f2, _ in high_corr:
        if f1 not in selected or f2 not in selected:
            continue
        
        # Decide which to remove
        p1 = priority.get(f1, len(priority))
        p2 = priority.get(f2, len(priority))
        
        if p1 <= p2:
            # Keep f1, remove f2
            selected.discard(f2)
            logger.info(f"Removed {f2} (correlated with {f1})")
        else:
            # Keep f2, remove f1
            selected.discard(f1)
            logger.info(f"Removed {f1} (correlated with {f2})")
    
    result = sorted(selected)
    logger.info(f"Selected {len(result)} orthogonal features from {len(corr_matrix.columns)}")
    
    return result


# =============================================================================
# STRING DISTANCE FEATURE ANALYSIS
# =============================================================================

# These are the known highly-correlated string distance features
STRING_DISTANCE_FEATURES = [
    'name_jw',            # Jaro-Winkler
    'name_token_jaccard', # Token Jaccard
    'name_rapidfuzz_ratio', # RapidFuzz token sort
]

# Recommended priority (based on literature and interpretability)
STRING_DISTANCE_PRIORITY = [
    'name_jw',            # Most interpretable, widely used
    'name_token_jaccard', # Captures different aspect than JW
    'name_rapidfuzz_ratio', # Often redundant with JW
]


def analyze_string_distance_redundancy(
    features_df: pd.DataFrame,
) -> Dict:
    """
    Specifically analyze string distance feature redundancy.
    
    Returns:
        Dict with analysis results and recommendations
    """
    available = [f for f in STRING_DISTANCE_FEATURES if f in features_df.columns]
    
    if len(available) < 2:
        return {'status': 'insufficient_features', 'features': available}
    
    corr_matrix = compute_feature_correlations(features_df, available)
    high_corr = find_highly_correlated_pairs(corr_matrix, threshold=0.7)
    
    # Recommendation
    if len(high_corr) > 0:
        recommended = select_orthogonal_features(
            features_df,
            feature_columns=available,
            threshold=0.7,
            priority_order=STRING_DISTANCE_PRIORITY
        )
    else:
        recommended = available
    
    return {
        'status': 'analyzed',
        'available_features': available,
        'correlation_matrix': corr_matrix.to_dict(),
        'high_correlation_pairs': high_corr,
        'recommended_features': recommended,
        'explanation': (
            f"Among string distance features, recommending: {recommended}. "
            f"Removed {len(available) - len(recommended)} redundant features."
        )
    }


# =============================================================================
# FEATURE IMPORTANCE COMPARISON
# =============================================================================

def compare_feature_importance_with_ablation(
    model,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract and format feature importances from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances sorted by magnitude
    """
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model has no feature_importances_ or coef_ attribute")
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['cumulative_importance'] = df['importance'].cumsum() / df['importance'].sum()
    
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Feature Analysis module loaded.")
    print("Key functions:")
    print("  - compute_feature_correlations(features_df)")
    print("  - select_orthogonal_features(features_df, threshold)")
    print("  - analyze_string_distance_redundancy(features_df)")
