"""
modeling.py - Machine Learning Model Training & Scoring

Trains a matching model using the platinum/manual matches as training data.
Uses supervised learning with the existing verified matches.

TRAINING STRATEGY:
-----------------
With ~8K platinum matches and ~7K manual matches, we have enough
labeled data for supervised learning without weak supervision.

Positive labels: Platinum domain matches, verified manual matches
Negative labels: Same-country different-company pairs, random negatives
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, average_precision_score
)
import joblib

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# CRITICAL: Features used for LABELING (weak supervision)
# These should NOT be used as model features to avoid leakage
LABELING_FEATURES = frozenset({
    'domain_exact',      # Used to identify positive examples
    'email_domain_exact',  # Used to identify positive examples
})

# Features used for MODEL TRAINING (safe - no leakage)
# These were NOT used to create labels, so they can be learned
TRAINING_FEATURES = [
    # String similarity (primary learning signal)
    'name_jw',
    'name_token_jaccard',
    'name_rapidfuzz_ratio',
    'name_prefix_match',
    'name_acronym_match',
    'alias_best_sim',
    
    # Geographic (secondary signal)
    'country_match',
    'city_sim',
    
    # Temporal
    'year_compat',
    
    # Structural
    'is_guo',
    'is_subsidiary',
    'is_branch',
    'family_size_log',
    
    # Disambiguation
    'disambiguation_score',
    
    # Meta
    'is_generic_name',
    'is_free_email',
    
    # Domain features for INFERENCE only (not training)
    # 'domain_exact',  # EXCLUDED - used for labeling
    # 'domain_in_family',  # EXCLUDED - proxy for domain
]

# Full features for inference (after model is trained)
# Can include domain features since we're not training anymore
INFERENCE_FEATURES = TRAINING_FEATURES + [
    'domain_exact',
    'email_domain_exact',
    'domain_in_family',
]

# Legacy alias for compatibility
MODEL_FEATURES = TRAINING_FEATURES


# =============================================================================
# WEAK SUPERVISION LABEL MODEL (SOTA: FORMALIZED CONFLICT RESOLUTION)
# =============================================================================

class Rule:
    """
    A labeling rule for weak supervision.
    
    Each rule examines features and returns:
    - +1: positive label (match)
    - -1: negative label (non-match)
    - 0: abstain (no opinion)
    """
    
    def __init__(
        self, 
        name: str, 
        condition: callable, 
        label: int,
        weight: float = 1.0,
        description: str = ""
    ):
        self.name = name
        self.condition = condition
        self.label = label  # +1 or -1
        self.weight = weight
        self.description = description
    
    def apply(self, row: Dict) -> int:
        """Apply rule to a row. Returns label (+1/-1) or 0 (abstain)."""
        if self.condition(row):
            return self.label
        return 0


class LabelModel:
    """
    Snorkel-style label model with conflict resolution.
    
    Aggregates votes from multiple labeling rules and resolves conflicts
    using configurable strategies.
    
    SOTA RATIONALE:
    --------------------
    This formalizes the weak supervision approach documented in METHODOLOGY.md.
    Instead of ad-hoc rule application, we explicitly:
    1. Define rules with weights
    2. Resolve conflicts via weighted voting
    3. Provide confidence estimates
    
    Usage:
        label_model = LabelModel(resolution='weighted')
        label_model.add_rule(Rule('domain_match', lambda r: r['domain_exact'], +1, weight=3.0))
        label_model.add_rule(Rule('low_name_sim', lambda r: r['name_jw'] < 0.5, -1, weight=2.0))
        
        labels = label_model.apply_all(features_df)
    """
    
    # Pre-defined labeling rules from METHODOLOGY.md
    DEFAULT_POSITIVE_RULES = [
        # Positive Rules from methodology
        Rule(
            'domain_exact_country_name',
            lambda r: r.get('domain_exact', False) and r.get('country_match', False) and r.get('name_jw', 0) >= 0.85,
            label=+1,
            weight=3.0,
            description="Domain exact AND country match AND name_sim >= 0.85"
        ),
        Rule(
            'domain_family_high_desc',
            lambda r: r.get('domain_in_family', False) and r.get('desc_embedding_cos', 0) >= 0.90,
            label=+1,
            weight=2.5,
            description="Domain in family AND description embedding cos >= 0.90"
        ),
    ]
    
    DEFAULT_NEGATIVE_RULES = [
        # Negative Rules from methodology
        Rule(
            'same_country_diff_domain_low_sim',
            lambda r: r.get('country_match', False) and not r.get('domain_exact', True) and r.get('name_jw', 1) < 0.50,
            label=-1,
            weight=2.0,
            description="Same country AND different domains AND name_sim < 0.50"
        ),
        Rule(
            'different_country_low_sim',
            lambda r: not r.get('country_match', True) and r.get('name_jw', 1) < 0.60,
            label=-1,
            weight=1.5,
            description="Different country AND name_sim < 0.60"
        ),
    ]
    
    def __init__(self, resolution: str = 'weighted'):
        """
        Initialize label model.
        
        Args:
            resolution: Conflict resolution strategy
                - 'majority': Simple majority vote
                - 'weighted': Weighted vote (default, recommended)
                - 'probabilistic': Probabilistic label aggregation
        """
        self.rules: List[Rule] = []
        self.resolution = resolution
    
    def add_rule(self, rule: Rule) -> 'LabelModel':
        """Add a labeling rule."""
        self.rules.append(rule)
        return self
    
    def add_default_rules(self) -> 'LabelModel':
        """Add the default positive and negative rules from methodology."""
        for rule in self.DEFAULT_POSITIVE_RULES:
            self.rules.append(rule)
        for rule in self.DEFAULT_NEGATIVE_RULES:
            self.rules.append(rule)
        logger.info(f"Added {len(self.rules)} default labeling rules")
        return self
    
    def apply(self, row: Dict) -> Tuple[int, float]:
        """
        Apply all rules to a single row.
        
        Args:
            row: Feature dict
        
        Returns:
            (label, confidence) where:
            - label: +1 (positive), -1 (negative), or 0 (abstain)
            - confidence: 0-1 scale confidence in the label
        """
        votes = []
        weights = []
        
        for rule in self.rules:
            vote = rule.apply(row)
            if vote != 0:  # Non-abstain
                votes.append(vote)
                weights.append(rule.weight)
        
        if not votes:
            return (0, 0.0)  # All rules abstained
        
        if self.resolution == 'majority':
            # Simple majority
            pos_votes = sum(1 for v in votes if v > 0)
            neg_votes = sum(1 for v in votes if v < 0)
            
            if pos_votes > neg_votes:
                return (+1, pos_votes / len(votes))
            elif neg_votes > pos_votes:
                return (-1, neg_votes / len(votes))
            else:
                return (0, 0.5)  # Tie
        
        elif self.resolution == 'weighted':
            # Weighted vote
            weighted_sum = sum(v * w for v, w in zip(votes, weights))
            total_weight = sum(weights)
            
            # Normalize to [-1, 1] scale
            score = weighted_sum / total_weight
            
            if score > 0:
                return (+1, min(1.0, abs(score)))
            elif score < 0:
                return (-1, min(1.0, abs(score)))
            else:
                return (0, 0.0)
        
        else:  # probabilistic
            # Treat weights as log-odds
            log_odds = sum(v * np.log(1 + w) for v, w in zip(votes, weights))
            prob = 1 / (1 + np.exp(-log_odds))
            
            if prob > 0.5:
                return (+1, prob)
            elif prob < 0.5:
                return (-1, 1 - prob)
            else:
                return (0, 0.5)
    
    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label model to all rows in a DataFrame.
        
        Args:
            df: Feature DataFrame
        
        Returns:
            DataFrame with 'weak_label' and 'weak_confidence' columns added
        """
        result = df.copy()
        labels = []
        confidences = []
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            label, conf = self.apply(row_dict)
            labels.append(label)
            confidences.append(conf)
        
        result['weak_label'] = labels
        result['weak_confidence'] = confidences
        
        # Log distribution
        n_pos = sum(1 for l in labels if l > 0)
        n_neg = sum(1 for l in labels if l < 0)
        n_abstain = sum(1 for l in labels if l == 0)
        
        logger.info(f"Weak supervision applied to {len(df)} rows:")
        logger.info(f"  Positive: {n_pos}")
        logger.info(f"  Negative: {n_neg}")
        logger.info(f"  Abstain: {n_abstain}")
        
        return result
    
    def get_rule_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for each rule's coverage and conflicts."""
        stats = []
        
        for rule in self.rules:
            covered = 0
            for _, row in df.iterrows():
                if rule.apply(row.to_dict()) != 0:
                    covered += 1
            
            stats.append({
                'rule_name': rule.name,
                'label': 'positive' if rule.label > 0 else 'negative',
                'weight': rule.weight,
                'coverage': covered / len(df) if len(df) > 0 else 0,
                'n_covered': covered,
            })
        
        return pd.DataFrame(stats)


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data_from_matches(
    platinum_matches: pd.DataFrame,
    manual_matches: pd.DataFrame,
    cb_data: pd.DataFrame,
    orbis_data: pd.DataFrame,
    features_df: pd.DataFrame,
    neg_ratio: float = 3.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate labeled training data from verified matches.
    
    Args:
        platinum_matches: Platinum domain-matched pairs (cb_name, orbis_name)
        manual_matches: Manually verified matches
        cb_data: Full Crunchbase data
        orbis_data: Full Orbis data
        features_df: Pre-computed features for candidate pairs
        neg_ratio: Ratio of negatives to positives
        random_state: Random seed
    
    Returns:
        (X, y) where X is feature matrix and y is labels
    """
    np.random.seed(random_state)
    
    # Create positive labels from platinum matches
    positive_pairs = set()
    
    # Add platinum matches
    # Note: Need to map names to IDs
    for _, row in platinum_matches.iterrows():
        cb_name = row.get('nome_df2_match', '')
        orbis_name = row.get('nome_df1', '')
        if cb_name and orbis_name:
            # Create normalized key
            cb_key = str(cb_name).lower().replace(' ', '-')
            positive_pairs.add((cb_key, orbis_name.lower()))
    
    logger.info(f"Created {len(positive_pairs)} positive pairs from matches")
    
    # Find matching pairs in features_df
    features_df['_pair_key'] = features_df.apply(
        lambda r: (str(r['cb_id']).lower(), str(r.get('orbis_name', '')).lower()),
        axis=1
    )
    
    # Label the data
    features_df['label'] = 0
    
    # This is a simplified approach - in practice you'd do proper ID matching
    # For now, we'll use features to identify likely positives
    
    # Positive examples: domain exact match with high name similarity
    positive_mask = (
        (features_df['domain_exact'] == True) & 
        (features_df['country_match'] == True) &
        (features_df['name_jw'] >= 0.7)
    )
    features_df.loc[positive_mask, 'label'] = 1
    
    num_positives = positive_mask.sum()
    logger.info(f"Labeled {num_positives} positive examples based on domain+country+name match")
    
    # =========================================================================
    # HARD NEGATIVE MINING (P1 SOTA OPTIMIZATION)
    # Instead of random negatives, find pairs that are HARD to distinguish:
    # High name similarity but NOT a match (no domain, different entity)
    # =========================================================================
    
    # Strategy 1: Same name similarity range as positives but no domain match
    hard_negatives = features_df[
        (features_df['label'] == 0) &
        (features_df['name_jw'] >= 0.6) &  # Similar names
        (features_df['name_jw'] < 0.9) &   # But not too similar
        (features_df['domain_exact'] == False) &  # No domain match
        (features_df.get('email_domain_exact', False) == False)  # No email match
    ]
    
    # Strategy 2: Different country but similar name (likely different entity)
    cross_country_hard = features_df[
        (features_df['label'] == 0) &
        (features_df['country_match'] == False) &  # Different countries
        (features_df['name_jw'] >= 0.7)  # Very similar names
    ]
    
    # Combine hard negatives
    hard_neg_pool = pd.concat([hard_negatives, cross_country_hard]).drop_duplicates()
    
    # Also include some easy negatives for calibration
    easy_negatives = features_df[
        (features_df['label'] == 0) &
        (features_df['name_jw'] < 0.4) &
        (features_df['domain_exact'] == False)
    ]
    
    logger.info(f"Hard negative pools: similar_name={len(hard_negatives)}, cross_country={len(cross_country_hard)}, easy={len(easy_negatives)}")
    
    # Sample: 60% hard negatives, 40% easy negatives
    n_neg_needed = int(num_positives * neg_ratio)
    n_hard = int(n_neg_needed * 0.6)
    n_easy = n_neg_needed - n_hard
    
    hard_sample = hard_neg_pool.sample(min(n_hard, len(hard_neg_pool)), random_state=random_state)
    easy_sample = easy_negatives.sample(min(n_easy, len(easy_negatives)), random_state=random_state)
    
    neg_indices = list(hard_sample.index) + list(easy_sample.index)
    
    logger.info(f"Sampled {len(hard_sample)} HARD negatives + {len(easy_sample)} easy negatives")
    
    # Create final training set
    train_indices = list(features_df[features_df['label'] == 1].index) + list(neg_indices)
    train_df = features_df.loc[train_indices].copy()
    
    # Clean up temp column
    if '_pair_key' in features_df.columns:
        features_df.drop('_pair_key', axis=1, inplace=True)
    if '_pair_key' in train_df.columns:
        train_df.drop('_pair_key', axis=1, inplace=True)
    
    # Extract features and labels
    available_features = [f for f in MODEL_FEATURES if f in train_df.columns]
    X = train_df[available_features].copy()
    y = train_df['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    
    logger.info(f"Training data: {len(X)} samples, {len(available_features)} features")
    logger.info(f"  Positives: {(y == 1).sum()}, Negatives: {(y == 0).sum()}")
    
    return X, y


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_matching_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'gradient_boosting',
    calibrate: bool = True,
    random_state: int = 42,
) -> Tuple[object, object, Dict]:
    """
    Train a matching model with calibration.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: 'logistic', 'gradient_boosting', or 'random_forest'
        calibrate: Whether to apply isotonic calibration
        random_state: Random seed
    
    Returns:
        (model, calibrator, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if model_type == 'logistic':
        base_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=random_state
        )
    
    # Train model
    logger.info(f"Training {model_type} model...")
    base_model.fit(X_train_scaled, y_train)
    
    # Get raw predictions
    y_pred = base_model.predict(X_test_scaled)
    y_prob = base_model.predict_proba(X_test_scaled)[:, 1]
    
    # Compute metrics
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    
    logger.info(f"Model metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    
    # Calibration
    calibrator = None
    if calibrate:
        logger.info("Applying isotonic calibration...")
        calibrator = CalibratedClassifierCV(
            base_model, method='isotonic', cv=5
        )
        calibrator.fit(X_train_scaled, y_train)
        
        # Re-compute metrics with calibrated model
        y_prob_cal = calibrator.predict_proba(X_test_scaled)[:, 1]
        metrics['roc_auc_calibrated'] = roc_auc_score(y_test, y_prob_cal)
        metrics['avg_precision_calibrated'] = average_precision_score(y_test, y_prob_cal)
    
    # Feature importance (for tree-based models)
    if hasattr(base_model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, base_model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        logger.info("Top 10 features by importance:")
        sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
    
    # Wrap model with scaler
    model_package = {
        'model': calibrator if calibrate else base_model,
        'base_model': base_model,
        'scaler': scaler,
        'features': list(X.columns),
        'model_type': model_type,
        'calibrated': calibrate,
    }
    
    return model_package, metrics


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(
    model_package: Dict,
    metrics: Dict,
    output_dir: str,
    model_name: str = 'company_match'
) -> Dict[str, str]:
    """
    Save model and artifacts to disk.
    
    Args:
        model_package: Dict with model, scaler, features
        metrics: Training metrics
        output_dir: Directory to save to
        model_name: Name prefix for files
    
    Returns:
        Dict mapping artifact types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save model
    model_path = output_dir / f'{model_name}_model.joblib'
    joblib.dump(model_package, model_path)
    paths['model'] = str(model_path)
    
    # Save model card (metadata)
    model_card = {
        'model_type': model_package['model_type'],
        'features': model_package['features'],
        'calibrated': model_package['calibrated'],
        'metrics': {k: v for k, v in metrics.items() if k != 'feature_importance'},
    }
    
    if 'feature_importance' in metrics:
        model_card['feature_importance'] = {
            k: float(v) for k, v in metrics['feature_importance'].items()
        }
    
    card_path = output_dir / f'{model_name}_card.json'
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    paths['model_card'] = str(card_path)
    
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved model card to {card_path}")
    
    return paths


def load_model(model_path: str) -> Dict:
    """Load model package from disk."""
    return joblib.load(model_path)


# =============================================================================
# SCORING
# =============================================================================

def score_candidates(
    features_df: pd.DataFrame,
    model_package: Dict,
) -> pd.DataFrame:
    """
    Score all candidate pairs using trained model.
    
    Args:
        features_df: DataFrame with features (from features.py)
        model_package: Loaded model package
    
    Returns:
        DataFrame with p_match probabilities added
    """
    model = model_package['model']
    scaler = model_package['scaler']
    feature_cols = model_package['features']
    
    # Extract features
    X = features_df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Score
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Add to dataframe
    result = features_df.copy()
    result['p_match'] = probabilities
    
    return result


# =============================================================================
# QUICK TRAINING PIPELINE
# =============================================================================

def train_from_platinum_matches(
    features_path: str,
    output_dir: str,
    platinum_df: Optional[pd.DataFrame] = None,
    model_type: str = 'gradient_boosting',
    cb_data_path: Optional[str] = None,
    orbis_data_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Train model using features and platinum matches (if provided) or heuristics.
    
    Args:
        features_path: Path to computed features parquet
        output_dir: Where to save model
        platinum_df: DataFrame of verified matches (golden labels)
        model_type: 'gradient_boosting' or 'logistic'
        cb_data_path: Path to CB clean parquet (for names)
        orbis_data_path: Path to Orbis clean parquet (for names)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Loading features for training...")
    features_df = pd.read_parquet(features_path)
    
    # If names not in features, load them from source data
    if 'cb_name' not in features_df.columns and cb_data_path:
        logger.info("Merging CB names from source data...")
        cb_df = pd.read_parquet(cb_data_path, columns=['cb_id', 'cb_name'])
        features_df = features_df.merge(cb_df, on='cb_id', how='left')
        
    if 'orbis_name' not in features_df.columns and orbis_data_path:
        logger.info("Merging Orbis names from source data...")
        orbis_df = pd.read_parquet(orbis_data_path, columns=['bvd_id', 'orbis_name'])
        features_df = features_df.merge(orbis_df, on='bvd_id', how='left')
    
    if platinum_df is not None and not platinum_df.empty:
        logger.info(f"Training with {len(platinum_df)} platinum labels")
        
        # 1. Mark positive examples from platinum
        # Create a set of (cb_name_norm, orbis_name_norm) for O(1) lookup
        platinum_pairs = set(zip(
            platinum_df['cb_name_norm'], 
            platinum_df['orbis_name_norm']
        ))
        
        # Normalize features df names
        features_df['cb_name_norm'] = features_df['cb_name'].astype(str).str.lower().str.strip()
        features_df['orbis_name_norm'] = features_df['orbis_name'].astype(str).str.lower().str.strip()
        
        # Create labels
        def get_label(row):
            if (row['cb_name_norm'], row['orbis_name_norm']) in platinum_pairs:
                return 1
            # Heuristic negative: different country + high name sim but not in platinum
            if row['country_match'] == 0 and row['name_token_jaccard'] > 0.5:
                return 0
            return -1 # Unlabeled
            
        features_df['label'] = features_df.apply(get_label, axis=1)
        
        # Filter for labeled data
        labeled_df = features_df[features_df['label'] != -1].copy()
        
        # If we don't have enough negatives, sample some random ones
        n_pos = (labeled_df['label'] == 1).sum()
        n_neg = (labeled_df['label'] == 0).sum()
        
        if n_neg < n_pos:
            logger.info(f"Sampling efficient negatives to balance {n_pos} positives...")
            potential_negatives = features_df[features_df['label'] == -1]
            if not potential_negatives.empty:
                n_needed = n_pos - n_neg
                sampled_neg = potential_negatives.sample(n=min(n_needed * 2, len(potential_negatives)), random_state=42)
                sampled_neg['label'] = 0
                labeled_df = pd.concat([labeled_df, sampled_neg])
        
    else:
        logger.warning("No platinum matches provided! Falling back to heuristic high-precision rules (WEAK SUPERVISION).")
        # Use existing rule-based logic...
        # (This block effectively keeps the old logic as fallback)
        labeled_df = _generate_heuristic_labels(features_df)
    
    logger.info(f"Training data: {len(labeled_df)} samples")
    logger.info(f"Class balance: {labeled_df['label'].value_counts().to_dict()}")
    
    # Get feature columns
    available_features = [f for f in MODEL_FEATURES if f in labeled_df.columns]
    X = labeled_df[available_features].fillna(0)
    y = labeled_df['label'].astype(int)

    # Convert booleans
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Train
    model_package, metrics = train_matching_model(X, y, model_type=model_type)
    
    # Save
    paths = save_model(model_package, metrics, output_dir)
    
    return paths


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Modeling module loaded.")
    print("Key functions:")
    print("  - generate_training_data_from_matches(...)")
    print("  - train_matching_model(X, y)")
    print("  - score_candidates(features_df, model_package)")
    print("  - train_from_platinum_matches(features_path, output_dir)")
