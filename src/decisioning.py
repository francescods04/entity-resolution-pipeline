"""
decisioning.py - Match Decisioning & Tier Assignment

Takes scored candidate pairs and produces final match decisions.
Handles: tier assignment, conflict resolution, evidence generation.

TIER SYSTEM:
-----------
A (Auto-accept): p >= 0.98, domain match, country match
B (High confidence): 0.93 <= p < 0.98
C (Review): 0.75 <= p < 0.93
Reject: p < 0.75
"""

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TIER THRESHOLDS
# =============================================================================

TIER_THRESHOLDS = {
    'A': 0.98,  # Auto-accept
    'B': 0.93,  # High confidence
    'C': 0.75,  # Needs review
    # Below C threshold = Reject
}


# =============================================================================
# TIER ASSIGNMENT
# =============================================================================

def assign_tier(
    row: pd.Series,
    thresholds: Dict[str, float] = None,
) -> str:
    """
    Assign tier based on probability and additional signals.
    
    Rules:
    - Tier A: p >= 0.98 AND (domain_exact OR (country_match AND name_jw >= 0.9))
    - Tier B: 0.93 <= p < 0.98 AND country_match
    - Tier C: 0.75 <= p < 0.93
    - Reject: p < 0.75 OR (is_generic_name AND no domain match)
    """
    thresholds = thresholds or TIER_THRESHOLDS
    
    p = row.get('p_match', 0.0)
    domain_exact = row.get('domain_exact', False)
    country_match = row.get('country_match', False)
    name_jw = row.get('name_jw', 0.0)
    is_generic = row.get('is_generic_name', False)
    is_free_email = row.get('is_free_email', False)
    
    # Reject cases
    if p < thresholds['C']:
        return 'Reject'
    
    # Generic name without domain = reject unless very high confidence
    if is_generic and not domain_exact and p < 0.95:
        return 'Reject'
    
    # Free email only match = reject
    if is_free_email and not domain_exact and p < 0.95:
        return 'Reject'
    
    # Tier A: Highest confidence
    if p >= thresholds['A']:
        if domain_exact or (country_match and name_jw >= 0.9):
            return 'A'
        else:
            return 'B'  # High prob but needs more signals
    
    # Tier B: High confidence
    if p >= thresholds['B']:
        if country_match:
            return 'B'
        else:
            return 'C'  # Country mismatch = review
    
    # Tier C: Review needed
    return 'C'


def assign_tiers_batch(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Assign tiers to all scored candidates."""
    result = scored_df.copy()
    result['tier'] = result.apply(assign_tier, axis=1)
    
    # Log tier distribution
    tier_counts = result['tier'].value_counts()
    logger.info("Tier distribution:")
    for tier, count in tier_counts.items():
        logger.info(f"  {tier}: {count:,} ({100*count/len(result):.1f}%)")
    
    return result


# =============================================================================
# MATCH TYPE ASSIGNMENT (Orbis corporate structure)
# =============================================================================

def determine_match_type(row: pd.Series) -> str:
    """
    Determine the type of match based on Orbis corporate structure.
    
    Types:
    - GUO: Match is Global Ultimate Owner
    - LEGAL_ENTITY: Match is the primary operating entity
    - SUBSIDIARY: Match is a subsidiary
    - BRANCH: Match is a branch (same legal entity)
    """
    entity_role = str(row.get('entity_role', '')).upper()
    
    if entity_role == 'GUO':
        return 'GUO'
    elif entity_role == 'BRANCH':
        return 'BRANCH'
    elif entity_role in ['SUBSIDIARY', 'SUB']:
        return 'SUBSIDIARY'
    else:
        return 'LEGAL_ENTITY'


# =============================================================================
# TOP-1 SELECTION WITH CONFLICT RESOLUTION
# =============================================================================

def select_top_matches(
    scored_df: pd.DataFrame,
    max_alternatives: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top-1 match for each CB company, with conflict resolution.
    
    Handles:
    - Multiple candidates per CB company (pick best)
    - Many-to-one: Multiple CB companies matching same Orbis (flag for review)
    
    Args:
        scored_df: DataFrame with p_match and tier columns
        max_alternatives: Number of alternative matches to keep
    
    Returns:
        (matches_df, review_queue_df)
    """
    # Sort by (cb_id, -p_match)
    sorted_df = scored_df.sort_values(['cb_id', 'p_match'], ascending=[True, False])
    
    # Group by cb_id and select top matches
    matches = []
    alternatives_data = []
    
    for cb_id, group in sorted_df.groupby('cb_id'):
        # Skip if no non-reject tiers
        non_reject = group[group['tier'] != 'Reject']
        if len(non_reject) == 0:
            continue
        
        # Top match
        top = non_reject.iloc[0]
        match_record = {
            'cb_id': cb_id,
            'bvd_id': top['bvd_id'],
            'p_match': top['p_match'],
            'tier': top['tier'],
            'match_type': determine_match_type(top),
            'domain_exact': top.get('domain_exact', False),
            'country_match': top.get('country_match', False),
            'name_jw': top.get('name_jw', 0.0),
        }
        
        # Alternatives
        alts = []
        for _, alt in non_reject.iloc[1:max_alternatives+1].iterrows():
            alts.append({
                'bvd_id': alt['bvd_id'],
                'p_match': alt['p_match'],
                'tier': alt['tier'],
            })
        
        match_record['alternatives'] = json.dumps(alts) if alts else None
        match_record['n_alternatives'] = len(alts)
        
        matches.append(match_record)
    
    matches_df = pd.DataFrame(matches)
    
    # Detect many-to-one conflicts (same Orbis matched to multiple CB)
    bvd_counts = matches_df['bvd_id'].value_counts()
    conflicting_bvds = set(bvd_counts[bvd_counts > 1].index)
    
    matches_df['has_conflict'] = matches_df['bvd_id'].isin(conflicting_bvds)
    
    logger.info(f"Selected {len(matches_df):,} matches")
    logger.info(f"  Many-to-one conflicts: {len(conflicting_bvds)}")
    
    # Build review queue
    review_mask = (
        (matches_df['tier'] == 'C') |
        (matches_df['has_conflict']) |
        (matches_df['n_alternatives'] > 3)  # Many close alternatives
    )
    
    review_queue = matches_df[review_mask].copy()
    logger.info(f"  Review queue: {len(review_queue):,} matches")
    
    return matches_df, review_queue


# =============================================================================
# EVIDENCE GENERATION
# =============================================================================

def generate_evidence(row: pd.Series) -> str:
    """
    Generate human-readable evidence JSON for a match decision.
    
    Includes:
    - Match probability and tier
    - Top contributing features
    - Warnings if any
    """
    evidence = {
        'p_match': float(row.get('p_match', 0)),
        'tier': row.get('tier', 'Unknown'),
        'match_type': row.get('match_type', 'Unknown'),
        
        'top_features': [],
        'blocking_sources': row.get('blocking_sources', '').split('|') if row.get('blocking_sources') else [],
        'warnings': [],
    }
    
    # Add top contributing features
    feature_contributions = [
        ('domain_exact', row.get('domain_exact', False), 0.35 if row.get('domain_exact') else 0),
        ('country_match', row.get('country_match', False), 0.15 if row.get('country_match') else 0),
        ('name_jw', row.get('name_jw', 0), row.get('name_jw', 0) * 0.25),
        ('city_sim', row.get('city_sim', 0), row.get('city_sim', 0) * 0.10),
        ('year_compat', row.get('year_compat', 0.5), (row.get('year_compat', 0.5) - 0.5) * 0.10),
    ]
    
    # Sort by contribution
    feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for name, value, contrib in feature_contributions[:5]:
        if contrib != 0:
            evidence['top_features'].append({
                'name': name,
                'value': bool(value) if isinstance(value, (bool, np.bool_)) else round(float(value), 3),
                'contribution': round(contrib, 3),
            })
    
    # Add warnings
    if row.get('is_generic_name', False):
        evidence['warnings'].append('Generic company name - verify manually')
    
    if row.get('is_free_email', False):
        evidence['warnings'].append('Uses free email domain')
    
    if row.get('has_conflict', False):
        evidence['warnings'].append('Multiple CB companies match this Orbis record')
    
    if not row.get('country_match', False):
        evidence['warnings'].append('Country mismatch between CB and Orbis')
    
    return json.dumps(evidence)


def add_evidence_column(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add evidence JSON column to matches dataframe."""
    result = matches_df.copy()
    result['evidence_json'] = result.apply(generate_evidence, axis=1)
    return result


# =============================================================================
# FINAL OUTPUT GENERATION
# =============================================================================

def generate_final_output(
    scored_df: pd.DataFrame,
    output_matches_path: str,
    output_review_path: str,
) -> Dict[str, int]:
    """
    Generate final match output files.
    
    Args:
        scored_df: DataFrame with all features and p_match
        output_matches_path: Path for matches parquet
        output_review_path: Path for review CSV
    
    Returns:
        Stats dict
    """
    # Assign tiers
    tiered_df = assign_tiers_batch(scored_df)
    
    # Select top matches
    matches_df, review_queue = select_top_matches(tiered_df)
    
    # Add evidence
    matches_df = add_evidence_column(matches_df)
    
    # Save matches
    matches_df.to_parquet(output_matches_path, index=False)
    logger.info(f"Saved {len(matches_df)} matches to {output_matches_path}")
    
    # Save review queue as CSV (easier for manual review)
    review_cols = [
        'cb_id', 'bvd_id', 'p_match', 'tier', 'match_type',
        'has_conflict', 'n_alternatives', 'domain_exact', 'country_match', 'name_jw'
    ]
    review_cols = [c for c in review_cols if c in review_queue.columns]
    
    # Add empty columns for human review
    review_queue['human_label'] = ''
    review_queue['human_notes'] = ''
    
    review_queue[review_cols + ['human_label', 'human_notes']].to_csv(
        output_review_path, index=False
    )
    logger.info(f"Saved {len(review_queue)} review items to {output_review_path}")
    
    # Compute stats
    stats = {
        'total_matches': len(matches_df),
        'tier_A': (matches_df['tier'] == 'A').sum(),
        'tier_B': (matches_df['tier'] == 'B').sum(),
        'tier_C': (matches_df['tier'] == 'C').sum(),
        'conflicts': matches_df['has_conflict'].sum(),
        'review_queue_size': len(review_queue),
    }
    
    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test tier assignment
    test_row = pd.Series({
        'p_match': 0.95,
        'domain_exact': True,
        'country_match': True,
        'name_jw': 0.85,
        'is_generic_name': False,
        'is_free_email': False,
    })
    
    tier = assign_tier(test_row)
    print(f"Test tier: {tier}")
    
    evidence = generate_evidence(test_row)
    print(f"Evidence: {evidence}")
