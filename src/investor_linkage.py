"""
investor_linkage.py - Investor Consistency Validation

Validates matches using investor overlap between Crunchbase and Orbis.
Can be used as a feature in the model or as a post-processing guardrail.

STRATEGY:
---------
If a CB startup has known investors (Top 5 Investors), and the Orbis match
shows shareholders (SH - Name), we can validate by checking overlap.

High investor overlap → increases match confidence
Zero overlap when expected → flags for review
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# INVESTOR NORMALIZATION
# =============================================================================

def normalize_investor_name(name: str) -> str:
    """
    Normalize investor/shareholder name for matching.
    
    Handles:
    - Legal suffixes (LP, LLC, Ltd, etc.)
    - Common abbreviations
    - Punctuation and whitespace
    """
    if not name or pd.isna(name):
        return ''
    
    name = str(name).lower().strip()
    
    # Remove common suffixes
    suffixes = [
        r'\b(lp|llp|llc|ltd|inc|corp|gmbh|ag|sa|s\.a\.|s\.r\.l\.|bv|nv)\b\.?',
        r'\b(limited|partners|partnership|ventures|capital|fund|management)\b',
        r'\b(investments|holdings|group)\b',
    ]
    
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name


def parse_investor_list(investor_str: str) -> Set[str]:
    """
    Parse investor string into set of normalized names.
    
    Handles:
    - Comma-separated lists
    - "and" separators
    - Pipe-separated (Orbis style)
    """
    if not investor_str or pd.isna(investor_str):
        return set()
    
    investor_str = str(investor_str)
    
    # Split by common delimiters
    investors = re.split(r'[,|]|\band\b', investor_str)
    
    # Normalize each
    normalized = set()
    for inv in investors:
        norm = normalize_investor_name(inv)
        if norm and len(norm) >= 3:  # Skip very short names
            normalized.add(norm)
    
    return normalized


# =============================================================================
# INVESTOR OVERLAP FEATURES
# =============================================================================

def compute_investor_overlap(
    cb_investors: Set[str],
    orbis_shareholders: Set[str],
) -> Dict:
    """
    Compute investor overlap features.
    
    Args:
        cb_investors: Set of normalized CB investor names
        orbis_shareholders: Set of normalized Orbis shareholder names
    
    Returns:
        Dict of overlap features
    """
    if not cb_investors or not orbis_shareholders:
        return {
            'investor_overlap_count': 0,
            'investor_jaccard': 0.0,
            'investor_cb_coverage': 0.0,
            'has_investor_data': bool(cb_investors) or bool(orbis_shareholders),
        }
    
    # Exact match overlap
    overlap = cb_investors & orbis_shareholders
    
    # Fuzzy match (any token overlap)
    fuzzy_overlap = 0
    for cb_inv in cb_investors:
        cb_tokens = set(cb_inv.split())
        for orbis_sh in orbis_shareholders:
            orbis_tokens = set(orbis_sh.split())
            # If significant token overlap, count as match
            if len(cb_tokens & orbis_tokens) >= max(1, min(len(cb_tokens), len(orbis_tokens)) // 2):
                fuzzy_overlap += 1
                break
    
    union_size = len(cb_investors | orbis_shareholders)
    jaccard = len(overlap) / union_size if union_size > 0 else 0
    
    cb_coverage = (len(overlap) + fuzzy_overlap) / len(cb_investors) if cb_investors else 0
    
    return {
        'investor_overlap_count': len(overlap),
        'investor_fuzzy_overlap': fuzzy_overlap,
        'investor_jaccard': jaccard,
        'investor_cb_coverage': min(cb_coverage, 1.0),
        'has_investor_data': True,
    }


def add_investor_features(
    features_df: pd.DataFrame,
    cb_data: pd.DataFrame,
    orbis_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add investor overlap features to pair features.
    
    Args:
        features_df: Pair features with cb_id, bvd_id
        cb_data: CB data with cb_top5_investors column
        orbis_data: Orbis data with sh_name column (if available)
    
    Returns:
        Features DataFrame with investor columns added
    """
    # Build lookup dicts
    cb_investors_lookup = {}
    for _, row in cb_data.iterrows():
        cb_id = row.get('cb_id', '')
        investors = row.get('cb_top5_investors', '')
        if cb_id:
            cb_investors_lookup[cb_id] = parse_investor_list(investors)
    
    orbis_sh_lookup = {}
    for _, row in orbis_data.iterrows():
        bvd_id = row.get('bvd_id', '')
        shareholders = row.get('sh_name', '')
        if bvd_id:
            orbis_sh_lookup[bvd_id] = parse_investor_list(shareholders)
    
    # Compute features
    investor_features = []
    
    for _, row in features_df.iterrows():
        cb_id = row['cb_id']
        bvd_id = row['bvd_id']
        
        cb_inv = cb_investors_lookup.get(cb_id, set())
        orbis_sh = orbis_sh_lookup.get(bvd_id, set())
        
        overlap = compute_investor_overlap(cb_inv, orbis_sh)
        investor_features.append(overlap)
    
    # Add to dataframe
    result = features_df.copy()
    inv_df = pd.DataFrame(investor_features)
    
    for col in inv_df.columns:
        result[col] = inv_df[col].values
    
    logger.info(f"Added investor features to {len(result)} pairs")
    logger.info(f"  Pairs with investor data: {result['has_investor_data'].sum()}")
    logger.info(f"  Pairs with exact overlap: {(result['investor_overlap_count'] > 0).sum()}")
    
    return result


# =============================================================================
# POST-PROCESSING GUARDRAILS
# =============================================================================

def apply_investor_guardrails(
    matches: pd.DataFrame,
    features_df: pd.DataFrame,
    promote_threshold: float = 0.3,
    demote_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Apply investor-based guardrails to match decisions.
    
    Rules:
    1. If investor_cb_coverage >= promote_threshold → promote tier
    2. If has_investor_data but coverage = 0 for Tier A → flag for review
    
    Args:
        matches: Match decisions
        features_df: Features including investor overlap
        promote_threshold: Coverage threshold for tier promotion
        demote_threshold: Coverage below which to flag
    
    Returns:
        Updated matches with investor adjustments
    """
    # Merge investor features
    result = matches.merge(
        features_df[['cb_id', 'bvd_id', 'investor_overlap_count', 'investor_cb_coverage', 'has_investor_data']],
        on=['cb_id', 'bvd_id'],
        how='left'
    )
    
    # Track adjustments
    result['investor_adjustment'] = 'none'
    
    # Promotion: high investor overlap
    promote_mask = (
        (result['investor_cb_coverage'] >= promote_threshold) &
        (result['tier'].isin(['B', 'C']))
    )
    result.loc[promote_mask, 'investor_adjustment'] = 'promoted'
    
    # Count potential promotions (don't actually change tier, just flag)
    n_promoted = promote_mask.sum()
    
    # Flag: Tier A with investor data but zero overlap (suspicious)
    flag_mask = (
        (result['tier'] == 'A') &
        (result['has_investor_data'] == True) &
        (result['investor_cb_coverage'] == 0)
    )
    result.loc[flag_mask, 'investor_adjustment'] = 'flagged'
    n_flagged = flag_mask.sum()
    
    logger.info(f"Investor guardrails applied:")
    logger.info(f"  Candidates for promotion: {n_promoted}")
    logger.info(f"  Tier A flagged for review: {n_flagged}")
    
    return result


# =============================================================================
# INVESTOR REGISTRY (Optional)
# =============================================================================

class InvestorRegistry:
    """
    Registry of known investors for normalization and matching.
    
    Helps map variations of investor names to canonical forms.
    """
    
    def __init__(self):
        self.canonical_names: Dict[str, str] = {}  # normalized → canonical
        self.aliases: Dict[str, Set[str]] = {}  # canonical → set of aliases
    
    def add_investor(self, canonical: str, aliases: List[str] = None) -> None:
        """Add investor with optional aliases."""
        can_norm = normalize_investor_name(canonical)
        self.canonical_names[can_norm] = canonical
        
        if aliases:
            if canonical not in self.aliases:
                self.aliases[canonical] = set()
            for alias in aliases:
                alias_norm = normalize_investor_name(alias)
                self.canonical_names[alias_norm] = canonical
                self.aliases[canonical].add(alias_norm)
    
    def get_canonical(self, name: str) -> Optional[str]:
        """Get canonical name for an investor."""
        norm = normalize_investor_name(name)
        return self.canonical_names.get(norm)
    
    def are_same(self, name1: str, name2: str) -> bool:
        """Check if two names refer to the same investor."""
        can1 = self.get_canonical(name1) or normalize_investor_name(name1)
        can2 = self.get_canonical(name2) or normalize_investor_name(name2)
        return can1 == can2


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test
    cb_investors = parse_investor_list("Sequoia Capital, Andreessen Horowitz, Index Ventures")
    orbis_shareholders = parse_investor_list("SEQUOIA CAPITAL LP|A16Z|Index Ventures Ltd")
    
    overlap = compute_investor_overlap(cb_investors, orbis_shareholders)
    
    print("=== Investor Overlap Test ===")
    print(f"CB Investors: {cb_investors}")
    print(f"Orbis Shareholders: {orbis_shareholders}")
    print(f"Overlap: {overlap}")
