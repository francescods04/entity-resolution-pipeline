"""
orbis_graph.py - Orbis Corporate Graph Construction

Builds corporate family structure from Orbis relationship IDs.
Derives family_id, entity_role, and family-level aggregates.

RELATIONSHIPS:
-------------
- GUO (Global Ultimate Owner) - The top of the corporate tree
- SUB (Subsidiary) - Owned by another company
- SH (Shareholder) - Owns shares in another company
- BRANCH - Branch office (same legal entity)

OUTPUT:
-------
- orbis_family.parquet: bvd_id → family_id + entity_role + edges
- orbis_group_features.parquet: family_id → aggregated features
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FAMILY ID & ENTITY ROLE DERIVATION
# =============================================================================

def derive_family_structure(orbis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive family_id and entity_role for each Orbis company.
    
    Rules:
    - family_id = guo_bvd_id if present, else bvd_id
    - entity_role priority: GUO > BRANCH > SUBSIDIARY > ENTITY
    
    Args:
        orbis_df: DataFrame with bvd_id, guo_bvd_id, sub_bvd_id, sh_bvd_id, branch_bvd_id
    
    Returns:
        DataFrame with bvd_id, family_id, entity_role, plus edge counts
    """
    result = []
    
    for _, row in orbis_df.iterrows():
        bvd_id = row.get('bvd_id', '')
        if pd.isna(bvd_id) or not bvd_id:
            continue
        
        guo_id = row.get('guo_bvd_id', '')
        sub_id = row.get('sub_bvd_id', '')
        sh_id = row.get('sh_bvd_id', '')
        branch_id = row.get('branch_bvd_id', '')
        
        # Determine family_id
        if pd.notna(guo_id) and guo_id:
            family_id = str(guo_id)
        else:
            family_id = str(bvd_id)
        
        # Determine entity_role (priority order)
        if pd.notna(guo_id) and str(bvd_id) == str(guo_id):
            entity_role = 'GUO'
        elif pd.notna(branch_id) and branch_id:
            entity_role = 'BRANCH'
        elif pd.notna(sub_id) and sub_id:
            entity_role = 'SUBSIDIARY'
        elif pd.notna(sh_id) and sh_id:
            entity_role = 'ENTITY'  # Has shareholders but not clearly a sub
        else:
            entity_role = 'ENTITY'  # Standalone or unknown
        
        # Count edges
        has_guo = 1 if pd.notna(guo_id) and guo_id else 0
        has_sub = 1 if pd.notna(sub_id) and sub_id else 0
        has_sh = 1 if pd.notna(sh_id) and sh_id else 0
        has_branch = 1 if pd.notna(branch_id) and branch_id else 0
        
        result.append({
            'bvd_id': bvd_id,
            'family_id': family_id,
            'entity_role': entity_role,
            'guo_bvd_id': guo_id if pd.notna(guo_id) else None,
            'has_guo': has_guo,
            'has_sub': has_sub,
            'has_sh': has_sh,
            'has_branch': has_branch,
            'edge_count': has_guo + has_sub + has_sh + has_branch,
        })
    
    family_df = pd.DataFrame(result)
    
    # Log stats
    role_counts = family_df['entity_role'].value_counts()
    logger.info(f"Entity role distribution:")
    for role, count in role_counts.items():
        logger.info(f"  {role}: {count:,}")
    
    unique_families = family_df['family_id'].nunique()
    logger.info(f"Total companies: {len(family_df):,}")
    logger.info(f"Unique families: {unique_families:,}")
    
    return family_df


# =============================================================================
# FAMILY-LEVEL FEATURE AGGREGATION
# =============================================================================

def aggregate_family_features(
    orbis_df: pd.DataFrame,
    family_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute aggregated features at the family (GUO) level.
    
    For each family_id:
    - family_domains_set: union of all domains (excluding free email)
    - family_name_repr: representative name (GUO name or mode)
    - family_countries_set: unique countries
    - family_size: number of members
    - family_has_guo: whether family has a clear GUO
    
    Args:
        orbis_df: Full Orbis data with normalized fields
        family_df: Family structure from derive_family_structure()
    
    Returns:
        DataFrame with one row per family_id
    """
    # Merge family info onto orbis data
    merged = orbis_df.merge(
        family_df[['bvd_id', 'family_id', 'entity_role']], 
        on='bvd_id', 
        how='left'
    )
    
    # Aggregate by family_id
    family_groups = merged.groupby('family_id')
    
    aggregations = []
    
    for family_id, group in family_groups:
        # Domain aggregation (exclude free emails)
        domains = set()
        for website in group['orbis_website'].dropna():
            for domain in str(website).split('|'):
                domain_clean = _extract_domain(domain)
                if domain_clean and not _is_free_email(domain_clean):
                    domains.add(domain_clean)
        
        # Country aggregation
        countries = set(group['orbis_country'].dropna().astype(str))
        
        # City aggregation (top 3)
        cities = group['orbis_city'].dropna().value_counts().head(3).index.tolist()
        
        # Representative name (GUO name if exists, else mode)
        guo_rows = group[group['entity_role'] == 'GUO']
        if len(guo_rows) > 0:
            name_repr = guo_rows['orbis_name'].iloc[0]
            has_guo = True
        else:
            # Use most common name
            name_counts = group['orbis_name'].value_counts()
            name_repr = name_counts.index[0] if len(name_counts) > 0 else None
            has_guo = False
        
        aggregations.append({
            'family_id': family_id,
            'family_name_repr': name_repr,
            'family_domains': '|'.join(sorted(domains)) if domains else None,
            'family_domains_count': len(domains),
            'family_countries': '|'.join(sorted(countries)) if countries else None,
            'family_countries_count': len(countries),
            'family_cities_top': '|'.join(cities) if cities else None,
            'family_size': len(group),
            'family_has_guo': has_guo,
            'family_guo_count': (group['entity_role'] == 'GUO').sum(),
            'family_sub_count': (group['entity_role'] == 'SUBSIDIARY').sum(),
            'family_branch_count': (group['entity_role'] == 'BRANCH').sum(),
        })
    
    family_features = pd.DataFrame(aggregations)
    
    # Log stats
    logger.info(f"Family feature aggregation complete:")
    logger.info(f"  Total families: {len(family_features):,}")
    logger.info(f"  Families with GUO: {family_features['family_has_guo'].sum():,}")
    logger.info(f"  Avg family size: {family_features['family_size'].mean():.1f}")
    logger.info(f"  Max family size: {family_features['family_size'].max()}")
    
    return family_features


def _extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL."""
    if not url:
        return None
    import re
    url = str(url).lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\d?\.', '', url)
    url = url.split('/')[0]
    return url if url else None


def _is_free_email(domain: str) -> bool:
    """Check if domain is a free email provider."""
    FREE_EMAILS = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'live.com', 'icloud.com', 'mail.com', 'aol.com', 'gmx.com',
        'protonmail.com', 'zoho.com', 'yandex.com'
    }
    return domain.lower() in FREE_EMAILS


# =============================================================================
# FAMILY EXPANSION FOR BLOCKING
# =============================================================================

def expand_candidates_by_family(
    candidates: pd.DataFrame,
    family_df: pd.DataFrame,
    max_family_members: int = 10,
) -> pd.DataFrame:
    """
    Expand candidate set by adding family members.
    
    For each candidate (cb_id, bvd_id), also add:
    - The GUO of that family
    - Top members with domains
    
    Args:
        candidates: Original candidate pairs
        family_df: Family structure
        max_family_members: Max additional candidates per family
    
    Returns:
        Expanded candidates with 'FAMILY_EXPAND' source flag
    """
    # Build family lookup
    bvd_to_family = dict(zip(family_df['bvd_id'], family_df['family_id']))
    bvd_to_guo = dict(zip(family_df['bvd_id'], family_df['guo_bvd_id']))
    
    # Group family members
    family_to_members = family_df.groupby('family_id')['bvd_id'].apply(list).to_dict()
    
    # Expand
    expanded = []
    
    for _, row in candidates.iterrows():
        cb_id = row['cb_id']
        bvd_id = row['bvd_id']
        
        family_id = bvd_to_family.get(bvd_id)
        if not family_id:
            continue
        
        # Add GUO if different
        guo_id = bvd_to_guo.get(bvd_id)
        if guo_id and guo_id != bvd_id:
            expanded.append({
                'cb_id': cb_id,
                'bvd_id': guo_id,
                'blocking_sources': 'FAMILY_EXPAND_GUO',
                'blocking_score': 10,
                'rank': 999,
            })
        
        # Add family members (limited)
        members = family_to_members.get(family_id, [])
        for member in members[:max_family_members]:
            if member != bvd_id and member != guo_id:
                expanded.append({
                    'cb_id': cb_id,
                    'bvd_id': member,
                    'blocking_sources': 'FAMILY_EXPAND',
                    'blocking_score': 5,
                    'rank': 999,
                })
    
    if expanded:
        expanded_df = pd.DataFrame(expanded)
        # Combine with original, dedupe
        combined = pd.concat([candidates, expanded_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['cb_id', 'bvd_id'], keep='first')
        
        logger.info(f"Family expansion: {len(candidates)} → {len(combined)} candidates")
        return combined
    
    return candidates


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def build_orbis_family_files(
    orbis_clean_path: str,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Build both family structure files from clean Orbis data.
    
    Args:
        orbis_clean_path: Path to orbis_clean.parquet
        output_dir: Directory to save output files
    
    Returns:
        (orbis_family_path, orbis_group_features_path)
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clean Orbis data
    logger.info(f"Loading Orbis data from {orbis_clean_path}")
    orbis_df = pd.read_parquet(orbis_clean_path)
    
    # Build family structure
    logger.info("Building family structure...")
    family_df = derive_family_structure(orbis_df)
    
    family_path = output_dir / 'orbis_family.parquet'
    family_df.to_parquet(family_path, index=False)
    logger.info(f"Saved family structure to {family_path}")
    
    # Build family aggregates
    logger.info("Aggregating family features...")
    group_features = aggregate_family_features(orbis_df, family_df)
    
    group_path = output_dir / 'orbis_group_features.parquet'
    group_features.to_parquet(group_path, index=False)
    logger.info(f"Saved group features to {group_path}")
    
    return str(family_path), str(group_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Orbis Graph module loaded.")
    print("Key functions:")
    print("  - derive_family_structure(orbis_df)")
    print("  - aggregate_family_features(orbis_df, family_df)")
    print("  - expand_candidates_by_family(candidates, family_df)")
    print("  - build_orbis_family_files(orbis_path, output_dir)")
