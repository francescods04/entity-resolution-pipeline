"""
research_analytics.py - Analytics for Economic Research

Generates statistics and visualizations suitable for:
- PhD dissertation appendix
- Journal paper methodology section
- Robustness checks

LEVERAGES EXISTING WORK:
- database-done.xlsx platinum matches for validation
- Compares our matches to pre-existing verified matches

OUTPUT:
-------
- research_stats.json: Key statistics for paper
- match_summary.csv: Per-country/industry breakdown
- validation_report.md: Comparison with platinum matches
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CORE STATISTICS FOR PAPERS
# =============================================================================

def compute_research_statistics(
    matches: pd.DataFrame,
    cb_data: pd.DataFrame,
    orbis_data: pd.DataFrame,
) -> Dict:
    """
    Compute statistics suitable for academic papers.
    
    Returns dict with:
    - Sample sizes (N)
    - Match rates
    - Tier distribution
    - Quality metrics
    """
    stats = {
        'generated_at': datetime.now().isoformat(),
        'data_version': '1.0',
    }
    
    # =========================================================================
    # SAMPLE SIZES (Table 1 in any paper)
    # =========================================================================
    stats['sample_sizes'] = {
        'crunchbase_total': len(cb_data),
        'orbis_total': len(orbis_data),
        'matched_total': len(matches),
        'match_rate': round(len(matches) / len(cb_data) * 100, 1) if len(cb_data) > 0 else 0,
    }
    
    # =========================================================================
    # TIER DISTRIBUTION (Appendix Table)
    # =========================================================================
    if 'tier' in matches.columns:
        tier_counts = matches['tier'].value_counts()
        stats['tier_distribution'] = {
            'tier_A': int(tier_counts.get('A', 0)),
            'tier_B': int(tier_counts.get('B', 0)),
            'tier_C': int(tier_counts.get('C', 0)),
            'rejected': int(tier_counts.get('Reject', 0)),
        }
        
        # Percentages
        total = len(matches)
        stats['tier_percentages'] = {
            'tier_A_pct': round(tier_counts.get('A', 0) / total * 100, 1),
            'tier_B_pct': round(tier_counts.get('B', 0) / total * 100, 1),
            'tier_C_pct': round(tier_counts.get('C', 0) / total * 100, 1),
        }
    
    # =========================================================================
    # CONFIDENCE DISTRIBUTION
    # =========================================================================
    if 'p_match' in matches.columns:
        stats['confidence'] = {
            'mean': round(matches['p_match'].mean(), 3),
            'median': round(matches['p_match'].median(), 3),
            'std': round(matches['p_match'].std(), 3),
            'p25': round(matches['p_match'].quantile(0.25), 3),
            'p75': round(matches['p_match'].quantile(0.75), 3),
            'p90': round(matches['p_match'].quantile(0.90), 3),
            'p95': round(matches['p_match'].quantile(0.95), 3),
        }
    
    # =========================================================================
    # MATCH TYPE DISTRIBUTION (for corporate structure analysis)
    # =========================================================================
    if 'match_type' in matches.columns:
        type_counts = matches['match_type'].value_counts()
        stats['match_types'] = {
            k: int(v) for k, v in type_counts.items()
        }
    
    # =========================================================================
    # DATA AVAILABILITY (for robustness discussion)
    # =========================================================================
    stats['data_availability'] = {
        'cb_with_website': int(cb_data['cb_website'].notna().sum()) if 'cb_website' in cb_data.columns else 0,
        'cb_with_email': int(cb_data['cb_email'].notna().sum()) if 'cb_email' in cb_data.columns else 0,
        'cb_with_location': int(cb_data['cb_hq_location'].notna().sum()) if 'cb_hq_location' in cb_data.columns else 0,
    }
    
    # Percentages
    if len(cb_data) > 0:
        stats['data_availability_pct'] = {
            'website_pct': round(stats['data_availability']['cb_with_website'] / len(cb_data) * 100, 1),
            'email_pct': round(stats['data_availability']['cb_with_email'] / len(cb_data) * 100, 1),
        }
    
    return stats


# =============================================================================
# COUNTRY/INDUSTRY BREAKDOWN
# =============================================================================

def compute_breakdown_by_dimension(
    matches: pd.DataFrame,
    cb_data: pd.DataFrame,
    dimension: str = 'country',
) -> pd.DataFrame:
    """
    Compute match statistics by country or industry.
    
    Useful for:
    - Checking geographic coverage
    - Identifying problem areas
    - Robustness by subsample
    """
    # Merge CB data for dimension
    if dimension == 'country':
        dim_col = 'cb_country_iso'
    elif dimension == 'industry':
        dim_col = 'cb_industries'
    else:
        dim_col = dimension
    
    if dim_col not in cb_data.columns:
        return pd.DataFrame()
    
    # Merge
    merged = matches.merge(
        cb_data[['cb_id', dim_col]].drop_duplicates(),
        on='cb_id',
        how='left'
    )
    
    # Group and aggregate
    grouped = merged.groupby(dim_col).agg({
        'cb_id': 'count',
        'p_match': ['mean', 'median'],
        'tier': lambda x: (x == 'A').sum(),  # Count Tier A
    }).reset_index()
    
    # Flatten columns
    grouped.columns = [dim_col, 'n_matches', 'avg_confidence', 'median_confidence', 'n_tier_a']
    
    # Add tier A percentage
    grouped['tier_a_pct'] = (grouped['n_tier_a'] / grouped['n_matches'] * 100).round(1)
    
    # Sort by count
    grouped = grouped.sort_values('n_matches', ascending=False)
    
    return grouped


# =============================================================================
# VALIDATION AGAINST PLATINUM MATCHES
# =============================================================================

def validate_against_platinum(
    our_matches: pd.DataFrame,
    db_done_path: str,
) -> Dict:
    """
    Validate our matches against the platinum matches from database-done.xlsx.
    
    This is CRITICAL for research credibility:
    - How many platinum matches did we find?
    - What tier did we assign them?
    - Which did we miss and why?
    """
    # Load platinum matches
    xls = pd.ExcelFile(db_done_path)
    platinum = pd.read_excel(xls, sheet_name=2)  # Matching 1 platinum
    
    logger.info(f"Loaded {len(platinum)} platinum matches for validation")
    
    # Normalize names for matching
    def norm(s):
        if pd.isna(s):
            return ''
        return str(s).lower().strip()
    
    platinum['_cb_name'] = platinum['nome_df2_match'].apply(norm)
    platinum['_orbis_name'] = platinum['nome_df1'].apply(norm)
    
    # Build set of platinum pairs
    platinum_pairs = set()
    for _, row in platinum.iterrows():
        if row['_cb_name'] and row['_orbis_name']:
            platinum_pairs.add((row['_cb_name'], row['_orbis_name']))
    
    logger.info(f"Created {len(platinum_pairs)} platinum pair keys")
    
    # Check our matches
    if 'cb_name' in our_matches.columns and 'orbis_name' in our_matches.columns:
        our_matches['_cb_name'] = our_matches['cb_name'].apply(norm)
        our_matches['_orbis_name'] = our_matches['orbis_name'].apply(norm)
        
        matched_platinum = 0
        missed_platinum = []
        tier_of_platinum = {'A': 0, 'B': 0, 'C': 0, 'Reject': 0}
        
        for cb_name, orbis_name in platinum_pairs:
            # Check if we found this pair
            found = our_matches[
                (our_matches['_cb_name'] == cb_name) |
                (our_matches['_orbis_name'] == orbis_name)
            ]
            
            if len(found) > 0:
                matched_platinum += 1
                tier = found.iloc[0].get('tier', 'Unknown')
                if tier in tier_of_platinum:
                    tier_of_platinum[tier] += 1
            else:
                missed_platinum.append((cb_name, orbis_name))
        
        validation = {
            'platinum_total': len(platinum_pairs),
            'platinum_found': matched_platinum,
            'platinum_recall': round(matched_platinum / len(platinum_pairs) * 100, 1) if platinum_pairs else 0,
            'platinum_by_tier': tier_of_platinum,
            'platinum_missed_count': len(missed_platinum),
            'platinum_missed_sample': missed_platinum[:20],  # First 20 for inspection
        }
    else:
        validation = {
            'error': 'Missing cb_name or orbis_name columns in matches',
        }
    
    return validation


# =============================================================================
# RESEARCH-READY EXPORT
# =============================================================================

def export_for_stata(
    matches: pd.DataFrame,
    cb_data: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Export matches in Stata-friendly format.
    
    Creates:
    - Wide format with one row per CB company
    - Numeric tiers for easy filtering
    - Key variables for merge
    """
    # Merge with CB data
    export = matches.merge(cb_data, on='cb_id', how='left')
    
    # Create numeric tier
    tier_map = {'A': 1, 'B': 2, 'C': 3, 'Reject': 4}
    export['tier_num'] = export['tier'].map(tier_map)
    
    # Select key columns
    key_cols = [
        'cb_id', 'bvd_id', 'p_match', 'tier', 'tier_num',
        'cb_name', 'cb_website', 'cb_country_iso',
        'orbis_name', 'orbis_country',
    ]
    
    export_cols = [c for c in key_cols if c in export.columns]
    export = export[export_cols]
    
    # Save
    export.to_csv(output_path, index=False)
    logger.info(f"Exported {len(export)} rows to {output_path}")
    
    return output_path


def generate_methodology_snippet(stats: Dict) -> str:
    """
    Generate methodology text for paper appendix.
    
    Returns markdown text ready to paste into paper.
    """
    snippet = f"""
## Data Matching Methodology

We link Crunchbase startup records to Bureau van Dijk's Orbis corporate database 
using a multi-signal matching approach. Our methodology proceeds as follows:

### Matching Signals

1. **Domain matching**: Exact match on website domain (strongest signal)
2. **Name similarity**: Jaro-Winkler similarity with threshold ≥ 0.7
3. **Geographic constraints**: Country ISO code matching
4. **Temporal compatibility**: Founded year vs. incorporation year (±5 years)
5. **Corporate structure**: Orbis GUO/subsidiary relationships

### Sample Construction

| Dataset | N |
|---------|---|
| Crunchbase companies | {stats['sample_sizes']['crunchbase_total']:,} |
| Orbis companies | {stats['sample_sizes']['orbis_total']:,} |
| Matched pairs | {stats['sample_sizes']['matched_total']:,} |
| Match rate | {stats['sample_sizes']['match_rate']}% |

### Confidence Tiers

We assign matches to confidence tiers based on model probability and signal strength:

| Tier | Criteria | N | % |
|------|----------|---|---|
| A (Auto-accept) | p ≥ 0.98 or domain+country+name | {stats['tier_distribution']['tier_A']:,} | {stats['tier_percentages']['tier_A_pct']}% |
| B (High confidence) | 0.93 ≤ p < 0.98 | {stats['tier_distribution']['tier_B']:,} | {stats['tier_percentages']['tier_B_pct']}% |
| C (Manual review) | 0.75 ≤ p < 0.93 | {stats['tier_distribution']['tier_C']:,} | {stats['tier_percentages']['tier_C_pct']}% |

### Robustness

We validate our matching against {stats.get('validation', {}).get('platinum_total', 'N')} 
manually verified matches, achieving a recall of {stats.get('validation', {}).get('platinum_recall', 'N')}%.
"""
    
    return snippet.strip()


# =============================================================================
# FULL ANALYTICS PIPELINE
# =============================================================================

def run_full_analytics(
    matches_path: str,
    cb_data_path: str,
    orbis_data_path: str,
    db_done_path: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    Run complete analytics pipeline for research.
    
    Generates all outputs needed for paper appendix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Load data
    logger.info("Loading data for analytics...")
    matches = pd.read_parquet(matches_path)
    cb_data = pd.read_parquet(cb_data_path)
    orbis_data = pd.read_parquet(orbis_data_path)
    
    # Merge names for validation (if not already present)
    if 'cb_name' not in matches.columns and 'cb_id' in matches.columns:
        cb_names = cb_data[['cb_id', 'cb_name']].drop_duplicates('cb_id')
        matches = matches.merge(cb_names, on='cb_id', how='left')
        
    if 'orbis_name' not in matches.columns and 'bvd_id' in matches.columns:
        orbis_names = orbis_data[['bvd_id', 'orbis_name']].drop_duplicates('bvd_id')
        matches = matches.merge(orbis_names, on='bvd_id', how='left')
    
    # 1. Core statistics
    logger.info("Computing research statistics...")
    stats = compute_research_statistics(matches, cb_data, orbis_data)
    
    # 2. Validation against platinum
    if Path(db_done_path).exists():
        logger.info("Validating against platinum matches...")
        validation = validate_against_platinum(matches, db_done_path)
        stats['validation'] = validation
    
    # Save stats
    stats_path = output_dir / 'research_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    outputs['stats'] = str(stats_path)
    
    # 3. Country breakdown
    logger.info("Computing country breakdown...")
    country_breakdown = compute_breakdown_by_dimension(matches, cb_data, 'country')
    country_path = output_dir / 'match_by_country.csv'
    country_breakdown.to_csv(country_path, index=False)
    outputs['country_breakdown'] = str(country_path)
    
    # 4. Stata export
    logger.info("Exporting for Stata...")
    stata_path = output_dir / 'matches_stata.csv'
    export_for_stata(matches, cb_data, str(stata_path))
    outputs['stata_export'] = str(stata_path)
    
    # 5. Methodology snippet
    logger.info("Generating methodology text...")
    snippet = generate_methodology_snippet(stats)
    snippet_path = output_dir / 'methodology_snippet.md'
    with open(snippet_path, 'w') as f:
        f.write(snippet)
    outputs['methodology'] = str(snippet_path)
    
    logger.info(f"Analytics complete. Outputs in {output_dir}")
    
    return outputs


# =============================================================================
# ABLATION STUDY FRAMEWORK (PHD REQUIREMENT)
# =============================================================================

ABLATION_CONFIGS = {
    'string_only': {
        'features': ['name_jw', 'name_token_jaccard', 'name_rapidfuzz_ratio', 'name_prefix_match'],
        'description': 'String similarity features only',
    },
    'string_domain': {
        'features': ['name_jw', 'name_token_jaccard', 'domain_exact', 'domain_in_family'],
        'description': 'String + Domain matching',
    },
    'string_domain_geo': {
        'features': ['name_jw', 'name_token_jaccard', 'domain_exact', 'country_match', 'city_sim'],
        'description': 'String + Domain + Geographic',
    },
    'full_no_family': {
        'features': [
            'name_jw', 'name_token_jaccard', 'domain_exact', 'country_match',
            'city_sim', 'year_compat', 'disambiguation_score',
        ],
        'description': 'All features except family expansion',
    },
    'full_no_investors': {
        'features': [
            'name_jw', 'name_token_jaccard', 'domain_exact', 'country_match',
            'is_guo', 'family_size_log',
        ],
        'description': 'All features except investor overlap',
    },
    'full_model': {
        'features': None,  # Use all available
        'description': 'Full model with all features',
    },
}


def run_ablation_study(
    features_df: pd.DataFrame,
    labels: pd.Series,
    model_type: str = 'gradient_boosting',
    configs: Dict = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run ablation study to measure marginal contribution of feature groups.
    
    PHD REQUIREMENT: Proves that each feature group provides measurable value.
    
    Args:
        features_df: Full feature DataFrame
        labels: Ground truth labels
        model_type: Model type for training
        configs: Ablation configs (default: ABLATION_CONFIGS)
        random_state: Random seed
    
    Returns:
        DataFrame with metrics for each ablation configuration
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    if configs is None:
        configs = ABLATION_CONFIGS
    
    results = []
    
    for config_name, config in configs.items():
        logger.info(f"Running ablation: {config_name}")
        
        # Get features
        if config['features'] is None:
            # Use all numeric features
            feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if not c.endswith('_id')]
        else:
            feature_cols = [f for f in config['features'] if f in features_df.columns]
        
        if len(feature_cols) == 0:
            logger.warning(f"  No features available for {config_name}, skipping")
            continue
        
        # Prepare data
        X = features_df[feature_cols].fillna(0)
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=random_state
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # Cross-validation
        cv_auc = cross_val_score(model, X_scaled, labels, cv=5, scoring='roc_auc')
        cv_f1 = cross_val_score(model, X_scaled, labels, cv=5, scoring='f1')
        cv_precision = cross_val_score(model, X_scaled, labels, cv=5, scoring='precision')
        cv_recall = cross_val_score(model, X_scaled, labels, cv=5, scoring='recall')
        
        results.append({
            'config': config_name,
            'description': config['description'],
            'n_features': len(feature_cols),
            'auc_mean': round(cv_auc.mean(), 4),
            'auc_std': round(cv_auc.std(), 4),
            'f1_mean': round(cv_f1.mean(), 4),
            'f1_std': round(cv_f1.std(), 4),
            'precision_mean': round(cv_precision.mean(), 4),
            'recall_mean': round(cv_recall.mean(), 4),
            'features': ', '.join(feature_cols[:5]) + ('...' if len(feature_cols) > 5 else ''),
        })
        
        logger.info(f"  AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    
    return pd.DataFrame(results)


def generate_ablation_table(ablation_results: pd.DataFrame, format: str = 'markdown') -> str:
    """
    Generate publication-ready ablation table.
    
    Args:
        ablation_results: Results from run_ablation_study
        format: 'markdown', 'latex', or 'html'
    
    Returns:
        Formatted table string
    """
    if format == 'latex':
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Ablation Study: Feature Group Contributions}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Configuration & Features & AUC & F1 & Precision \\",
            r"\hline",
        ]
        
        for _, row in ablation_results.iterrows():
            lines.append(
                f"{row['description']} & {row['n_features']} & "
                f"{row['auc_mean']:.3f} & {row['f1_mean']:.3f} & {row['precision_mean']:.3f} \\\\"
            )
        
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return '\n'.join(lines)
    
    else:  # markdown
        lines = [
            "## Ablation Study Results",
            "",
            "| Configuration | N Features | AUC | F1 | Δ AUC |",
            "|---------------|------------|-----|----|----|",
        ]
        
        # Calculate delta from full model
        full_auc = ablation_results[ablation_results['config'] == 'full_model']['auc_mean'].values
        full_auc = full_auc[0] if len(full_auc) > 0 else 1.0
        
        for _, row in ablation_results.iterrows():
            delta = row['auc_mean'] - full_auc
            delta_str = f"{delta:+.3f}" if row['config'] != 'full_model' else '-'
            lines.append(
                f"| {row['description']} | {row['n_features']} | "
                f"{row['auc_mean']:.3f} | {row['f1_mean']:.3f} | {delta_str} |"
            )
        
        return '\n'.join(lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Research Analytics module loaded.")
    print("Key functions:")
    print("  - compute_research_statistics(matches, cb, orbis)")
    print("  - validate_against_platinum(matches, db_done_path)")
    print("  - export_for_stata(matches, cb, output_path)")
    print("  - generate_methodology_snippet(stats)")
    print("  - run_full_analytics(...)")
    print("  - run_ablation_study(features_df, labels)  [NEW]")
    print("  - generate_ablation_table(results)  [NEW]")
