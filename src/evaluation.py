"""
evaluation.py - Quality Report Generation

Generates comprehensive quality reports for entity resolution results.
Includes precision/recall analysis, coverage breakdown, and error analysis.

OUTPUT:
-------
- quality_report.html: Interactive HTML report for review
- run_manifest.json: Versioning and reproducibility metadata
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
# METRICS COMPUTATION
# =============================================================================

def compute_tier_metrics(
    matches: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute precision/recall metrics per tier.
    
    Args:
        matches: DataFrame with tier, p_match columns
        labels: Optional labeled data with cb_id, bvd_id, label
    
    Returns:
        Dict with metrics per tier
    """
    metrics = {}
    
    # Coverage by tier
    tier_counts = matches['tier'].value_counts()
    total = len(matches)
    
    for tier in ['A', 'B', 'C', 'Reject']:
        count = tier_counts.get(tier, 0)
        metrics[f'count_{tier}'] = count
        metrics[f'pct_{tier}'] = 100 * count / total if total > 0 else 0
    
    # If labels available, compute precision/recall
    if labels is not None and len(labels) > 0:
        labeled_matches = matches.merge(
            labels[['cb_id', 'bvd_id', 'label']], 
            on=['cb_id', 'bvd_id'], 
            how='inner'
        )
        
        if len(labeled_matches) > 0:
            for tier in ['A', 'B', 'C']:
                tier_data = labeled_matches[labeled_matches['tier'] == tier]
                if len(tier_data) > 0:
                    tp = (tier_data['label'] == 1).sum()
                    fp = (tier_data['label'] == 0).sum()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    metrics[f'precision_{tier}'] = precision
                    metrics[f'n_labeled_{tier}'] = len(tier_data)
    
    return metrics


def compute_coverage_breakdown(
    matches: pd.DataFrame,
    cb_data: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute coverage breakdown by country, industry, etc.
    
    Args:
        matches: Match results
        cb_data: Crunchbase data for breakdown dimensions
    
    Returns:
        Dict with coverage statistics
    """
    breakdown = {}
    
    # Basic coverage
    breakdown['total_matches'] = len(matches)
    breakdown['matches_tier_a'] = (matches['tier'] == 'A').sum()
    breakdown['matches_tier_b'] = (matches['tier'] == 'B').sum()
    breakdown['matches_tier_c'] = (matches['tier'] == 'C').sum()
    
    if 'has_conflict' in matches.columns:
        breakdown['conflicts'] = matches['has_conflict'].sum()
    
    if 'match_type' in matches.columns:
        for mt in matches['match_type'].unique():
            breakdown[f'match_type_{mt}'] = (matches['match_type'] == mt).sum()
    
    # Country breakdown if CB data available
    if cb_data is not None:
        merged = matches.merge(
            cb_data[['cb_id', 'cb_country_iso']], 
            on='cb_id', 
            how='left'
        )
        
        country_coverage = merged.groupby('cb_country_iso').agg({
            'cb_id': 'count',
            'p_match': 'mean',
        }).rename(columns={'cb_id': 'count', 'p_match': 'avg_confidence'})
        
        breakdown['by_country'] = country_coverage.to_dict('index')
    
    return breakdown


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def analyze_errors(
    matches: pd.DataFrame,
    labels: pd.DataFrame,
    n_examples: int = 20,
) -> Dict:
    """
    Analyze false positives and false negatives.
    
    Args:
        matches: Match results with features
        labels: Ground truth labels
        n_examples: Number of examples per error type
    
    Returns:
        Dict with FP/FN analysis
    """
    analysis = {
        'false_positives': [],
        'false_negatives': [],
    }
    
    # Merge with labels
    labeled = matches.merge(
        labels[['cb_id', 'bvd_id', 'label']], 
        on=['cb_id', 'bvd_id'],
        how='inner'
    )
    
    if len(labeled) == 0:
        return analysis
    
    # False positives: high p_match but label=0
    fp = labeled[(labeled['p_match'] >= 0.75) & (labeled['label'] == 0)]
    fp_sorted = fp.sort_values('p_match', ascending=False)
    
    for _, row in fp_sorted.head(n_examples).iterrows():
        analysis['false_positives'].append({
            'cb_id': row['cb_id'],
            'bvd_id': row['bvd_id'],
            'p_match': float(row['p_match']),
            'tier': row.get('tier', ''),
            'domain_exact': bool(row.get('domain_exact', False)),
            'country_match': bool(row.get('country_match', False)),
            'name_jw': float(row.get('name_jw', 0)),
        })
    
    # False negatives: low p_match but label=1
    fn = labeled[(labeled['p_match'] < 0.75) & (labeled['label'] == 1)]
    fn_sorted = fn.sort_values('p_match', ascending=True)
    
    for _, row in fn_sorted.head(n_examples).iterrows():
        analysis['false_negatives'].append({
            'cb_id': row['cb_id'],
            'bvd_id': row['bvd_id'],
            'p_match': float(row['p_match']),
            'tier': row.get('tier', ''),
            'domain_exact': bool(row.get('domain_exact', False)),
            'country_match': bool(row.get('country_match', False)),
            'name_jw': float(row.get('name_jw', 0)),
        })
    
    return analysis


# =============================================================================
# SOTA SAMPLING STRATEGIES
# =============================================================================

def generate_stratified_test_set(
    candidates: pd.DataFrame,
    n_samples: int = 500,
    strata_col: str = 'tier',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a stratified test set for rigorous evaluation.
    
    Ensures representation across all tiers rather than random sampling
    (which would be dominated by easy negatives).
    
    Args:
        candidates: All candidate pairs with tier assignments
        n_samples: Target total samples
        strata_col: Column to stratify by
        random_state: Random seed
    
    Returns:
        Stratified sample DataFrame
    """
    strata = candidates[strata_col].unique()
    samples_per_stratum = max(1, n_samples // len(strata))
    
    sampled = []
    for stratum in strata:
        stratum_data = candidates[candidates[strata_col] == stratum]
        n_take = min(samples_per_stratum, len(stratum_data))
        sampled.append(stratum_data.sample(n=n_take, random_state=random_state))
    
    result = pd.concat(sampled, ignore_index=True)
    logger.info(f"Stratified sampling: {len(result)} samples across {len(strata)} strata")
    
    for stratum in strata:
        count = (result[strata_col] == stratum).sum()
        logger.info(f"  {stratum}: {count}")
    
    return result


def sample_hard_negatives(
    candidates: pd.DataFrame,
    n_samples: int = 100,
    name_sim_threshold: float = 0.7,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample hard negatives for challenging the model.
    
    Hard negatives = high string similarity but likely incorrect matches
    (e.g., same prefix but different country, or similar names in same industry).
    
    Args:
        candidates: Candidate pairs with features
        n_samples: Number of hard negatives to sample
        name_sim_threshold: Minimum name similarity to consider "hard"
        random_state: Random seed
    
    Returns:
        DataFrame of hard negative candidates for labeling
    """
    hard_mask = (
        (candidates.get('name_jw', pd.Series([0]*len(candidates))) >= name_sim_threshold) &
        (candidates.get('country_match', pd.Series([True]*len(candidates))) == False)
    )
    
    hard_candidates = candidates[hard_mask]
    
    if len(hard_candidates) == 0:
        # Fallback: just take high name similarity
        hard_mask = candidates.get('name_jw', pd.Series([0]*len(candidates))) >= name_sim_threshold
        hard_candidates = candidates[hard_mask]
    
    n_take = min(n_samples, len(hard_candidates))
    
    if n_take == 0:
        logger.warning("No hard negatives found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    result = hard_candidates.sample(n=n_take, random_state=random_state)
    logger.info(f"Sampled {len(result)} hard negatives (name_sim >= {name_sim_threshold})")
    
    return result


def generate_comprehensive_test_set(
    candidates: pd.DataFrame,
    n_stratified: int = 400,
    n_hard_negatives: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a comprehensive test set combining stratified and hard negative samples.
    
    This creates a balanced, challenging test set suitable for rigorous evaluation.
    
    Args:
        candidates: All candidate pairs with features
        n_stratified: Number of stratified samples
        n_hard_negatives: Number of additional hard negatives
        random_state: Random seed
    
    Returns:
        Combined test set DataFrame
    """
    stratified = generate_stratified_test_set(
        candidates, n_samples=n_stratified, random_state=random_state
    )
    
    hard_negs = sample_hard_negatives(
        candidates, n_samples=n_hard_negatives, random_state=random_state
    )
    
    # Combine, removing duplicates
    combined = pd.concat([stratified, hard_negs], ignore_index=True)
    combined = combined.drop_duplicates(subset=['cb_id', 'bvd_id'], keep='first')
    
    logger.info(f"Comprehensive test set: {len(combined)} unique pairs")
    
    return combined


# =============================================================================
# ERROR BOUND COMPUTATION (WILSON SCORE INTERVALS)
# =============================================================================

def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.
    
    More accurate than normal approximation, especially for small samples
    or extreme proportions.
    
    Args:
        successes: Number of successes (e.g., correct predictions)
        total: Total number of trials
        confidence: Confidence level (default 95%)
    
    Returns:
        (lower_bound, upper_bound) of the confidence interval
    """
    from scipy import stats
    
    if total == 0:
        return (0.0, 1.0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return (lower, upper)


def compute_error_bounds(
    matches: pd.DataFrame,
    labels: pd.DataFrame,
    confidence: float = 0.95,
) -> Dict:
    """
    Compute error rate bounds with confidence intervals per tier.
    
    Critical for research-grade claims like:
    "With 95% confidence, Tier A error rate is below X%"
    
    Args:
        matches: Match results with tier column
        labels: Labeled data with cb_id, bvd_id, label columns
        confidence: Confidence level
    
    Returns:
        Dict with error bounds per tier
    """
    labeled = matches.merge(
        labels[['cb_id', 'bvd_id', 'label']],
        on=['cb_id', 'bvd_id'],
        how='inner'
    )
    
    if len(labeled) == 0:
        return {}
    
    bounds = {}
    
    for tier in ['A', 'B', 'C']:
        tier_data = labeled[labeled['tier'] == tier]
        if len(tier_data) == 0:
            continue
        
        n_correct = (tier_data['label'] == 1).sum()
        n_total = len(tier_data)
        n_errors = n_total - n_correct
        
        # Error rate bounds
        error_rate = n_errors / n_total
        lower, upper = wilson_confidence_interval(n_errors, n_total, confidence)
        
        bounds[f'tier_{tier}'] = {
            'n_labeled': n_total,
            'n_correct': n_correct,
            'n_errors': n_errors,
            'error_rate': error_rate,
            'error_rate_lower': lower,
            'error_rate_upper': upper,
            'confidence': confidence,
            'claim': f"With {confidence*100:.0f}% confidence, Tier {tier} error rate is below {upper*100:.2f}%"
        }
        
        logger.info(f"Tier {tier}: error rate = {error_rate*100:.2f}% "
                   f"(95% CI: [{lower*100:.2f}%, {upper*100:.2f}%])")
    
    return bounds

def generate_html_report(
    metrics: Dict,
    breakdown: Dict,
    error_analysis: Optional[Dict] = None,
    output_path: str = 'quality_report.html',
) -> str:
    """
    Generate an HTML quality report.
    
    Args:
        metrics: Tier metrics
        breakdown: Coverage breakdown
        error_analysis: FP/FN analysis
        output_path: Output file path
    
    Returns:
        Path to generated report
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Entity Resolution Quality Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                     padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 20px; border-radius: 8px; color: white; text-align: center; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; }}
        .metric-label {{ opacity: 0.9; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .tier-A {{ color: #10b981; font-weight: bold; }}
        .tier-B {{ color: #3b82f6; font-weight: bold; }}
        .tier-C {{ color: #f59e0b; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Entity Resolution Quality Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Tier Distribution</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('count_A', 0):,}</div>
                <div class="metric-label">Tier A (Auto-accept)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('count_B', 0):,}</div>
                <div class="metric-label">Tier B (High conf)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('count_C', 0):,}</div>
                <div class="metric-label">Tier C (Review)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{breakdown.get('conflicts', 0):,}</div>
                <div class="metric-label">Conflicts</div>
            </div>
        </div>
        
        <h2>📈 Coverage Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Matches</td><td>{breakdown.get('total_matches', 0):,}</td></tr>
            <tr><td>Tier A %</td><td>{metrics.get('pct_A', 0):.1f}%</td></tr>
            <tr><td>Tier B %</td><td>{metrics.get('pct_B', 0):.1f}%</td></tr>
            <tr><td>Tier C %</td><td>{metrics.get('pct_C', 0):.1f}%</td></tr>
        </table>
        
        <h2>🏢 Match Types</h2>
        <table>
            <tr><th>Type</th><th>Count</th></tr>
"""
    
    # Add match type rows
    for key, val in breakdown.items():
        if key.startswith('match_type_'):
            mt = key.replace('match_type_', '')
            html += f"            <tr><td>{mt}</td><td>{val:,}</td></tr>\n"
    
    html += """        </table>
        
        <h2>✅ Precision by Tier (if labels available)</h2>
        <table>
            <tr><th>Tier</th><th>Labeled Samples</th><th>Precision</th></tr>
"""
    
    for tier in ['A', 'B', 'C']:
        n = metrics.get(f'n_labeled_{tier}', 0)
        p = metrics.get(f'precision_{tier}', 'N/A')
        p_str = f"{p*100:.1f}%" if isinstance(p, float) else p
        html += f"            <tr><td class='tier-{tier}'>{tier}</td><td>{n}</td><td>{p_str}</td></tr>\n"
    
    html += """        </table>
    </div>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")
    return output_path


# =============================================================================
# RUN MANIFEST
# =============================================================================

def generate_run_manifest(
    config: Dict,
    metrics: Dict,
    breakdown: Dict,
    output_path: str = 'run_manifest.json',
) -> str:
    """
    Generate run manifest for versioning and reproducibility.
    
    Args:
        config: Pipeline configuration
        metrics: Computed metrics
        breakdown: Coverage breakdown
        output_path: Output file path
    
    Returns:
        Path to manifest file
    """
    import hashlib
    
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'pipeline_version': '1.0.0',
        
        # Config summary
        'config': {
            'tier_thresholds': config.get('tiers', {}),
            'max_candidates_per_cb': config.get('blocking', {}).get('max_candidates_per_cb', 300),
            'model_type': config.get('model', {}).get('type', 'gradient_boosting'),
        },
        
        # Results summary
        'results': {
            'total_matches': breakdown.get('total_matches', 0),
            'tier_A': metrics.get('count_A', 0),
            'tier_B': metrics.get('count_B', 0),
            'tier_C': metrics.get('count_C', 0),
            'conflicts': breakdown.get('conflicts', 0),
        },
        
        # Metrics
        'metrics': {
            'pct_tier_A': metrics.get('pct_A', 0),
            'pct_tier_B': metrics.get('pct_B', 0),
            'pct_tier_C': metrics.get('pct_C', 0),
        },
    }
    
    # Add precision if available
    for tier in ['A', 'B', 'C']:
        if f'precision_{tier}' in metrics:
            manifest['metrics'][f'precision_{tier}'] = metrics[f'precision_{tier}']
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Generated run manifest: {output_path}")
    return output_path


# =============================================================================
# MAIN REPORT GENERATION
# =============================================================================

def generate_full_report(
    matches_path: str,
    output_dir: str,
    cb_data_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Generate complete quality report package.
    
    Args:
        matches_path: Path to matches_final.parquet
        output_dir: Output directory
        cb_data_path: Optional path to CB clean data
        labels_path: Optional path to labeled data
        config: Pipeline configuration
    
    Returns:
        Dict of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    matches = pd.read_parquet(matches_path)
    
    cb_data = None
    if cb_data_path and Path(cb_data_path).exists():
        cb_data = pd.read_parquet(cb_data_path)
    
    labels = None
    if labels_path and Path(labels_path).exists():
        labels = pd.read_parquet(labels_path)
    
    # Compute metrics
    metrics = compute_tier_metrics(matches, labels)
    breakdown = compute_coverage_breakdown(matches, cb_data)
    
    error_analysis = None
    if labels is not None:
        error_analysis = analyze_errors(matches, labels)
    
    # Generate outputs
    outputs = {}
    
    # HTML report
    html_path = output_dir / 'quality_report.html'
    outputs['html_report'] = generate_html_report(
        metrics, breakdown, error_analysis, str(html_path)
    )
    
    # Run manifest
    manifest_path = output_dir / 'run_manifest.json'
    outputs['manifest'] = generate_run_manifest(
        config or {}, metrics, breakdown, str(manifest_path)
    )
    
    # Error analysis JSON
    if error_analysis:
        error_path = output_dir / 'error_analysis.json'
        with open(error_path, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        outputs['error_analysis'] = str(error_path)
    
    logger.info(f"Generated {len(outputs)} report files in {output_dir}")
    return outputs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Evaluation module loaded.")
    print("Key functions:")
    print("  - compute_tier_metrics(matches, labels)")
    print("  - compute_coverage_breakdown(matches, cb_data)")
    print("  - generate_html_report(metrics, breakdown)")
    print("  - generate_full_report(matches_path, output_dir)")
