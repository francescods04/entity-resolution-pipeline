#!/usr/bin/env python3
"""
run_pipeline.py - Master Pipeline Orchestration

Executes the entity resolution pipeline incrementally.
Can start with current Excel files and process more as they arrive.

USAGE:
------
# Run full pipeline
python run_pipeline.py --config configs/colab_gpu.yaml

# Run specific step
python run_pipeline.py --step blocking --config configs/base.yaml

# Process only current Orbis files (don't wait for all)
python run_pipeline.py --step all --incremental

STEPS:
------
1. ingest     - Load and convert data to Parquet
2. normalize  - Apply normalization (names, domains, geo)
3. alias      - Build alias registry from database-done.xlsx
4. index      - Build blocking indexes for Orbis
5. blocking   - Generate candidate pairs
6. features   - Compute pair features
7. train      - Train matching model
8. score      - Score all candidates
9. decide     - Final match decisions + tiers
10. report    - Generate quality report
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config, get_project_paths, ensure_directories


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config, paths: Dict) -> tuple:
    """Setup structured logging with RunContext."""
    from logging_config import configure_global_logging, RunContext
    
    # Create run context for correlation
    ctx = RunContext()
    
    # Log file path
    log_dir = paths.get('reports', Path('logs'))
    if hasattr(log_dir, 'mkdir'):
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f'run_{ctx.run_id}.log')
    
    # Configure ROOT logger
    logger = configure_global_logging(
        context=ctx,
        level=logging.INFO,
        json_output=False,  # Human-readable for console
        log_file=log_file,  # JSON for parsing
    )
    
    return logger, ctx


# =============================================================================
# STEP IMPLEMENTATIONS
# =============================================================================

def step_ingest(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 1: Data ingestion and Parquet conversion."""
    from data_io import ingest_all_data
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)
    
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION (Optimized)")
    logger.info("=" * 60)
    
    try:
        # Check if Orbis parquet already exists to avoid re-processing 900+ Excel files
        orbis_output = paths['interim'] / 'orbis_clean' / 'orbis_raw.parquet'
        skip_orbis = orbis_output.exists() and orbis_output.stat().st_size > 100_000_000 # > 100MB
        
        if skip_orbis:
            logger.info(f"Found existing Orbis data ({orbis_output.stat().st_size / 1e6:.1f} MB). Skipping Excel conversion.")
        
        # Use the robust data_io module which handles the specific folder structure
        ingest_all_data(
            crunchbase_dir=str(paths['raw_crunchbase']),
            orbis_dir=str(paths['raw_orbis']),
            output_dir=str(paths['interim']),
            skip_orbis=skip_orbis
        )
        logger.info("Ingestion complete")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


def step_normalize(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 2: Apply normalization to all data."""
    import pandas as pd
    from normalize import normalize_name_column
    from domains import extract_domain_column
    from geo import parse_location_column
    
    logger.info("=" * 60)
    logger.info("STEP 2: NORMALIZATION")
    logger.info("=" * 60)
    
    # Normalize Crunchbase
    # Normalize Crunchbase
    cb_path = paths['cb_clean'] / 'cb_raw_companies.parquet' # Use output from data_io
    if cb_path.exists():
        logger.info(f"Normalizing Crunchbase data from {cb_path}...")
        cb = pd.read_parquet(cb_path)
        
        # Name normalization
        name_features = normalize_name_column(cb['cb_name'])
        for col in name_features.columns:
            cb[f'cb_{col}'] = name_features[col]
        
        # Domain extraction
        domain_features = extract_domain_column(cb['cb_website'])
        cb['cb_domain'] = domain_features['etld1']
        cb['cb_is_free_email'] = extract_domain_column(cb['cb_email'])['is_free_email']
        
        # Location parsing
        geo_features = parse_location_column(cb['cb_hq_location'])
        cb['cb_country_iso'] = geo_features['country_iso']
        cb['cb_city_norm'] = geo_features['city_norm']
        
        # Extract year
        cb['cb_founded_year'] = pd.to_datetime(cb['cb_founded_date'], errors='coerce').dt.year
        
        # Save
        output_path = paths['cb_clean'] / 'cb_clean.parquet'
        cb.to_parquet(output_path, index=False)
        logger.info(f"Saved normalized CB data: {len(cb)} rows")
    
    # Normalize Orbis (if available)
    orbis_path = paths['orbis_clean'] / 'orbis_raw.parquet'
    if orbis_path.exists():
        logger.info(f"Normalizing Orbis data from {orbis_path}...")
        orbis = pd.read_parquet(orbis_path)
        
        # Name normalization
        logger.info("  Normalizing names...")
        name_features = normalize_name_column(orbis['orbis_name'])
        for col in name_features.columns:
            orbis[f'orbis_{col}'] = name_features[col]
        
        # =============================================================
        # VECTORIZED Domain extraction (handle multiple websites via '|')
        # This replaces the slow apply() that took 10+ minutes
        # =============================================================
        logger.info("  Extracting domains (vectorized with progress)...")
        # Take first website from pipe-separated list
        first_websites = orbis['orbis_website'].fillna('').str.split('|').str[0]
        domain_features = extract_domain_column(first_websites)
        orbis['orbis_domain'] = domain_features['etld1']
        
        # Also extract ALL domains for broader matching (pipe-separated)
        # Use chunked processing with tqdm
        logger.info("  Extracting all domains...")
        from tqdm import tqdm
        chunk_size = 100000
        n_rows = len(orbis)
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        all_domains_results = []
        
        for i in tqdm(range(0, n_rows, chunk_size), 
                      total=n_chunks, 
                      desc="Extracting all domains",
                      unit="chunk"):
            chunk = orbis['orbis_website'].iloc[i:i + chunk_size]
            chunk_results = chunk.apply(lambda website_str: 
                '' if pd.isna(website_str) or not website_str else
                '|'.join([w.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
                          for w in str(website_str).split('|') if w.strip()])
            )
            all_domains_results.append(chunk_results)
        
        orbis['orbis_all_domains'] = pd.concat(all_domains_results, ignore_index=True)
        
        # Email domain extraction (vectorized - fast already)
        logger.info("  Extracting email domains (vectorized)...")
        first_emails = orbis['orbis_email'].fillna('').str.split('|').str[0]
        # Extract domain from email: user@domain.com -> domain.com
        orbis['orbis_email_domain'] = first_emails.str.extract(r'@([^\s]+)', expand=False).fillna('')
        
        # =============================================================
        # CORPORATE STRUCTURE: Extract multi-value IDs for family matching
        # =============================================================
        logger.info("  Processing corporate structure fields...")
        
        # Determine entity role from structure fields (VECTORIZED)
        # Using np.select for vectorized conditional logic
        import numpy as np
        conditions = [
            (orbis['guo_bvd_id'].notna()) & (orbis['guo_bvd_id'] == orbis['bvd_id']),
            (orbis['branch_bvd_id'].notna()) & (orbis['branch_bvd_id'] != '<NA>'),
            (orbis['sub_bvd_id'].notna()) & (orbis['sub_bvd_id'] != '<NA>'),
        ]
        choices = ['GUO', 'BRANCH', 'SUBSIDIARY']
        orbis['entity_role'] = np.select(conditions, choices, default='LEGAL_ENTITY')
        
        # Count subsidiaries and shareholders for family_size (VECTORIZED)
        sub_count = orbis['sub_bvd_id'].fillna('').str.count(r'\|') + 1
        sh_count = orbis['sh_bvd_id'].fillna('').str.count(r'\|') + 1
        orbis['family_size'] = sub_count + sh_count
        orbis.loc[orbis['sub_bvd_id'].isna() | (orbis['sub_bvd_id'] == '<NA>'), 'family_size'] = 1
        
        # Year extraction (handle Excel serial dates) - FULLY VECTORIZED (FAST!)
        logger.info("  Extracting incorporation year (vectorized)...")
        
        # Convert to string for processing
        date_col = orbis['orbis_incorp_date'].astype(str)
        
        # Detect numeric Excel serial dates (e.g., "44562" for 2022-01-01)
        is_numeric = date_col.str.match(r'^\d+$', na=False)
        
        # Convert Excel serial dates: Excel epoch is 1899-12-30
        import numpy as np
        numeric_dates = pd.to_datetime('1899-12-30') + pd.to_timedelta(
            pd.to_numeric(date_col.where(is_numeric), errors='coerce'), unit='D'
        )
        
        # Parse non-numeric dates normally (much faster than apply!)
        parsed_dates = pd.to_datetime(date_col.where(~is_numeric), errors='coerce', dayfirst=True)
        
        # Combine: use numeric where available, else parsed
        combined_dates = numeric_dates.fillna(parsed_dates)
        
        # Extract year
        orbis['orbis_incorp_year'] = combined_dates.dt.year
        
        # Save
        logger.info("  Saving normalized Orbis data...")
        output_path = paths['orbis_clean'] / 'orbis_clean.parquet'
        orbis.to_parquet(output_path, index=False)
        logger.info(f"Saved normalized Orbis data: {len(orbis)} rows")


def step_alias(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 3: Build alias registry from existing work."""
    from alias_registry import load_alias_registry_from_excel, save_registry_to_parquet
    
    logger.info("=" * 60)
    logger.info("STEP 3: ALIAS REGISTRY")
    logger.info("=" * 60)
    
    # Load from database-done.xlsx
    db_done_path = paths['project_root'].parent / 'database-done.xlsx'
    if db_done_path.exists():
        registry = load_alias_registry_from_excel(str(db_done_path))
        
        # Save registry
        output_path = paths['interim'] / 'alias_registry.parquet'
        save_registry_to_parquet(registry, str(output_path))
        
        stats = registry.get_stats()
        logger.info(f"Alias registry: {stats['total_cb_ids']} companies, {stats['total_aliases']} aliases")
    else:
        logger.warning(f"database-done.xlsx not found at {db_done_path}")


def step_index(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 4: Build blocking indexes for Orbis."""
    from blocking import build_blocking_indexes
    
    logger.info("=" * 60)
    logger.info("STEP 4: BUILD BLOCKING INDEXES")
    logger.info("=" * 60)
    
    orbis_path = paths['orbis_clean'] / 'orbis_clean.parquet'
    if orbis_path.exists():
        indexes = build_blocking_indexes(
            orbis_parquet_path=str(orbis_path),
            output_dir=str(paths['indexes']),
            chunk_size=config.get('processing.chunk_size', 50000)
        )
        logger.info(f"Built {len(indexes)} indexes")
    else:
        logger.warning("Orbis clean data not found, skipping index build")


def step_blocking(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 5: Generate candidate pairs."""
    import pandas as pd
    from blocking import BlockingIndex, generate_candidates
    from alias_registry import AliasRegistry, load_prematched_bvd_pairs, load_platinum_pairs
    
    logger.info("=" * 60)
    logger.info("STEP 5: CANDIDATE GENERATION (BLOCKING)")
    logger.info("=" * 60)
    
    # Load CB data
    cb_path = paths['cb_clean'] / 'cb_clean.parquet'
    cb_data = pd.read_parquet(cb_path)
    
    # Load Orbis data for ID lookups
    orbis_path = paths['orbis_clean'] / 'orbis_clean.parquet'
    orbis_data = pd.read_parquet(orbis_path, columns=['bvd_id', 'orbis_name'])
    
    # Load indexes
    indexes = {}
    for idx_name in ['domain', 'country_prefix', 'rare_token']:
        idx_path = paths['indexes'] / f'{idx_name}_index.parquet'
        if idx_path.exists():
            indexes[idx_name] = BlockingIndex.load(str(idx_path), idx_name)
    
    # Load Alias Registry
    alias_registry = None
    alias_path = paths['interim'] / 'alias_registry.parquet'
    if alias_path.exists():
        logger.info("Loading Alias Registry for query expansion...")
        try:
            alias_registry = AliasRegistry()
            alias_df = pd.read_parquet(alias_path)
            for _, row in alias_df.iterrows():
                names = row['names'].split('|') if row['names'] else []
                for name in names:
                    alias_registry.add_alias(row['cb_id'], name)
            logger.info(f"Loaded {len(alias_df)} companies into registry")
        except Exception as e:
            logger.warning(f"Failed to load alias registry: {e}")

    # =========================================================================
    # INJECT PRE-MATCHED PAIRS (HIGH CONFIDENCE - SKIP BLOCKING)
    # =========================================================================
    prematched_candidates = []
    db_done_path = paths['project_root'].parent / 'database-done.xlsx'
    
    if db_done_path.exists():
        # Load pre-matched BVD pairs (Sheet 1)
        try:
            bvd_pairs = load_prematched_bvd_pairs(str(db_done_path))
            
            # Map CB names to cb_id
            cb_name_to_id = cb_data.set_index('cb_name')['cb_id'].to_dict()
            
            # Convert to candidate format
            for _, row in bvd_pairs.iterrows():
                cb_id = cb_name_to_id.get(row['cb_name'])
                if cb_id and row['bvd_id'] in orbis_data['bvd_id'].values:
                    prematched_candidates.append({
                        'cb_id': cb_id,
                        'bvd_id': row['bvd_id'],
                        'blocking_source': 'prematched_bvdid',
                        'blocking_confidence': row['confidence']
                    })
            
            logger.info(f"Added {len(prematched_candidates)} pre-matched BVD pairs")
        except Exception as e:
            logger.warning(f"Failed to load pre-matched BVD pairs: {e}")
        
        # Load platinum pairs (Sheet 2) 
        try:
            platinum_pairs = load_platinum_pairs(str(db_done_path))
            
            # Map Orbis names to bvd_id
            orbis_name_to_id = orbis_data.set_index('orbis_name')['bvd_id'].to_dict()
            
            platinum_count = 0
            for _, row in platinum_pairs.iterrows():
                cb_id = cb_name_to_id.get(row['cb_name'])
                bvd_id = orbis_name_to_id.get(row['orbis_name'])
                
                if cb_id and bvd_id:
                    prematched_candidates.append({
                        'cb_id': cb_id,
                        'bvd_id': bvd_id,
                        'blocking_source': 'platinum_match',
                        'blocking_confidence': row['confidence']
                    })
                    platinum_count += 1
            
            logger.info(f"Added {platinum_count} platinum pairs")
        except Exception as e:
            logger.warning(f"Failed to load platinum pairs: {e}")
    
    # =========================================================================
    # REGULAR BLOCKING (VECTORIZED)
    # =========================================================================
    candidates = generate_candidates(
        cb_companies=cb_data,
        indexes=indexes,
        max_candidates_per_cb=config.get('blocking.max_candidates_per_cb', 300),
        alias_registry=alias_registry
    )
    
    # Merge pre-matched with blocking results
    if prematched_candidates:
        prematched_df = pd.DataFrame(prematched_candidates)
        
        # Remove duplicates (prefer pre-matched over blocking)
        candidates = candidates[~candidates.set_index(['cb_id', 'bvd_id']).index.isin(
            prematched_df.set_index(['cb_id', 'bvd_id']).index
        )]
        
        candidates = pd.concat([prematched_df, candidates], ignore_index=True)
        logger.info(f"Total candidates after merging: {len(candidates):,}")
    
    # Save
    output_path = paths['candidates'] / 'candidates.parquet'
    paths['candidates'].mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(candidates)} candidates")


def step_features(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 6: Compute pair features."""
    import pandas as pd
    from features import compute_features_batch
    from alias_registry import AliasRegistry
    
    logger.info("=" * 60)
    logger.info("STEP 6: FEATURE COMPUTATION")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading Crunchbase data...")
    cb_columns = ['cb_id', 'cb_name', 'cb_domain', 'cb_email_domain', 'cb_country_iso', 
                  'cb_city_norm', 'cb_founded_year', 'cb_is_free_email', 'cb_name_is_generic',
                  'cb_industries', 'cb_industry_groups', 'cb_description'] # Added desc cols for context if needed
    
    # Check available columns first to avoid errors
    # (Parquet doesn't let us peek easily without reading metadata, but pyarrow does. 
    #  For simplicity, we'll try to read all and prune, OR assume standard schema.
    #  Let's stick to standard schema but be safe.)
    
    cb_data = pd.read_parquet(paths['cb_clean'] / 'cb_clean.parquet')
    
    logger.info("Loading Orbis data (selected columns)...")
    orbis_columns = ['bvd_id', 'orbis_name', 'orbis_website', 'orbis_email', 
                     'orbis_country', 'orbis_city', 'orbis_incorp_year', 
                     'entity_role', 'family_size']
    
    # Load Orbis with pruning for memory safety
    try:
        orbis_data = pd.read_parquet(
            paths['orbis_clean'] / 'orbis_clean.parquet', 
            columns=orbis_columns
        )
    except Exception as e:
        logger.warning(f"Could not prune Orbis columns: {e}. Loading full.")
        orbis_data = pd.read_parquet(paths['orbis_clean'] / 'orbis_clean.parquet')
    
    # Load alias registry if available
    alias_registry = None
    alias_path = paths['interim'] / 'alias_registry.parquet'
    if alias_path.exists():
        # Load into registry object
        alias_registry = AliasRegistry()
        alias_df = pd.read_parquet(alias_path)
        for _, row in alias_df.iterrows():
            names = row['names'].split('|') if row['names'] else []
            for name in names:
                alias_registry.add_alias(row['cb_id'], name)
    
    # Load Embedding Resources (for feature computation)
    from features import load_embedding_resources
    embeddings_dir = paths['interim'] / 'embeddings'
    embedding_resources = load_embedding_resources(str(embeddings_dir))
    
    # Compute features in chunks to save memory (16GB RAM constraint)
    # 50k candidates ~ 200MB RAM, safe for laptop
    chunk_size = config.get('processing.feature_chunk_size', 50000) 
    
    features_dfs = []
    
    # Load candidates from blocking step
    candidates_path = paths['candidates'] / 'candidates.parquet'
    if not candidates_path.exists():
        logger.error(f"Candidates file not found at {candidates_path}. Run blocking step first.")
        return
    candidates = pd.read_parquet(candidates_path)
    
    logger.info(f"Computing features for {len(candidates)} candidates in chunks of {chunk_size}...")
    
    for i in range(0, len(candidates), chunk_size):
        chunk = candidates.iloc[i:i + chunk_size].copy()
        logger.info(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} pairs)...")
        
        chunk_features = compute_features_batch(
            candidates=chunk,
            cb_data=cb_data,
            orbis_data=orbis_data,
            alias_registry=alias_registry,
            embedding_resources=embedding_resources
        )
        features_dfs.append(chunk_features)
    
    features_df = pd.concat(features_dfs, ignore_index=True)
    
    # Save
    output_path = paths['features'] / 'pair_features.parquet'
    paths['features'].mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    logger.info(f"Saved features for {len(features_df)} pairs")


def step_train(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 7: Train matching model."""
    from modeling import train_from_platinum_matches
    from data_loader import load_platinum_matches
    
    logger.info("=" * 60)
    logger.info("STEP 7: MODEL TRAINING")
    logger.info("=" * 60)
    
    features_path = paths['features'] / 'pair_features.parquet'
    if features_path.exists():
        # Load platinum matches to use as GROUND TRUTH
        db_done_path = paths['project_root'].parent / 'database-done.xlsx'
        platinum_df = None
        
        if db_done_path.exists():
            logger.info("Loading platinum matches for supervised training...")
            platinum_df = load_platinum_matches(str(db_done_path))
        else:
            logger.warning("database-done.xlsx not found - training will use heuristics!")
            
        model_paths = train_from_platinum_matches(
            features_path=str(features_path),
            output_dir=str(paths['models'] / 'company_match'),
            platinum_df=platinum_df,
            model_type=config.get('model.type', 'gradient_boosting'),
            cb_data_path=str(paths['cb_clean'] / 'cb_clean.parquet'),
            orbis_data_path=str(paths['orbis_clean'] / 'orbis_clean.parquet'),
        )
        logger.info(f"Model saved to {model_paths['model']}")
    else:
        logger.warning("Features not found, skipping training")


def step_score(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 8: Score all candidates."""
    import pandas as pd
    from modeling import load_model, score_candidates
    
    logger.info("=" * 60)
    logger.info("STEP 8: SCORING")
    logger.info("=" * 60)
    
    # Load model
    model_path = paths['models'] / 'company_match' / 'company_match_model.joblib'
    if not model_path.exists():
        logger.warning("Model not found, skipping scoring")
        return
    
    model_package = load_model(str(model_path))
    
    # Load features
    features_df = pd.read_parquet(paths['features'] / 'pair_features.parquet')
    
    # Score
    scored_df = score_candidates(features_df, model_package)
    
    # Save
    output_path = paths['features'] / 'scored_candidates.parquet'
    scored_df.to_parquet(output_path, index=False)
    logger.info(f"Scored {len(scored_df)} candidates")


def step_decide(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 9: Final match decisions."""
    import pandas as pd
    from decisioning import generate_final_output
    
    logger.info("=" * 60)
    logger.info("STEP 9: DECISIONING")
    logger.info("=" * 60)
    
    scored_path = paths['features'] / 'scored_candidates.parquet'
    if not scored_path.exists():
        logger.warning("Scored candidates not found, skipping decisioning")
        return
    
    scored_df = pd.read_parquet(scored_path)
    
    # Ensure output dirs exist
    paths['matches'].mkdir(parents=True, exist_ok=True)
    paths['review'].mkdir(parents=True, exist_ok=True)
    
    stats = generate_final_output(
        scored_df=scored_df,
        output_matches_path=str(paths['matches'] / 'matches_final.parquet'),
        output_review_path=str(paths['review'] / 'review_queue.csv')
    )
    
    logger.info(f"Final stats: {stats}")


def step_report(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 10: Generate quality report."""
    import pandas as pd
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("STEP 10: QUALITY REPORT")
    logger.info("=" * 60)
    
    matches_path = paths['matches'] / 'matches_final.parquet'
    if not matches_path.exists():
        logger.warning("Matches not found, skipping report")
        return
    
    matches = pd.read_parquet(matches_path)
    
    # Helper to convert numpy types to native Python (for JSON serialization)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Generate simple report
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_matches': len(matches),
        'tier_distribution': matches['tier'].value_counts().to_dict(),
        'conflict_count': matches['has_conflict'].sum() if 'has_conflict' in matches.columns else 0,
        'match_type_distribution': matches['match_type'].value_counts().to_dict() if 'match_type' in matches.columns else {},
    }
    
    # Convert numpy types
    report = convert_numpy(report)
    
    report_path = paths['reports'] / 'run_manifest.json'
    paths['reports'].mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Summary: {report}")


def step_analytics(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 11: Research analytics with platinum validation."""
    import pandas as pd
    from research_analytics import run_full_analytics
    
    logger.info("=" * 60)
    logger.info("STEP 11: RESEARCH ANALYTICS")
    logger.info("=" * 60)
    
    matches_path = paths['matches'] / 'matches_final.parquet'
    cb_path = paths['cb_clean'] / 'cb_clean.parquet'
    orbis_path = paths['orbis_clean'] / 'orbis_clean.parquet'
    
    if not matches_path.exists():
        logger.warning("Matches not found, skipping analytics")
        return
    
    # Find database-done.xlsx
    db_done_path = paths['project_root'].parent / 'database-done.xlsx'
    
    # Run full analytics
    outputs = run_full_analytics(
        matches_path=str(matches_path),
        cb_data_path=str(cb_path),
        orbis_data_path=str(orbis_path),
        db_done_path=str(db_done_path),
        output_dir=str(paths['reports']),
    )
    
    logger.info(f"Analytics outputs: {list(outputs.keys())}")
    
    # Log key metrics
    stats_path = paths['reports'] / 'research_stats.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        
        if 'validation' in stats:
            val = stats['validation']
            logger.info(f"Platinum validation: {val.get('platinum_found', 0)}/{val.get('platinum_total', 0)} = {val.get('platinum_recall', 0)}%")
        
        logger.info(f"Match rate: {stats.get('sample_sizes', {}).get('match_rate', 0)}%")


# =============================================================================
# STEP REGISTRY
# =============================================================================


def step_embeddings(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 4b: Compute Embeddings (GPU Accelerated)."""
    from embeddings import compute_all_embeddings, build_all_faiss_indexes
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("STEP 4b: EMBEDDINGS (GPU)")
    logger.info("=" * 60)
    
    # Load data
    cb_path = paths['cb_clean'] / 'cb_clean.parquet'
    orbis_path = paths['orbis_clean'] / 'orbis_clean.parquet'
    
    if cb_path.exists() and orbis_path.exists():
        cb_data = pd.read_parquet(cb_path)
        orbis_data = pd.read_parquet(orbis_path)
        
        # Compute embeddings
        embeddings_dir = paths['interim'] / 'embeddings'
        outputs = compute_all_embeddings(
            cb_data=cb_data,
            orbis_data=orbis_data,
            output_dir=str(embeddings_dir),
            model_name=config.get('embeddings.model_name', 'all-MiniLM-L6-v2'),
            batch_size=config.get('embeddings.batch_size', 4096),
            use_streaming=config.get('embeddings.use_streaming', True),
            streaming_chunk_size=config.get('embeddings.streaming_chunk_size', 500000),
        )
        
        # Build FAISS (Optional, good for advanced blocking)
        indexes = build_all_faiss_indexes(
            embeddings_dir=str(embeddings_dir),
            use_gpu=config.get('faiss.use_gpu', False) # On M1 FAISS GPU might be tricky, CPU FAISS is fine
        )
        
        logger.info("Embedding computation complete.")
    else:
        logger.warning(f"Data not found for embeddings. CB: {cb_path.exists()}, Orbis: {orbis_path.exists()}")


# =============================================================================
# STEP REGISTRY
# =============================================================================

def step_rerank(config: Config, paths: Dict, logger: logging.Logger) -> None:
    """Step 8b: Re-rank borderline matches with cross-encoder (SOTA)."""
    import pandas as pd
    from reranking import rerank_step
    
    logger.info("=" * 60)
    logger.info("STEP 8b: CROSS-ENCODER RE-RANKING (SOTA)")
    logger.info("=" * 60)
    
    # Check if re-ranking is enabled
    if not config.get('reranking.enabled', True):
        logger.info("Re-ranking disabled in config, skipping")
        return
    
    # Load scored candidates
    scored_path = paths['features'] / 'scored_candidates.parquet'
    if not scored_path.exists():
        logger.warning("Scored candidates not found, skipping re-ranking")
        return
    
    scored_df = pd.read_parquet(scored_path)
    
    # Re-rank
    model_name = config.get('reranking.model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    result_df = rerank_step(
        scored_df,
        output_path=str(paths['features'] / 'reranked_candidates.parquet'),
        model_name=model_name
    )
    
    logger.info(f"Re-ranked {len(result_df)} candidates")


STEPS = {
    'ingest': step_ingest,
    'normalize': step_normalize,
    'alias': step_alias,
    'index': step_index,
    'embeddings': step_embeddings,
    'blocking': step_blocking,
    'features': step_features,
    'train': step_train,
    'score': step_score,
    'rerank': step_rerank,  # NEW: Cross-encoder re-ranking
    'decide': step_decide,
    'report': step_report,
    'analytics': step_analytics,
}

STEP_ORDER = ['ingest', 'normalize', 'alias', 'index', 'embeddings', 'blocking', 'features', 'train', 'score', 'rerank', 'decide', 'report', 'analytics']



# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('--config', type=str, default='configs/local.yaml',
                       help='Path to config file')
    parser.add_argument('--step', type=str, default='all',
                       help='Step to run (or "all" for full pipeline)')
    parser.add_argument('--incremental', action='store_true',
                       help='Process with current files (don\'t wait for all)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last completed step (skip already done steps)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    config = Config(str(config_path) if config_path.exists() else None)
    
    # Setup paths
    config.set('paths.project_root', str(Path(__file__).parent))
    paths = get_project_paths(config)
    
    # Ensure data_root matches project structure
    if 'data_root' not in paths:
        paths['data_root'] = paths['project_root'] / 'data'
        
    ensure_directories(paths)
    
    # Setup logging with RunContext
    logger, ctx = setup_logging(config, paths)
    
    # =========================================================================
    # CHECKPOINT SYSTEM - Track completed steps for resume capability
    # =========================================================================
    checkpoint_file = paths['interim'] / 'pipeline_checkpoint.json'
    completed_steps = set()
    
    if args.resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed_steps = set(checkpoint_data.get('completed_steps', []))
                logger.info(f"Resuming pipeline. Already completed: {completed_steps}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    def save_checkpoint(step_name: str):
        """Save checkpoint after completing a step."""
        completed_steps.add(step_name)
        checkpoint_data = {
            'completed_steps': list(completed_steps),
            'last_step': step_name,
            'timestamp': datetime.now().isoformat(),
            'run_id': ctx.run_id,
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Checkpoint saved: {step_name}")
    
    logger.info("=" * 60)
    logger.info("ENTITY RESOLUTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Run ID: {ctx.run_id}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Step: {args.step}")
    logger.info(f"Incremental: {args.incremental}")
    logger.info(f"Resume: {args.resume}")
    
    # Run steps
    if args.step == 'all':
        steps_to_run = STEP_ORDER
    else:
        steps_to_run = [args.step]
    
    for step_name in steps_to_run:
        if step_name not in STEPS:
            logger.error(f"Unknown step: {step_name}")
            continue
        
        # Skip completed steps if resuming
        if args.resume and step_name in completed_steps:
            logger.info(f"SKIPPING {step_name} (already completed)")
            continue
        
        start_time = time.time()
        try:
            STEPS[step_name](config, paths, logger)
            # Save checkpoint on success
            save_checkpoint(step_name)
        except Exception as e:
            logger.error(f"Step {step_name} failed: {e}")
            logger.info(f"Pipeline stopped. To resume, run with --resume flag.")
            if not args.incremental:
                raise
        
        elapsed = time.time() - start_time
        logger.info(f"Step {step_name} completed in {elapsed:.1f}s")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    # Clear checkpoint on full completion
    if args.step == 'all' and checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint cleared (pipeline complete)")


if __name__ == '__main__':
    main()
