"""
data_loader.py - Robust Data Ingestion for Entity Resolution

Handles:
1. Schema variations in Crunchbase CSVs
2. Encoding issues
3. Missing columns via smart mapping
4. Robust Excel loading for Orbis
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
import io

logger = logging.getLogger(__name__)

# Standard schema for normalized Crunchbase Data
CB_SCHEMA_MAP = {
    'Transaction Name': 'transaction_name',
    'Organization Name': 'cb_name',
    'Organization Website': 'cb_website',
    'Organization Location': 'cb_hq_location',
    'Organization Industries': 'cb_industries',
    'Organization Description': 'cb_description',
    'Total Funding Amount (in USD)': 'cb_total_funding',
    'Funding Status': 'cb_funding_status',
    'Announced Date': 'cb_founded_date',  # Often proxy for age
    'Contact Email': 'cb_email',
}

def load_crunchbase_csvs(directory: str) -> pd.DataFrame:
    """
    Robustly load and concatenate Crunchbase CSVs.
    
    Handles:
    - Different column orders
    - Missing columns (adds them as NaN)
    - Encoding errors
    """
    dir_path = Path(directory)
    # Recursively find all CSVs in all subfolders
    all_files = sorted(list(dir_path.rglob('*.csv')))
    
    if not all_files:
        logger.warning(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    logger.info(f"Loading {len(all_files)} Crunchbase CSVs from {directory}")
    
    dfs = []
    total_rows = 0
    
    for f in all_files:
        try:
            # Try efficient reading first
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
        except Exception:
            # Fallback to python engine for more robustness
            logger.warning(f"  Standard load failed for {f.name}, retrying with python engine...")
            try:
                df = pd.read_csv(f, on_bad_lines='skip', engine='python')
            except Exception as e:
                logger.error(f"  Failed to load {f.name}: {e}")
                continue
        
        # Normalize columns
        df = _normalize_cb_columns(df)
        
        dfs.append(df)
        total_rows += len(df)
        
        if len(dfs) % 10 == 0:
            logger.info(f"  Processed {len(dfs)}/{len(all_files)} files...")
    
    if not dfs:
        return pd.DataFrame()
    
    # Concatenate - pandas handles missing columns automatically by aligning
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate by name + website to handle overlapping exports
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['cb_name', 'cb_website'])
    
    logger.info(f"Loaded {total_rows} raw rows, {len(combined)} unique companies after deduplication")
    
    # Sanitize for Parquet (PyArrow dislikes mixed types)
    # Convert object columns that might contain mixed floats (NaNs) and strings to string
    for col in combined.columns:
        if combined[col].dtype == 'object':
            # Check if column name suggests numeric
            if 'number of' in str(col).lower() or 'total funding' in str(col).lower():
                 # Try to convert to numeric, coercing errors
                 combined[col] = pd.to_numeric(combined[col], errors='coerce')
            else:
                 # Ensure strings are strings (convert NaN to None or empty string)
                 # Converting to str preserves content but turns None to 'None' or 'nan' string if not careful
                 # Better: fillna('') then astype(str)
                 combined[col] = combined[col].fillna('').astype(str)
                 
    return combined

def _normalize_cb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map columns to standard schema and ensure essential ones exist."""
    # Create map explicitly handling case sensitivity
    col_map = {k.lower(): v for k, v in CB_SCHEMA_MAP.items()}
    
    new_cols = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in col_map:
            new_cols[col] = col_map[col_lower]
    
    # Rename what we found
    df = df.rename(columns=new_cols)
    
    # Ensure key output columns exist
    for target_col in CB_SCHEMA_MAP.values():
        if target_col not in df.columns:
            df[target_col] = None
            
    return df

def load_platinum_matches(file_path: str) -> pd.DataFrame:
    """
    Load verified matches from database-done.xlsx.
    
    Specifically reads Sheet 2 ('Matching 1 platinum-gold-etc').
    """
    try:
        df = pd.read_excel(file_path, sheet_name=2)
        
        # Standardize columns
        df = df.rename(columns={
            'nome_df1': 'orbis_name',
            'nome_df2_match': 'cb_name',
            'website_df1': 'orbis_website',
            'website_df2_match': 'cb_website',
            'match_type': 'match_tier'
        })
        
        # Filter for valid pairs
        df = df[df['cb_name'].notna() & df['orbis_name'].notna()].copy()
        
        # Normalize names for matching key
        df['cb_name_norm'] = df['cb_name'].astype(str).str.lower().str.strip()
        df['orbis_name_norm'] = df['orbis_name'].astype(str).str.lower().str.strip()
        
        logger.info(f"Loaded {len(df)} platinum matches from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load platinum matches: {e}")
        return pd.DataFrame()
