"""
io.py - Data Ingestion & Schema Standardization

This module handles loading raw data from heterogeneous Crunchbase CSV files
and Orbis Excel exports, applying schema normalization.

CRUNCHBASE DATA STRUCTURE:
-------------------------
Folder 1 (OG Crunchbase Data): Funding transactions (30 cols)
    - Primary key: Transaction Name URL
    - Contains: Funding Type, Investor Names, Money Raised
    - Use for: Investor relationship extraction
    
Folder 2 (Company Data to Feed): Minimal lookup (2 cols)
    - Primary key: Name + Domain
    - Use for: Domain validation only

Folder 3 (Company Big All Information): Full profiles (78 cols) <-- PRIMARY FOR MATCHING
    - Primary key: Organization Name URL
    - Contains: Full company data, investors, funding, M&A, IPO
    - Use for: Main matching pipeline

Folder 4 (Investor Data to Feed): Minimal investor lookup (2 cols)
    - Use for: Investor name normalization

Folder 5 (Investors Big All Information): Full investor profiles (30+ cols)
    - Use for: Investor entity resolution
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Folder 3: Company Big All Information (PRIMARY SOURCE)
# 78 columns - This is the main source for entity matching
COMPANY_FULL_SCHEMA = {
    # Core Identifiers
    'cb_id': 'Organization Name URL',
    'cb_name': 'Organization Name',
    
    # Web Presence
    'cb_website': 'Website',
    'cb_twitter': 'Twitter',
    'cb_facebook': 'Facebook',
    'cb_linkedin': 'LinkedIn',
    'cb_email': 'Contact Email',
    'cb_phone': 'Phone Number',
    
    # Location
    'cb_hq_location': 'Headquarters Location',
    'cb_hq_regions': 'Headquarters Regions',
    'cb_postal_code': 'Postal Code',
    
    # Business Description
    'cb_description': 'Description',
    'cb_full_description': 'Full Description',
    'cb_industries': 'Industries',
    'cb_industry_groups': 'Industry Groups',
    
    # Company Metadata
    'cb_rank': 'CB Rank (Company)',
    'cb_operating_status': 'Operating Status',
    'cb_company_type': 'Company Type',
    'cb_diversity_spotlight': 'Diversity Spotlight',
    'cb_revenue_range': 'Estimated Revenue Range',
    'cb_actively_hiring': 'Actively Hiring',
    'cb_hub_tags': 'Hub Tags',
    
    # Temporal
    'cb_founded_date': 'Founded Date',
    'cb_founded_precision': 'Founded Date Precision',
    'cb_exit_date': 'Exit Date',
    'cb_closed_date': 'Closed Date',
    
    # Organization Structure
    'cb_num_suborgs': 'Number of Sub-Orgs',
    'cb_num_founders': 'Number of Founders',
    'cb_founders': 'Founders',
    'cb_num_employees': 'Number of Employees',
    
    # Funding Data
    'cb_funding_status': 'Funding Status',
    'cb_num_funding_rounds': 'Number of Funding Rounds',
    'cb_last_funding_date': 'Last Funding Date',
    'cb_last_funding_amount_usd': 'Last Funding Amount (in USD)',
    'cb_last_funding_type': 'Last Funding Type',
    'cb_total_equity_funding_usd': 'Total Equity Funding Amount (in USD)',
    
    # Investors
    'cb_top5_investors': 'Top 5 Investors',
    'cb_num_lead_investors': 'Number of Lead Investors',
    'cb_num_investors': 'Number of Investors',
    'cb_investor_type': 'Investor Type',
    'cb_investment_stage': 'Investment Stage',
    
    # M&A
    'cb_num_acquisitions': 'Number of Acquisitions',
    'cb_acquisition_status': 'Acquisition Status',
    'cb_acquired_by': 'Acquired by',
    'cb_acquired_by_url': 'Acquired by URL',
    'cb_acquisition_date': 'Announced Date',
    'cb_acquisition_price_usd': 'Price (in USD)',
    'cb_acquisition_type': 'Acquisition Type',
    
    # IPO
    'cb_ipo_status': 'IPO Status',
    'cb_ipo_date': 'IPO Date',
    'cb_ipo_money_raised_usd': 'Money Raised at IPO (in USD)',
    'cb_ipo_valuation_usd': 'Valuation at IPO (in USD)',
    'cb_stock_symbol': 'Stock Symbol',
    'cb_stock_exchange': 'Stock Exchange',
}

# Folder 1: Funding Transactions (for investor relationship enrichment)
FUNDING_SCHEMA = {
    'transaction_url': 'Transaction Name URL',
    'cb_id': 'Organization Name URL',
    'cb_name': 'Organization Name',
    'cb_website': 'Organization Website',
    'funding_type': 'Funding Type',
    'funding_stage': 'Funding Stage',
    'money_raised_usd': 'Money Raised (in USD)',
    'announced_date': 'Announced Date',
    'pre_money_valuation_usd': 'Pre-Money Valuation (in USD)',
    'lead_investors': 'Lead Investors',
    'investor_names': 'Investor Names',
    'num_investors': 'Number of Investors',
    'cb_hq_location': 'Organization Location',
    'cb_industries': 'Organization Industries',
    'cb_description': 'Organization Description',
    'total_funding_usd': 'Total Funding Amount (in USD)',
    'funding_status': 'Funding Status',
}

# Folder 2/4: Minimal lookup tables (Name, Domain only)
LOOKUP_SCHEMA = {
    'name': 'Name',
    'domain': 'Domain',
}

# Folder 5: Full Investor Profiles
INVESTOR_FULL_SCHEMA = {
    'investor_id': 'Organization/Person Name URL',
    'investor_name': 'Organization/Person Name',
    'investor_type': 'Investor Type',
    'num_investments': 'Number of Investments',
    'num_exits': 'Number of Exits',
    'num_exits_ipo': 'Number of Exits (IPO)',
    'num_portfolio_orgs': 'Number of Portfolio Organizations',
    'num_lead_investments': 'Number of Lead Investments',
    'investment_stage': 'Investment Stage',
    'location': 'Location',
    'description': 'Description',
    'full_description': 'Full Description',
    'twitter': 'Twitter',
    'linkedin': 'LinkedIn',
    'facebook': 'Facebook',
    # Person-specific (may be null for organizations)
    'first_name': 'First Name',
    'last_name': 'Last Name',
    'gender': 'Gender',
    'primary_job_title': 'Primary Job Title',
    'primary_organization': 'Primary Organization',
}

# Orbis Schema (expected columns from Excel exports)
# NOTE: Orbis exports may only have "Company name Latin alphabet", not separate "Company name"
# NOTE: Some columns contain newlines (e.g., "City\nLatin Alphabet")
ORBIS_SCHEMA = {
    'bvd_id': 'BvD ID number',
    'orbis_name': 'Company name Latin alphabet',  # This IS the company name in most exports
    'orbis_name_latin': 'Company name Latin alphabet',  # Alias for compatibility
    'orbis_country': 'Country ISO code',
    'orbis_city': 'City',  # May have newline variant
    'orbis_city_latin': 'City Latin Alphabet',  # Normalized name
    'orbis_postcode': 'Postcode',
    'orbis_website': 'Website address',
    'orbis_email': 'E-mail address',
    'orbis_phone': 'Phone number',
    'orbis_incorp_date': 'Date of incorporation',
    'orbis_trade_desc': 'Trade description (English)',
    'orbis_trade_desc_orig': 'Trade description (Original language)',
    'orbis_nace': 'NACE Rev. 2 core code',
    'orbis_nace_desc': 'NACE Rev. 2 core code description',
    'orbis_legal_form': 'Standardised legal form',
    'orbis_operating_revenue': 'Operating revenue (Turnover)',
    'orbis_total_assets': 'Total assets',
    'orbis_num_employees': 'Number of employees',
    # Corporate Structure IDs
    'sub_bvd_id': 'SUB - BvD ID number',
    'sh_bvd_id': 'SH - BvD ID number',
    'guo_bvd_id': 'GUO - BvD ID number',
    'branch_bvd_id': 'BRANCH - BvD ID number',
    # Shareholder Info
    'sh_name': 'SH - Name',
    'sh_country': 'SH - Country ISO code',
    'sh_type': 'SH - Type',
    'sh_direct_pct': 'SH - Direct %',
    'sh_total_pct': 'SH - Total %',
}

# Columns that might have alternative names across files
# Key: canonical name used in schema, Value: list of alternatives
COLUMN_ALIASES = {
    'Company name Latin alphabet': ['Latin alphabet', 'Company name(Latin alphabet)', 'Company name'],
    'City Latin Alphabet': ['City (Latin Alphabet)', 'City\nLatin Alphabet', 'City_Latin'],
    'City': ['City\nLatin Alphabet'],  # City column often has newline
    'E-mail address': ['Email address', 'E-Mail address', 'Email'],
    'Website address': ['Website', 'Web address'],
    'Date of incorporation': ['Incorporation date', 'Date of Incorporation'],
    'BvD ID number': ['BvD ID', 'BVDID', 'BvD_ID'],
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def detect_encoding(file_path: str) -> str:
    """Detect file encoding, handling BOM markers."""
    # Check for UTF-8 BOM
    with open(file_path, 'rb') as f:
        first_bytes = f.read(4)
    
    if first_bytes.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    elif first_bytes.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    elif first_bytes.startswith(b'\xfe\xff'):
        return 'utf-16-be'
    else:
        return 'utf-8'


def safe_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Safely read CSV with encoding detection and error handling."""
    encoding = detect_encoding(file_path)
    
    default_kwargs = {
        'encoding': encoding,
        'on_bad_lines': 'warn',
        'low_memory': False,
    }
    default_kwargs.update(kwargs)
    
    try:
        return pd.read_csv(file_path, **default_kwargs)
    except Exception as e:
        logger.warning(f"Error reading {file_path} with {encoding}: {e}")
        # Fallback to latin-1 which can read any byte sequence
        default_kwargs['encoding'] = 'latin-1'
        return pd.read_csv(file_path, **default_kwargs)


def normalize_column_name(col: str) -> str:
    """Normalize column name for matching - handles newlines, extra spaces, etc."""
    # Replace newlines and multiple spaces with single space
    col = ' '.join(col.split())
    return col.strip().lower().replace('_', ' ').replace('-', ' ')


def find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    """
    Find column by name or alias, case-insensitive.
    Handles: newlines in column names, extra whitespace, case variations.
    """
    target_norm = normalize_column_name(target)
    
    # Exact match first (normalized)
    for col in df.columns:
        if normalize_column_name(col) == target_norm:
            return col
    
    # Check aliases
    aliases = COLUMN_ALIASES.get(target, [])
    for alias in aliases:
        for col in df.columns:
            if normalize_column_name(col) == normalize_column_name(alias):
                return col
    
    # Partial match as fallback (for columns with extra qualifiers)
    for col in df.columns:
        col_norm = normalize_column_name(col)
        if target_norm in col_norm or col_norm in target_norm:
            logger.debug(f"Partial match: '{target}' -> '{col}'")
            return col
    
    return None


def apply_schema_mapping(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Apply schema mapping, handling missing columns gracefully."""
    result = pd.DataFrame()
    
    for new_name, original_name in schema.items():
        col = find_column(df, original_name)
        if col is not None:
            result[new_name] = df[col]
        else:
            result[new_name] = pd.NA
            logger.debug(f"Column '{original_name}' not found, setting to NA")
    
    return result


# =============================================================================
# CRUNCHBASE LOADERS
# =============================================================================

def load_crunchbase_companies(data_dir: str, use_parallel: bool = True) -> pd.DataFrame:
    """
    Load company data from Folder 3 (Company Big All Information).
    This is the PRIMARY source for entity matching.
    
    Args:
        data_dir: Path to 'dati europe cb' directory
        use_parallel: Whether to use parallel loading
    
    Returns:
        DataFrame with standardized schema
    """
    folder_path = Path(data_dir) / "3. Company Big All Information"
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    logger.info(f"Loading {len(csv_files)} company CSV files from Folder 3")
    
    dfs = []
    
    def load_single(f):
        try:
            df = safe_read_csv(f)
            return apply_schema_mapping(df, COMPANY_FULL_SCHEMA)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            return None
    
    if use_parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(load_single, f): f for f in csv_files}
            for future in tqdm(as_completed(futures), total=len(csv_files), desc="Loading companies"):
                result = future.result()
                if result is not None:
                    dfs.append(result)
    else:
        for f in tqdm(csv_files, desc="Loading companies"):
            result = load_single(f)
            if result is not None:
                dfs.append(result)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate by cb_id (keep first occurrence)
    original_count = len(combined)
    combined = combined.drop_duplicates(subset=['cb_id'], keep='first')
    logger.info(f"Loaded {len(combined)} unique companies (deduped from {original_count})")
    
    return combined


def load_crunchbase_funding(data_dir: str) -> pd.DataFrame:
    """
    Load funding transactions from Folder 1 (OG Crunchbase Data).
    Used for enriching investor relationships.
    """
    folder_path = Path(data_dir) / "1. OG Crunchbase Data"
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    
    # Exclude joiner.py and other non-CSV
    csv_files = [f for f in csv_files if f.endswith('.csv')]
    
    logger.info(f"Loading {len(csv_files)} funding CSV files from Folder 1")
    
    dfs = []
    for f in tqdm(csv_files, desc="Loading funding"):
        try:
            df = safe_read_csv(f)
            dfs.append(apply_schema_mapping(df, FUNDING_SCHEMA))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} funding transactions")
    
    return combined


def load_crunchbase_investors(data_dir: str) -> pd.DataFrame:
    """
    Load investor profiles from Folder 5 (Investors Big All Information).
    Used for investor entity resolution.
    """
    folder_path = Path(data_dir) / "5. Investors Big All Information"
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    
    logger.info(f"Loading {len(csv_files)} investor CSV files from Folder 5")
    
    dfs = []
    for f in tqdm(csv_files, desc="Loading investors"):
        try:
            df = safe_read_csv(f)
            dfs.append(apply_schema_mapping(df, INVESTOR_FULL_SCHEMA))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate by investor_id
    combined = combined.drop_duplicates(subset=['investor_id'], keep='first')
    logger.info(f"Loaded {len(combined)} unique investors")
    
    return combined


def load_company_lookup(data_dir: str) -> pd.DataFrame:
    """
    Load minimal company lookup from Folder 2 (Company Data to Feed).
    Used for domain validation.
    """
    folder_path = Path(data_dir) / "2. Company Data to Feed in CB"
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    
    dfs = []
    for f in csv_files:
        if f.endswith('.csv'):
            try:
                df = safe_read_csv(f)
                # Handle BOM in column names
                df.columns = [c.lstrip('\ufeff') for c in df.columns]
                if 'Name' in df.columns and 'Domain' in df.columns:
                    dfs.append(df[['Name', 'Domain']])
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.columns = ['name', 'domain']
        return combined.drop_duplicates()
    
    return pd.DataFrame(columns=['name', 'domain'])


# =============================================================================
# ORBIS LOADERS
# =============================================================================

def aggregate_orbis_multirow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Orbis multi-row structure into one row per company.
    
    ORBIS STRUCTURE EXPLANATION:
    ----------------------------
    Orbis exports use a multi-row format where:
    - First column (usually 'Unnamed: 0') contains the company index (1, 2, 3, ...)
    - Only the FIRST row for each company has the BvD ID and core company data
    - CONTINUATION rows (89.7% of data) have NaN in BvD ID but contain:
      - Additional websites
      - Additional subsidiaries (SUB - BvD ID)
      - Additional shareholders (SH - BvD ID)
      - Additional branches (BRANCH - BvD ID)
    
    This function:
    1. Groups rows by company index (first column)
    2. Takes core data from the first row (BvD ID, name, country, etc.)
    3. Aggregates multi-value fields into arrays/lists
    
    Returns:
        DataFrame with one row per company, multi-values as arrays
    """
    # Identify the index column (first column, usually 'Unnamed: 0')
    first_col = df.columns[0]
    
    # Forward-fill the index to group continuation rows
    df['_company_idx'] = df[first_col].ffill()
    
    # Define which columns should be aggregated (multi-value fields)
    AGGREGATE_COLS = [
        'orbis_website', 'orbis_email',
        'sub_bvd_id', 'sh_bvd_id', 'guo_bvd_id', 'branch_bvd_id',
        'sh_name', 'sh_country', 'sh_type', 'sh_direct_pct', 'sh_total_pct'
    ]
    
    # Define which columns should take first non-null value (core company data)
    FIRST_VALUE_COLS = [
        'bvd_id', 'orbis_name', 'orbis_name_latin', 
        'orbis_country', 'orbis_city', 'orbis_city_latin', 'orbis_postcode',
        'orbis_phone', 'orbis_incorp_date',
        'orbis_trade_desc', 'orbis_trade_desc_orig',
        'orbis_nace', 'orbis_nace_desc', 'orbis_legal_form',
        'orbis_operating_revenue', 'orbis_total_assets', 'orbis_num_employees'
    ]
    
    def aggregate_group(group: pd.DataFrame) -> pd.Series:
        """Aggregate a single company's rows."""
        result = {}
        
        # Core data: take first non-null value
        for col in FIRST_VALUE_COLS:
            if col in group.columns:
                non_null = group[col].dropna()
                result[col] = non_null.iloc[0] if len(non_null) > 0 else pd.NA
            else:
                result[col] = pd.NA
        
        # Multi-value fields: aggregate unique non-null values into list
        for col in AGGREGATE_COLS:
            if col in group.columns:
                values = group[col].dropna().unique().tolist()
                if len(values) == 0:
                    result[col] = pd.NA
                elif len(values) == 1:
                    result[col] = values[0]
                else:
                    # Store as pipe-separated string for parquet compatibility
                    result[col] = '|'.join(str(v) for v in values)
            else:
                result[col] = pd.NA
        
        return pd.Series(result)
    
    # Optimized Aggregation using vectorization instead of apply()
    
    # 1. First-value columns: Group by company index and take first
    # This is extremely fast
    first_vals = df.groupby('_company_idx')[FIRST_VALUE_COLS].first()
    
    # 2. Multi-value columns: We need to aggregate unique values into pipe-separated strings
    # We'll use a custom aggregation function only for these columns
    
    def agg_unique_join(x):
        # Drop NAs, unique, join
        vals = x.dropna().unique()
        if len(vals) == 0:
            return pd.NA
        return '|'.join(str(v) for v in vals)

    # Only aggregate multi-value cols if they exist in df
    cols_to_agg = [c for c in AGGREGATE_COLS if c in df.columns]
    
    if cols_to_agg:
        multi_vals = df.groupby('_company_idx')[cols_to_agg].agg(agg_unique_join)
        # Join back
        aggregated = first_vals.join(multi_vals)
    else:
        aggregated = first_vals

    # Reset index to make _company_idx a column again (or just drop it if not needed)
    aggregated = aggregated.reset_index(drop=True)
    
    # Filter out rows without BvD ID (invalid)
    aggregated = aggregated[aggregated['bvd_id'].notna()]
    
    return aggregated


def load_orbis_single_excel(file_path: str, aggregate: bool = True) -> Optional[pd.DataFrame]:
    """
    Load a single Orbis Excel file, extracting the Results sheet.
    
    IMPORTANT: Orbis Excel files use a multi-row format where companies with
    multiple subsidiaries/websites span multiple rows. The first column contains
    the company index, and continuation rows have NaN in the BvD ID column.
    
    Args:
        file_path: Path to Excel file
        aggregate: If True, aggregate multi-rows into single row per company
                  If False, return raw data (for inspection)
    
    Returns:
        DataFrame with standardized schema, one row per company
    """
    try:
        # Read the 'Results' sheet specifically
        # Use calamine engine for speed (requires python-calamine installed)
        try:
            xls = pd.ExcelFile(file_path, engine='calamine')
        except ImportError:
            # Fallback if calamine not installed
            xls = pd.ExcelFile(file_path)
        
        # Find the main data sheet
        if 'Results' in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name='Results')
        else:
            # Take the sheet with most rows
            # Optimization: don't read full sheets to count rows if possible, 
            # but with calamine reading is fast so this is okay-ish.
            # Ideally we just guess sheet 0 if Results isn't there.
            if len(xls.sheet_names) > 0:
                df = pd.read_excel(xls, sheet_name=0)
            else:
                return None
        
        # Apply schema mapping first
        df_mapped = apply_schema_mapping(df, ORBIS_SCHEMA)
        
        # Add the first column for grouping (company index)
        df_mapped.insert(0, '_raw_idx', df.iloc[:, 0])
        
        if aggregate:
            # Aggregate multi-row structure
            df_final = aggregate_orbis_multirow(df_mapped)
            logger.debug(f"Loaded {file_path}: {len(df)} rows -> {len(df_final)} companies")
            return df_final
        else:
            return df_mapped
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def load_orbis_raw(data_dir: str, max_workers: int = 4, batch_size: int = 50) -> pd.DataFrame:
    """
    Load all Orbis Excel files from directory.
    
    Args:
        data_dir: Path to 'new orbis' directory
        max_workers: Number of parallel workers
        batch_size: Files per batch (for memory management)
    
    Returns:
        Combined DataFrame with standardized schema
    """
    excel_files = sorted(glob.glob(str(Path(data_dir) / "*.xlsx")))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")
    
    logger.info(f"Loading {len(excel_files)} Orbis Excel files")
    
    all_dfs = []
    
    # Process in batches to manage memory
    for i in range(0, len(excel_files), batch_size):
        batch_files = excel_files[i:i+batch_size]
        batch_dfs = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_orbis_single_excel, f): f for f in batch_files}
            for future in tqdm(as_completed(futures), total=len(batch_files), 
                             desc=f"Loading batch {i//batch_size + 1}"):
                result = future.result()
                if result is not None and len(result) > 0:
                    batch_dfs.append(result)
        
        if batch_dfs:
            batch_combined = pd.concat(batch_dfs, ignore_index=True)
            all_dfs.append(batch_combined)
            logger.info(f"Batch {i//batch_size + 1}: {len(batch_combined)} records")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplicate by bvd_id
    original_count = len(combined)
    combined = combined.drop_duplicates(subset=['bvd_id'], keep='first')
    logger.info(f"Loaded {len(combined)} unique Orbis records (deduped from {original_count})")
    
    return combined


# =============================================================================
# PARQUET CONVERSION
# =============================================================================

def save_to_parquet(df: pd.DataFrame, output_path: str, compression: str = 'snappy'):
    """Save DataFrame to Parquet with optimized settings and robust type handling."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert ALL non-numeric columns to string for Arrow compatibility
    for col in df.columns:
        dtype = df[col].dtype
        # Skip numeric types
        if dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
            continue
        # Convert everything else to string
        try:
            df[col] = df[col].astype(str).replace('nan', None).replace('None', None)
        except Exception:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)
    
    df.to_parquet(output_path, compression=compression, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved {len(df)} records to {output_path} ({file_size_mb:.1f} MB)")


def convert_orbis_to_parquet_streaming(
    data_dir: str, 
    output_path: str,
    batch_size: int = 100
) -> None:
    """
    Convert Orbis Excel files to Parquet in streaming fashion.
    Optimized for disk space - processes and deletes as it goes.
    
    Args:
        data_dir: Path to 'new orbis' directory
        output_path: Path for output parquet file
        batch_size: Files to process before writing
    """
    excel_files = sorted(glob.glob(str(Path(data_dir) / "*.xlsx")))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")
    
    logger.info(f"Converting {len(excel_files)} Excel files to Parquet (streaming)")
    
    writer = None
    total_records = 0
    
    try:
        for i in range(0, len(excel_files), batch_size):
            batch_files = excel_files[i:i+batch_size]
            batch_dfs = []
            
            # Parallel processing for batch
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(load_orbis_single_excel, f): f for f in batch_files}
                
                for future in tqdm(as_completed(futures), total=len(batch_files), desc=f"Batch {i//batch_size + 1}"):
                    try:
                        df = future.result()
                        if df is not None and len(df) > 0:
                            batch_dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Worker failed: {e}")
            
            if batch_dfs:
                batch_df = pd.concat(batch_dfs, ignore_index=True)
                batch_df = batch_df.drop_duplicates(subset=['bvd_id'], keep='first')
                
                # Convert to PyArrow Table
                # CRITICAL: Reset index to avoid schema mismatch between batches
                # Some DFs have a RangeIndex, others have an explicit integer index.
                # Arrow may serialize the index as `__index_level_0__` inconsistently.
                batch_df = batch_df.reset_index(drop=True)
                
                for col in batch_df.select_dtypes(include=['object']).columns:
                    batch_df[col] = batch_df[col].astype('string')
                
                # Force preserve_index=False to NEVER include index in schema
                table = pa.Table.from_pandas(batch_df, preserve_index=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                
                writer.write_table(table)
                total_records += len(batch_df)
                
                logger.info(f"Written {total_records} records so far")
    
    finally:
        if writer is not None:
            writer.close()
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Conversion complete: {total_records} records, {file_size_mb:.1f} MB")


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================

def ingest_all_data(
    crunchbase_dir: str,
    orbis_dir: str,
    output_dir: str,
    skip_orbis: bool = False
) -> Dict[str, str]:
    """
    Run complete data ingestion pipeline.
    
    Args:
        crunchbase_dir: Path to 'dati europe cb'
        orbis_dir: Path to 'new orbis'
        output_dir: Path to output directory (e.g., 'data/interim')
        skip_orbis: Skip Orbis processing (useful if already done)
    
    Returns:
        Dict mapping dataset names to output paths
    """
    output_paths = {}
    
    # Create output directories
    cb_output = Path(output_dir) / "cb_clean"
    orbis_output = Path(output_dir) / "orbis_clean"
    cb_output.mkdir(parents=True, exist_ok=True)
    orbis_output.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and save Crunchbase companies (PRIMARY)
    logger.info("=== Loading Crunchbase Companies ===")
    cb_companies = load_crunchbase_companies(crunchbase_dir)
    cb_path = str(cb_output / "cb_raw_companies.parquet")
    save_to_parquet(cb_companies, cb_path)
    output_paths['cb_companies'] = cb_path
    
    # 2. Load and save Crunchbase funding (for investor enrichment)
    logger.info("=== Loading Crunchbase Funding ===")
    cb_funding = load_crunchbase_funding(crunchbase_dir)
    funding_path = str(cb_output / "cb_raw_funding.parquet")
    save_to_parquet(cb_funding, funding_path)
    output_paths['cb_funding'] = funding_path
    
    # 3. Load and save Crunchbase investors
    logger.info("=== Loading Crunchbase Investors ===")
    cb_investors = load_crunchbase_investors(crunchbase_dir)
    investors_path = str(cb_output / "cb_raw_investors.parquet")
    save_to_parquet(cb_investors, investors_path)
    output_paths['cb_investors'] = investors_path
    
    # 4. Load and save company lookup
    logger.info("=== Loading Company Lookup ===")
    cb_lookup = load_company_lookup(crunchbase_dir)
    lookup_path = str(cb_output / "cb_domain_lookup.parquet")
    save_to_parquet(cb_lookup, lookup_path)
    output_paths['cb_lookup'] = lookup_path
    
    # 5. Convert Orbis to Parquet (streaming for memory efficiency)
    if not skip_orbis:
        logger.info("=== Converting Orbis to Parquet ===")
        orbis_path = str(orbis_output / "orbis_raw.parquet")
        convert_orbis_to_parquet_streaming(orbis_dir, orbis_path)
        output_paths['orbis'] = orbis_path
    
    logger.info("=== Data Ingestion Complete ===")
    for name, path in output_paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"  {name}: {path} ({size_mb:.1f} MB)")
    
    return output_paths


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Data Ingestion Pipeline')
    parser.add_argument('--cb-dir', required=True, help='Path to Crunchbase data')
    parser.add_argument('--orbis-dir', required=True, help='Path to Orbis data')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--skip-orbis', action='store_true', help='Skip Orbis processing')
    
    args = parser.parse_args()
    
    ingest_all_data(
        crunchbase_dir=args.cb_dir,
        orbis_dir=args.orbis_dir,
        output_dir=args.output_dir,
        skip_orbis=args.skip_orbis
    )
