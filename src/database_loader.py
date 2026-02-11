"""
database_loader.py - Comprehensive Data Extraction from database-done.xlsx

ULTRATHINK ANALYSIS revealed we were only using ~45% of available data.
This module extracts MAXIMUM value from all 7 sheets.

SHEET BREAKDOWN:
----------------
Sheet 1 "DB company crunchbase" (18,218): CB companies with domains
Sheet 2 "bvd id" (5,823): CB→BVD mappings (NOTE: BVD ID column was EMPTY!)
Sheet 3 "Matching 1 platinum" (8,084): Platinum domain-matched pairs ✓
Sheet 4 "Matching 2" (12,206): AI-scraped legal names + VAT codes
Sheet 5 "Matching manuale" (7,085): Manual matches (314 with company names)
Sheet 6 "Matching ai" (5,013): AI-extracted legal names
Sheet 7 "Deal" (32,928): Funding transactions (for investor matching)

TOTAL RECOVERABLE DATA:
- Platinum matches: 8,084
- Legal name aliases: 5,141 + 4,575 = 9,716
- Manual verified: 314
- VAT codes: 2,064
- TOTAL: ~20,000+ data points (was using ~8,000)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_all_database_sheets(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all sheets from database-done.xlsx for inspection.
    """
    xl = pd.ExcelFile(file_path)
    sheets = {}
    for name in xl.sheet_names:
        sheets[name] = pd.read_excel(xl, sheet_name=name)
        logger.info(f"Loaded sheet '{name}': {len(sheets[name])} rows")
    return sheets


def extract_platinum_matches(file_path: str) -> pd.DataFrame:
    """
    Extract platinum matches from Sheet 3 "Matching 1 platinum".
    
    Returns DataFrame with:
    - cb_name: Crunchbase company name
    - orbis_name: Matched Orbis legal name
    - cb_website: CB website
    - orbis_website: Orbis website
    - match_type: platinum_website_exact, gold, etc.
    - confidence_score: Match confidence
    """
    df = pd.read_excel(file_path, sheet_name="Matching 1 platinum-gold-etc")
    
    result = pd.DataFrame({
        'cb_name': df['nome_df2_match'],
        'orbis_name': df['nome_df1'],
        'cb_website': df['website_df2_match'],
        'orbis_website': df['website_df1'],
        'match_type': df['match_type'],
        'confidence_score': df['confidence_score'],
    }).dropna(subset=['cb_name', 'orbis_name'])
    
    logger.info(f"Extracted {len(result)} platinum matches")
    return result


def extract_legal_names_sheet4(file_path: str) -> pd.DataFrame:
    """
    Extract AI-scraped legal names from Sheet 4 "Matching 2".
    
    Contains:
    - 5,141 legal company names
    - 3,536 addresses
    - 2,064 VAT codes (VALUABLE for exact matching!)
    """
    df = pd.read_excel(file_path, sheet_name="Matching 2")
    
    result = pd.DataFrame({
        'domain': df['dominio'],
        'legal_name': df['json_final_company_name'],
        'address': df['json_final_address'],
        'vat_code': df['json_final_vat_code'],
        'scrape_status': df['status_errore'],
    })
    
    # Filter to successful scrapes with legal names
    result = result[
        (result['scrape_status'] == 'OK') & 
        (result['legal_name'].notna())
    ]
    
    logger.info(f"Extracted {len(result)} legal names from Sheet 4")
    logger.info(f"  With VAT codes: {result['vat_code'].notna().sum()}")
    logger.info(f"  With addresses: {result['address'].notna().sum()}")
    
    return result


def extract_legal_names_sheet6(file_path: str) -> pd.DataFrame:
    """
    Extract AI-extracted legal names from Sheet 6 "Matching ai".
    
    Contains 4,575 legal names.
    """
    df = pd.read_excel(file_path, sheet_name="Matching ai ")
    
    result = pd.DataFrame({
        'input_ref': df['input_ref'],
        'cb_name': df['Name'],
        'domain': df['Website'],
        'legal_name': df['legal_name'],
        'address': df['address'],
        'notes': df['notes'],
    })
    
    # Filter to valid entries
    result = result[
        (result['legal_name'].notna()) & 
        (result['legal_name'] != 'Information not available')
    ]
    
    logger.info(f"Extracted {len(result)} legal names from Sheet 6")
    return result


def extract_manual_matches(file_path: str) -> pd.DataFrame:
    """
    Extract manually verified matches from Sheet 5 "Matching manuale".
    
    P3 FIX: The 'company name' column contains verbose text like:
      "WOODFLOW TECHNOLOGIES S.L., registered in Madrid, Spain."
    We parse out the actual legal name by splitting at known phrase boundaries.
    This recovers ~4,464 matches (was 314 due to overly strict filtering).
    """
    df = pd.read_excel(file_path, sheet_name="Matching manuale")
    
    result = pd.DataFrame({
        'domain': df['dominio'],
        'cb_name': df['name '],
        'legal_name_raw': df['company name'],
    })
    
    # Only rows with actual company names
    result = result[result['legal_name_raw'].notna()]
    
    # Parse verbose text → clean legal name
    import re
    
    def _parse_legal_name(raw: str) -> str:
        """Extract legal name from verbose description."""
        raw = str(raw).strip()
        if not raw:
            return ''
        # Split at known phrase boundaries
        # e.g. "WOODFLOW TECHNOLOGIES S.L., registered in Madrid, Spain."
        for sep in [', registered', ', based', ', founded', ', headquartered',
                    ', incorporated', ', located', ', operating', '. The company',
                    '. It ', '. Founded', '. Based']:
            if sep.lower() in raw.lower():
                idx = raw.lower().index(sep.lower())
                raw = raw[:idx]
                break
        # Also trim trailing ", [Country]" pattern (e.g. ", Spain")
        raw = re.sub(r',\s*[A-Z][a-z]+\.?$', '', raw)
        return raw.strip().rstrip('.,')
    
    result['legal_name'] = result['legal_name_raw'].apply(_parse_legal_name)
    result = result[result['legal_name'].str.len() > 1]
    result = result.drop(columns=['legal_name_raw'])
    
    logger.info(f"Extracted {len(result)} manual matches (P3: verbose text parsed)")
    return result


def extract_deal_metadata(file_path: str) -> pd.DataFrame:
    """
    P1: Extract comprehensive metadata from Sheet 7 "Deal".
    
    32,928 funding rounds → deduplicated to ~12K unique companies.
    100% fill on: Organization Industries, Description, Location.
    
    Returns DataFrame with:
    - cb_name: Organization Name
    - cb_industries: pipe-separated industry list
    - cb_description: company description
    - cb_location: full location string
    - cb_country_parsed: extracted country
    - cb_city_parsed: extracted city
    - cb_total_funding_usd: total funding amount
    - cb_num_rounds: number of funding rounds
    - cb_revenue_range: revenue range if available
    """
    df = pd.read_excel(file_path, sheet_name="Deal")
    
    # Core columns (names from typical CB export)
    col_map = {
        'Organization Name': 'cb_name',
        'Organization Industries': 'cb_industries',
        'Organization Description': 'cb_description',
        'Organization Location': 'cb_location',
        'Money Raised': 'money_raised',
        'Organization Revenue Range': 'cb_revenue_range',
    }
    
    # Map available columns
    available = {}
    for src, dst in col_map.items():
        if src in df.columns:
            available[dst] = df[src]
        else:
            # Try case-insensitive match
            for c in df.columns:
                if c.strip().lower() == src.lower():
                    available[dst] = df[c]
                    break
    
    if 'cb_name' not in available:
        logger.warning("Deal sheet: 'Organization Name' column not found")
        return pd.DataFrame()
    
    result = pd.DataFrame(available)
    
    # Deduplicate by company name — aggregate per company
    agg_funcs = {
        'cb_industries': 'first',
        'cb_description': 'first',
        'cb_location': 'first',
        'cb_revenue_range': 'first',
    }
    # Only aggregate columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in result.columns}
    
    if 'money_raised' in result.columns:
        agg_funcs['money_raised'] = 'sum'
    
    # Count rounds
    result['_round'] = 1
    agg_funcs['_round'] = 'sum'
    
    grouped = result.groupby('cb_name', as_index=False).agg(agg_funcs)
    grouped = grouped.rename(columns={
        '_round': 'cb_num_rounds',
        'money_raised': 'cb_total_funding_usd',
    })
    
    # Parse location → country, city
    if 'cb_location' in grouped.columns:
        # Location format: "City, Region, Country" or "City, Country"
        loc = grouped['cb_location'].fillna('').astype(str)
        parts = loc.str.rsplit(',', n=1)
        grouped['cb_country_parsed'] = parts.str[-1].str.strip()
        grouped['cb_city_parsed'] = loc.str.split(',').str[0].str.strip()
    
    logger.info(f"Extracted Deal metadata: {len(grouped)} unique companies "
                f"from {len(df)} funding rounds")
    if 'cb_industries' in grouped.columns:
        logger.info(f"  With industries: {grouped['cb_industries'].notna().sum()}")
    if 'cb_description' in grouped.columns:
        logger.info(f"  With descriptions: {grouped['cb_description'].notna().sum()}")
    
    return grouped


def get_silver_label_pairs(file_path: str) -> pd.DataFrame:
    """
    P2: Extract silver-label training pairs from Sheet 2 "bvd id".
    
    5,823 confirmed CB name → Orbis legal name mappings.
    Currently only used for alias blocking — NOT as training positives.
    
    Returns DataFrame with:
    - cb_name: CB brand name
    - orbis_name: Orbis legal name
    - cb_website: CB website
    - source: 'silver_sheet2'
    - confidence: 0.85
    """
    df = pd.read_excel(file_path, sheet_name="bvd id")
    
    result = pd.DataFrame({
        'cb_name': df['company name'],
        'orbis_name': df['legal name'],
        'cb_website': df['website'],
        'source': 'silver_sheet2',
        'confidence': 0.85,
    })
    
    result = result.dropna(subset=['cb_name', 'orbis_name'])
    # Filter out placeholder values
    result = result[
        ~result['orbis_name'].str.lower().isin(['n/a', 'na', '-', 'unknown', 'none', ''])
    ]
    
    logger.info(f"Extracted {len(result)} silver training label pairs from Sheet 2")
    return result


def extract_vat_codes(file_path: str) -> pd.DataFrame:
    """
    Extract VAT codes for exact matching.
    VAT codes are unique identifiers - perfect for Tier A matching!
    """
    df = pd.read_excel(file_path, sheet_name="Matching 2")
    
    result = pd.DataFrame({
        'domain': df['dominio'],
        'vat_code': df['json_final_vat_code'],
        'legal_name': df['json_final_company_name'],
    })
    
    result = result[result['vat_code'].notna()]
    
    # Clean VAT codes (remove spaces, uppercase)
    result['vat_code_clean'] = result['vat_code'].str.replace(r'\s+', '', regex=True).str.upper()
    
    logger.info(f"Extracted {len(result)} VAT codes for exact matching")
    return result


def extract_cb_companies_with_domains(file_path: str) -> pd.DataFrame:
    """
    Extract CB companies from Sheet 1 for domain-based lookup.
    """
    df = pd.read_excel(file_path, sheet_name="DB company crunchbase")
    
    result = pd.DataFrame({
        'cb_name': df['Organization Name'],
        'cb_website': df['Organization Website'],
        'domain': df['Unnamed: 2'],  # Extracted domain
    })
    
    result = result[result['domain'].notna()]
    
    logger.info(f"Extracted {len(result)} CB companies with domains")
    return result


def build_comprehensive_alias_registry(file_path: str) -> Dict[str, List[str]]:
    """
    Build comprehensive alias registry from ALL sources.
    
    Returns dict: domain -> [list of known legal names]
    """
    registry = {}
    
    # Source 1: Sheet 4 legal names
    try:
        sheet4 = extract_legal_names_sheet4(file_path)
        for _, row in sheet4.iterrows():
            domain = str(row['domain']).lower().strip()
            legal_name = str(row['legal_name']).strip()
            if domain and legal_name:
                if domain not in registry:
                    registry[domain] = []
                if legal_name not in registry[domain]:
                    registry[domain].append(legal_name)
    except Exception as e:
        logger.warning(f"Failed to load Sheet 4: {e}")
    
    # Source 2: Sheet 6 AI legal names
    try:
        sheet6 = extract_legal_names_sheet6(file_path)
        for _, row in sheet6.iterrows():
            domain = str(row['domain']).lower().strip()
            legal_name = str(row['legal_name']).strip()
            if domain and legal_name:
                if domain not in registry:
                    registry[domain] = []
                if legal_name not in registry[domain]:
                    registry[domain].append(legal_name)
    except Exception as e:
        logger.warning(f"Failed to load Sheet 6: {e}")
    
    # Source 3: Sheet 5 manual matches
    try:
        sheet5 = extract_manual_matches(file_path)
        for _, row in sheet5.iterrows():
            domain = str(row['domain']).lower().strip()
            legal_name = str(row['legal_name']).strip()
            if domain and legal_name:
                if domain not in registry:
                    registry[domain] = []
                if legal_name not in registry[domain]:
                    registry[domain].append(legal_name)
    except Exception as e:
        logger.warning(f"Failed to load Sheet 5: {e}")
    
    total_aliases = sum(len(v) for v in registry.values())
    logger.info(f"Built comprehensive alias registry:")
    logger.info(f"  Domains: {len(registry)}")
    logger.info(f"  Total aliases: {total_aliases}")
    
    return registry


def build_vat_lookup(file_path: str) -> Dict[str, str]:
    """
    Build VAT code lookup: vat_code -> domain
    
    For exact matching in Orbis.
    """
    vat_df = extract_vat_codes(file_path)
    
    lookup = {}
    for _, row in vat_df.iterrows():
        vat = row['vat_code_clean']
        domain = row['domain']
        if vat and domain:
            lookup[vat] = domain
    
    logger.info(f"Built VAT lookup with {len(lookup)} codes")
    return lookup


def get_all_prematched_pairs(file_path: str) -> pd.DataFrame:
    """
    Get ALL pre-matched pairs from ALL sources for direct injection.
    
    Returns DataFrame with:
    - cb_name, cb_domain, orbis_name, source, confidence
    """
    all_pairs = []
    
    # Source 1: Platinum matches (highest confidence)
    try:
        platinum = extract_platinum_matches(file_path)
        for _, row in platinum.iterrows():
            all_pairs.append({
                'cb_name': row['cb_name'],
                'cb_domain': row.get('cb_website', ''),
                'orbis_name': row['orbis_name'],
                'source': 'platinum_match',
                'confidence': 1.0,
            })
    except Exception as e:
        logger.warning(f"Failed to load platinum matches: {e}")
    
    # Source 2: Manual matches (high confidence)
    try:
        manual = extract_manual_matches(file_path)
        for _, row in manual.iterrows():
            all_pairs.append({
                'cb_name': row['cb_name'],
                'cb_domain': row['domain'],
                'orbis_name': row['legal_name'],
                'source': 'manual_match',
                'confidence': 0.95,
            })
    except Exception as e:
        logger.warning(f"Failed to load manual matches: {e}")
    
    # Source 3: Silver labels from Sheet 2 (P2)
    try:
        silver = get_silver_label_pairs(file_path)
        for _, row in silver.iterrows():
            all_pairs.append({
                'cb_name': row['cb_name'],
                'cb_domain': row.get('cb_website', ''),
                'orbis_name': row['orbis_name'],
                'source': 'silver_sheet2',
                'confidence': 0.85,
            })
    except Exception as e:
        logger.warning(f"Failed to load silver labels: {e}")
    
    result = pd.DataFrame(all_pairs)
    
    # Deduplicate
    result = result.drop_duplicates(subset=['cb_name', 'orbis_name'])
    
    logger.info(f"Total pre-matched pairs: {len(result)}")
    logger.info(f"  Platinum: {(result['source'] == 'platinum_match').sum()}")
    logger.info(f"  Manual: {(result['source'] == 'manual_match').sum()}")
    
    return result


def get_database_stats(file_path: str) -> Dict:
    """
    Get comprehensive stats about available data.
    """
    stats = {
        'platinum_matches': 0,
        'legal_names_sheet4': 0,
        'legal_names_sheet6': 0,
        'manual_matches': 0,
        'vat_codes': 0,
        'cb_with_domains': 0,
        'total_aliases': 0,
    }
    
    try:
        stats['platinum_matches'] = len(extract_platinum_matches(file_path))
    except: pass
    
    try:
        stats['legal_names_sheet4'] = len(extract_legal_names_sheet4(file_path))
    except: pass
    
    try:
        stats['legal_names_sheet6'] = len(extract_legal_names_sheet6(file_path))
    except: pass
    
    try:
        stats['manual_matches'] = len(extract_manual_matches(file_path))
    except: pass
    
    try:
        stats['vat_codes'] = len(extract_vat_codes(file_path))
    except: pass
    
    try:
        stats['cb_with_domains'] = len(extract_cb_companies_with_domains(file_path))
    except: pass
    
    stats['total_aliases'] = (
        stats['legal_names_sheet4'] + 
        stats['legal_names_sheet6'] + 
        stats['manual_matches']
    )
    
    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    file_path = '/Users/francescodelsesto/Downloads/ricerca/database-done.xlsx'
    
    print("\n" + "="*60)
    print("DATABASE-DONE.XLSX COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    stats = get_database_stats(file_path)
    
    print("\n📊 Available Data:")
    print(f"   Platinum matches: {stats['platinum_matches']:,}")
    print(f"   Legal names (Sheet 4): {stats['legal_names_sheet4']:,}")
    print(f"   Legal names (Sheet 6): {stats['legal_names_sheet6']:,}")
    print(f"   Manual matches: {stats['manual_matches']:,}")
    print(f"   VAT codes: {stats['vat_codes']:,}")
    print(f"   CB with domains: {stats['cb_with_domains']:,}")
    print(f"\n   TOTAL ALIASES: {stats['total_aliases']:,}")
    
    print("\n📈 Pre-matched Pairs:")
    pairs = get_all_prematched_pairs(file_path)
    print(f"   Total: {len(pairs):,}")
