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
    
    Only 314 have actual company name matches.
    """
    df = pd.read_excel(file_path, sheet_name="Matching manuale")
    
    result = pd.DataFrame({
        'domain': df['dominio'],
        'cb_name': df['name '],
        'legal_name': df['company name'],
    })
    
    # Only rows with actual company names
    result = result[result['legal_name'].notna()]
    
    logger.info(f"Extracted {len(result)} manual matches")
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
