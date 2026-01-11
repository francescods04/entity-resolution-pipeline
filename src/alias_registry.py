"""
alias_registry.py - Alias & Legal Name Registry

Manages the mapping between Crunchbase brand names and legal entity names.
Integrates existing work from database-done.xlsx to create a comprehensive
alias registry for entity matching.

KEY INSIGHT:
-----------
A company like "Deliveroo" (CB brand name) might be legally registered as:
- "Deliveroo Holdings PLC" (UK parent)
- "Deliveroo Italy S.r.l." (Italian subsidiary)
- "Deliveroo France SAS" (French subsidiary)

This module tracks these relationships and uses them during matching.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ALIAS REGISTRY STRUCTURE
# =============================================================================

class AliasRegistry:
    """
    Registry mapping brand names to known legal entity names.
    
    Structure:
        - cb_id → [brand_name, legal_name_1, legal_name_2, ...]
        - domain → cb_id (for domain-based lookup)
        - legal_name_norm → cb_id (for reverse lookup)
    """
    
    def __init__(self):
        # Primary mapping: cb_id → set of all known names
        self.cb_id_to_names: Dict[str, Set[str]] = {}
        
        # Domain mapping: domain → cb_id
        self.domain_to_cb_id: Dict[str, str] = {}
        
        # Reverse mapping: normalized_name → set of cb_ids (for collision detection)
        self.name_to_cb_ids: Dict[str, Set[str]] = {}
        
        # Legal name to CB mapping: legal_name_norm → (cb_id, confidence)
        self.legal_to_cb: Dict[str, Tuple[str, float]] = {}
        
        # VAT code mapping: vat_code → cb_id (UNIQUE IDENTIFIERS!)
        self.vat_to_cb_id: Dict[str, str] = {}
        
        # CB to VAT reverse lookup
        self.cb_id_to_vat: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'total_cb_ids': 0,
            'total_aliases': 0,
            'name_collisions': 0,
            'vat_codes': 0,
        }
    
    def add_alias(self, cb_id: str, name: str, confidence: float = 1.0) -> None:
        """Add a name alias for a cb_id."""
        if not cb_id or not name:
            return
        
        name_norm = self._normalize_name(name)
        if not name_norm:
            return
        
        # Add to cb_id → names mapping
        if cb_id not in self.cb_id_to_names:
            self.cb_id_to_names[cb_id] = set()
            self.stats['total_cb_ids'] += 1
        
        if name_norm not in self.cb_id_to_names[cb_id]:
            self.cb_id_to_names[cb_id].add(name_norm)
            self.stats['total_aliases'] += 1
        
        # Add to reverse mapping
        if name_norm not in self.name_to_cb_ids:
            self.name_to_cb_ids[name_norm] = set()
        self.name_to_cb_ids[name_norm].add(cb_id)
        
        # Track collisions
        if len(self.name_to_cb_ids[name_norm]) > 1:
            self.stats['name_collisions'] += 1
        
        # Add to legal name mapping if high confidence
        if confidence >= 0.9:
            existing = self.legal_to_cb.get(name_norm)
            if existing is None or existing[1] < confidence:
                self.legal_to_cb[name_norm] = (cb_id, confidence)
    
    def add_domain(self, cb_id: str, domain: str) -> None:
        """Map a domain to a cb_id."""
        if not cb_id or not domain:
            return
        
        domain_norm = self._normalize_domain(domain)
        if domain_norm:
            self.domain_to_cb_id[domain_norm] = cb_id
    
    def get_names_for_cb(self, cb_id: str) -> Set[str]:
        """Get all known names for a cb_id."""
        return self.cb_id_to_names.get(cb_id, set())
    
    def get_cb_for_name(self, name: str) -> Set[str]:
        """Get cb_ids that match a name (could be multiple for collisions)."""
        name_norm = self._normalize_name(name)
        return self.name_to_cb_ids.get(name_norm, set())
    
    def get_cb_for_domain(self, domain: str) -> Optional[str]:
        """Get cb_id for a domain."""
        domain_norm = self._normalize_domain(domain)
        return self.domain_to_cb_id.get(domain_norm)
    
    def is_name_collision(self, name: str) -> bool:
        """Check if a name maps to multiple cb_ids."""
        name_norm = self._normalize_name(name)
        cb_ids = self.name_to_cb_ids.get(name_norm, set())
        return len(cb_ids) > 1
    
    def add_vat(self, cb_id: str, vat_code: str) -> None:
        """Add a VAT code mapping for a cb_id. VAT codes are UNIQUE identifiers!"""
        if not cb_id or not vat_code:
            return
        
        vat_norm = self._normalize_vat(vat_code)
        if vat_norm:
            self.vat_to_cb_id[vat_norm] = cb_id
            self.cb_id_to_vat[cb_id] = vat_norm
            self.stats['vat_codes'] += 1
    
    def get_cb_for_vat(self, vat_code: str) -> Optional[str]:
        """Get cb_id for a VAT code (exact match = Tier A confidence!)."""
        vat_norm = self._normalize_vat(vat_code)
        return self.vat_to_cb_id.get(vat_norm)
    
    def get_vat_for_cb(self, cb_id: str) -> Optional[str]:
        """Get VAT code for a cb_id."""
        return self.cb_id_to_vat.get(cb_id)
    
    def _normalize_vat(self, vat: str) -> str:
        """Normalize VAT code for matching."""
        if not vat or pd.isna(vat):
            return ''
        # Remove spaces, convert to uppercase
        vat = str(vat).strip().upper()
        vat = re.sub(r'\s+', '', vat)
        return vat
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for registry lookup."""
        if not name or pd.isna(name):
            return ''
        name = str(name).strip().lower()
        # Remove punctuation except spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        # Normalize whitespace
        name = ' '.join(name.split())
        return name
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain for registry lookup."""
        if not domain or pd.isna(domain):
            return ''
        domain = str(domain).strip().lower()
        # Remove protocol
        domain = re.sub(r'^https?://', '', domain)
        # Remove www
        domain = re.sub(r'^www\.', '', domain)
        # Remove path
        domain = domain.split('/')[0]
        return domain
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        return {
            **self.stats,
            'domains_indexed': len(self.domain_to_cb_id),
            'unique_names': len(self.name_to_cb_ids),
            'vat_codes_indexed': len(self.vat_to_cb_id),
        }


# =============================================================================
# PRE-MATCHED PAIRS FROM database-done.xlsx
# =============================================================================

def load_prematched_bvd_pairs(file_path: str) -> pd.DataFrame:
    """
    Load pre-matched CB→BVD pairs from Sheet 1 ('bvd id').
    
    These are HIGH-CONFIDENCE matches that should SKIP blocking entirely
    and go directly to feature computation.
    
    Returns:
        DataFrame with columns: cb_name, bvd_id, legal_name, confidence
    """
    xls = pd.ExcelFile(file_path)
    
    # Sheet 1: 'bvd id' contains direct CB→Orbis mappings
    df = pd.read_excel(xls, sheet_name=1)
    
    # Clean and filter
    df = df.dropna(subset=['company name', 'bvd id'])
    
    # Normalize
    result = pd.DataFrame({
        'cb_name': df['company name'].astype(str).str.strip(),
        'bvd_id': df['bvd id'].astype(str).str.strip(),
        'legal_name': df['legal name'].fillna('').astype(str).str.strip(),
        'confidence': 0.99,  # High confidence pre-match
        'source': 'prematched_bvdid'
    })
    
    # Filter valid BVD IDs (should be alphanumeric)
    result = result[result['bvd_id'].str.len() > 0]
    
    logger.info(f"Loaded {len(result)} pre-matched BVD pairs from Sheet 1")
    
    return result


def load_platinum_pairs(file_path: str) -> pd.DataFrame:
    """
    Load platinum matches from Sheet 2 as additional high-confidence pairs.
    
    Returns:
        DataFrame with columns: cb_name, orbis_name, confidence, match_type
    """
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=2)
    
    # Clean and filter
    df = df.dropna(subset=['nome_df2_match', 'nome_df1'])
    
    result = pd.DataFrame({
        'cb_name': df['nome_df2_match'].astype(str).str.strip(),
        'orbis_name': df['nome_df1'].astype(str).str.strip(),
        'match_type': df['match_type'].fillna('').astype(str),
        'confidence': df['match_type'].apply(
            lambda x: 0.99 if 'platinum' in str(x).lower() else 0.90
        ),
        'source': 'platinum_match'
    })
    
    logger.info(f"Loaded {len(result)} platinum pairs from Sheet 2")
    
    return result


# =============================================================================
# LOADING FROM database-done.xlsx
# =============================================================================

def load_alias_registry_from_excel(file_path: str) -> AliasRegistry:
    """
    Load and integrate all alias sources from database-done.xlsx
    
    Sources integrated:
    1. Sheet 'DB company crunchbase': Base CB companies with domains
    2. Sheet 'bvd id': CB brand → Legal name mappings
    3. Sheet 'Matching 1 platinum': Domain-verified matches (high confidence)
    4. Sheet 'Matching 2': AI-scraped legal names from websites
    5. Sheet 'Matching manuale': Manual legal name discovery
    6. Sheet 'Matching ai': AI-extracted legal names
    """
    registry = AliasRegistry()
    xls = pd.ExcelFile(file_path)
    
    logger.info("Loading alias registry from database-done.xlsx")
    
    # 1. Load base CB companies (Sheet 0: 'DB company crunchbase')
    logger.info("  Loading base CB companies...")
    df_base = pd.read_excel(xls, sheet_name=0)
    # OPTIMIZED: itertuples is 10-100x faster than iterrows
    for row in df_base.itertuples(index=False):
        name = getattr(row, 'Organization Name', None) if hasattr(row, 'Organization Name') else row[0]
        website = getattr(row, 'Organization Website', None) if hasattr(row, 'Organization Website') else (row[1] if len(row) > 1 else None)
        
        if pd.notna(name):
            cb_id = str(name).lower().replace(' ', '-')
            registry.add_alias(cb_id, name, confidence=1.0)
            
            if pd.notna(website):
                registry.add_domain(cb_id, website)
    
    logger.info(f"    Loaded {len(df_base)} base companies")
    
    # 2. Load legal name mappings (Sheet 1: 'bvd id') - OPTIMIZED
    logger.info("  Loading legal name mappings...")
    df_legal = pd.read_excel(xls, sheet_name=1)
    cols = ['company name', 'legal name', 'website']
    for col in cols:
        if col not in df_legal.columns:
            df_legal[col] = None
    for brand_name, legal_name, website in zip(df_legal['company name'], df_legal['legal name'], df_legal['website']):
        if pd.notna(brand_name):
            cb_id = str(brand_name).lower().replace(' ', '-')
            registry.add_alias(cb_id, brand_name, confidence=1.0)
            if pd.notna(legal_name):
                registry.add_alias(cb_id, legal_name, confidence=0.95)
            if pd.notna(website):
                registry.add_domain(cb_id, website)
    logger.info(f"    Loaded {len(df_legal)} legal name mappings")
    
    # 3. Load platinum matches (Sheet 2) - OPTIMIZED
    logger.info("  Loading platinum domain matches...")
    df_platinum = pd.read_excel(xls, sheet_name=2)
    for orbis_name, cb_name, match_type in zip(df_platinum['nome_df1'], df_platinum['nome_df2_match'], df_platinum['match_type']):
        if pd.notna(cb_name) and pd.notna(orbis_name):
            cb_id = str(cb_name).lower().replace(' ', '-')
            confidence = 0.99 if 'platinum' in str(match_type).lower() else 0.90
            registry.add_alias(cb_id, cb_name, confidence=1.0)
            registry.add_alias(cb_id, orbis_name, confidence=confidence)
    logger.info(f"    Loaded {len(df_platinum)} platinum matches")
    
    # 4. Load AI-scraped legal names (Sheet 3) - OPTIMIZED
    logger.info("  Loading AI-scraped legal names + VAT codes...")
    df_scrape = pd.read_excel(xls, sheet_name=3)
    cols = ['dominio', 'json_final_company_name', 'json_final_vat_code']
    for col in cols:
        if col not in df_scrape.columns:
            df_scrape[col] = None
    vat_count = 0
    for domain, legal_name, vat_code in zip(df_scrape['dominio'], df_scrape['json_final_company_name'], df_scrape['json_final_vat_code']):
        if pd.notna(domain) and pd.notna(legal_name):
            cb_id = registry.get_cb_for_domain(domain)
            if cb_id:
                registry.add_alias(cb_id, legal_name, confidence=0.85)
            else:
                cb_id = str(domain).replace('.', '-')
                registry.add_alias(cb_id, legal_name, confidence=0.85)
                registry.add_domain(cb_id, domain)
            if pd.notna(vat_code) and str(vat_code).strip():
                registry.add_vat(cb_id, vat_code)
                vat_count += 1
    logger.info(f"    Processed {len(df_scrape)} AI-scraped entries")
    logger.info(f"    Extracted {vat_count} VAT codes (Tier A identifiers)")
    
    # 5. Load manual matches (Sheet 4) - OPTIMIZED
    logger.info("  Loading manual matches...")
    df_manual = pd.read_excel(xls, sheet_name=4)
    cols = ['dominio', 'name ', 'company name']
    for col in cols:
        if col not in df_manual.columns:
            df_manual[col] = None
    for domain, brand_name, legal_desc in zip(df_manual['dominio'], df_manual['name '], df_manual['company name']):
        if pd.notna(brand_name):
            cb_id = str(brand_name).lower().replace(' ', '-')
            registry.add_alias(cb_id, brand_name, confidence=1.0)
            if pd.notna(legal_desc):
                legal_name = _extract_legal_name_from_desc(str(legal_desc))
                if legal_name:
                    registry.add_alias(cb_id, legal_name, confidence=0.92)
            if pd.notna(domain):
                registry.add_domain(cb_id, domain)
    logger.info(f"    Loaded {len(df_manual)} manual matches")
    
    # 6. Load AI extraction (Sheet 5) - OPTIMIZED
    logger.info("  Loading AI legal name extractions...")
    df_ai = pd.read_excel(xls, sheet_name=5)
    cols = ['Name', 'legal_name', 'Website']
    for col in cols:
        if col not in df_ai.columns:
            df_ai[col] = None
    for brand_name, legal_name, website in zip(df_ai['Name'], df_ai['legal_name'], df_ai['Website']):
        if pd.notna(brand_name) and pd.notna(legal_name):
            if 'not available' in str(legal_name).lower() or 'unable to' in str(legal_name).lower():
                continue
            cb_id = str(brand_name).lower().replace(' ', '-')
            registry.add_alias(cb_id, brand_name, confidence=1.0)
            registry.add_alias(cb_id, legal_name, confidence=0.88)
            if pd.notna(website):
                registry.add_domain(cb_id, website)
    logger.info(f"    Processed {len(df_ai)} AI extractions")
    
    # Final stats
    stats = registry.get_stats()
    logger.info(f"\n  Registry built:")
    logger.info(f"    - {stats['total_cb_ids']:,} CB companies")
    logger.info(f"    - {stats['total_aliases']:,} total aliases")
    logger.info(f"    - {stats['domains_indexed']:,} domains indexed")
    logger.info(f"    - {stats['name_collisions']:,} name collisions detected")
    
    return registry


def _extract_legal_name_from_desc(desc: str) -> Optional[str]:
    """
    Extract legal entity name from description text.
    
    Examples:
        "WOODFLOW TECHNOLOGIES S.L., registered in Madrid" → "WOODFLOW TECHNOLOGIES S.L."
        "The legal entity is SoCyber Ltd., a limited liability..." → "SoCyber Ltd."
    """
    if not desc:
        return None
    
    # Pattern 1: Name followed by legal suffix and comma
    patterns = [
        r'^([A-Z][A-Z\s&\.\-]+(?:S\.?L\.?|SRL|GMBH|LTD\.?|INC\.?|AG|SA|SAS|AB|OY|BV|NV))',
        r'legal entity is\s+([^,\.]+(?:Ltd|GmbH|SRL|S\.L\.|Inc|AG|SA|SAS)\.?)',
        r'company is\s+([^,]+(?:Ltd|GmbH|SRL|S\.L\.|Inc|AG|SA|SAS)\.?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, desc, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


# =============================================================================
# DISAMBIGUATION FOR NAME COLLISIONS
# =============================================================================

def disambiguate_name_collision(
    name: str,
    candidates: List[Dict],
    cb_country: Optional[str] = None,
    cb_city: Optional[str] = None,
    cb_founded_year: Optional[int] = None,
    cb_domain: Optional[str] = None,
) -> List[Tuple[Dict, float]]:
    """
    Disambiguate when multiple Orbis records have the same name.
    
    This is CRITICAL for names like "B & C SRL" that appear hundreds of times.
    
    Args:
        name: The company name being matched
        candidates: List of Orbis candidate records
        cb_country: Crunchbase country ISO code
        cb_city: Crunchbase city (normalized)
        cb_founded_year: Crunchbase founded year
        cb_domain: Crunchbase domain
    
    Returns:
        List of (candidate, disambiguation_score) sorted by score descending
    """
    if not candidates:
        return []
    
    scored = []
    
    for cand in candidates:
        score = 0.0
        signals = []
        
        # Signal 1: Domain match (STRONGEST - essentially resolves the collision)
        cand_domain = cand.get('orbis_domain', '')
        if cb_domain and cand_domain:
            if _domains_match(cb_domain, cand_domain):
                score += 100.0
                signals.append('domain_exact')
        
        # Signal 2: Country match (STRONG)
        cand_country = cand.get('orbis_country', '')
        if cb_country and cand_country:
            if str(cb_country).upper() == str(cand_country).upper():
                score += 30.0
                signals.append('country_match')
            else:
                score -= 50.0  # Penalty for country mismatch
                signals.append('country_mismatch')
        
        # Signal 3: City match (MEDIUM)
        cand_city = cand.get('orbis_city', '')
        if cb_city and cand_city:
            city_sim = _city_similarity(cb_city, cand_city)
            if city_sim > 0.8:
                score += 20.0
                signals.append('city_match')
            elif city_sim > 0.5:
                score += 10.0
                signals.append('city_partial')
        
        # Signal 4: Founded year proximity (WEAK but useful)
        cand_year = cand.get('orbis_incorp_year')
        if cb_founded_year and cand_year:
            year_diff = abs(int(cb_founded_year) - int(cand_year))
            if year_diff == 0:
                score += 10.0
                signals.append('year_exact')
            elif year_diff <= 2:
                score += 5.0
                signals.append('year_close')
            elif year_diff > 10:
                score -= 10.0
                signals.append('year_far')
        
        # Signal 5: Company size/activity indicators
        # Active companies more likely to be VC-backed startups
        cand_status = cand.get('orbis_status', '')
        if 'active' in str(cand_status).lower():
            score += 5.0
        
        scored.append((cand, score, signals))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Return as (candidate, score) pairs
    return [(c, s) for c, s, _ in scored]


def _domains_match(d1: str, d2: str) -> bool:
    """Check if two domains match (normalized)."""
    def norm(d):
        d = str(d).lower().strip()
        d = re.sub(r'^https?://', '', d)
        d = re.sub(r'^www\.', '', d)
        d = d.split('/')[0]
        return d
    
    return norm(d1) == norm(d2)


def _city_similarity(c1: str, c2: str) -> float:
    """Compute city name similarity (0-1 scale)."""
    if not c1 or not c2:
        return 0.0
    
    c1 = str(c1).lower().strip()
    c2 = str(c2).lower().strip()
    
    if c1 == c2:
        return 1.0
    
    # Simple token overlap
    tokens1 = set(c1.split())
    tokens2 = set(c2.split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    overlap = len(tokens1 & tokens2)
    total = len(tokens1 | tokens2)
    
    return overlap / total if total > 0 else 0.0


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def save_registry_to_parquet(registry: AliasRegistry, output_path: str) -> None:
    """Save alias registry to Parquet for efficient lookup."""
    records = []
    
    for cb_id, names in registry.cb_id_to_names.items():
        domain = None
        for d, cid in registry.domain_to_cb_id.items():
            if cid == cb_id:
                domain = d
                break
        
        records.append({
            'cb_id': cb_id,
            'names': '|'.join(sorted(names)),
            'name_count': len(names),
            'domain': domain,
            'has_collision': any(
                len(registry.name_to_cb_ids.get(n, set())) > 1 
                for n in names
            ),
        })
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved registry with {len(df)} entries to {output_path}")


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Test loading
    file_path = "/Users/francescodelsesto/Downloads/ricerca/database-done.xlsx"
    registry = load_alias_registry_from_excel(file_path)
    
    print("\n=== Registry Stats ===")
    for k, v in registry.get_stats().items():
        print(f"  {k}: {v:,}")
