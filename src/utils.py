"""
utils.py - Shared Utilities (Single Source of Truth)

Consolidates commonly used functions to avoid code duplication.
All domain/text normalization should use these functions.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
    logger.warning("tldextract not installed. Domain extraction will be limited.")

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False


# =============================================================================
# FREE EMAIL DOMAINS (Single Source of Truth)
# =============================================================================

FREE_EMAIL_DOMAINS = frozenset({
    'gmail.com', 'googlemail.com', 'yahoo.com', 'yahoo.co.uk', 'yahoo.fr',
    'yahoo.de', 'yahoo.it', 'yahoo.es', 'hotmail.com', 'hotmail.co.uk',
    'hotmail.fr', 'hotmail.it', 'outlook.com', 'outlook.fr', 'outlook.de',
    'live.com', 'live.co.uk', 'live.fr', 'msn.com', 'icloud.com', 'me.com',
    'mac.com', 'mail.com', 'aol.com', 'gmx.com', 'gmx.de', 'gmx.net',
    'web.de', 't-online.de', 'protonmail.com', 'proton.me', 'zoho.com',
    'yandex.com', 'yandex.ru', 'mail.ru', 'qq.com', '163.com', '126.com',
    'libero.it', 'virgilio.it', 'alice.it', 'tin.it', 'wanadoo.fr',
    'orange.fr', 'free.fr', 'sfr.fr', 'laposte.net',
})


# =============================================================================
# DOMAIN EXTRACTION (Robust)
# =============================================================================

def normalize_domain(url_or_email: str) -> Optional[str]:
    """
    Extract and normalize domain from URL or email.
    
    Handles:
    - http(s):// and ftp://
    - www. prefix
    - Port numbers
    - Query strings
    - Protocol-relative URLs (//)
    - Punycode domains
    
    Returns:
        Lowercase registrable domain (e.g., 'example.co.uk') or None
    """
    if not url_or_email or pd.isna(url_or_email):
        return None
    
    text = str(url_or_email).strip().lower()
    
    if not text:
        return None
    
    # Handle email addresses
    if '@' in text and '/' not in text:
        text = text.split('@')[-1]
    
    # Handle protocol-relative URLs
    if text.startswith('//'):
        text = 'https:' + text
    
    # Add protocol if missing (for urlparse)
    if not text.startswith(('http://', 'https://', 'ftp://')):
        text = 'https://' + text
    
    try:
        # Use tldextract if available (best for TLD handling)
        if HAS_TLDEXTRACT:
            ext = tldextract.extract(text)
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}"
            elif ext.domain:
                return ext.domain
            return None
        
        # Fallback: simple parsing
        parsed = urlparse(text)
        host = parsed.netloc or parsed.path.split('/')[0]
        
        # Remove port
        host = host.split(':')[0]
        
        # Remove www
        if host.startswith('www.'):
            host = host[4:]
        
        return host if host else None
        
    except Exception:
        return None


def is_free_email_domain(domain: str) -> bool:
    """Check if domain is a free email provider."""
    if not domain:
        return False
    return domain.lower() in FREE_EMAIL_DOMAINS


def extract_domain_features(url_or_email: str) -> Dict:
    """
    Extract comprehensive domain features.
    
    Returns dict with:
    - domain: registrable domain
    - is_free_email: boolean
    - is_valid: boolean
    - subdomain: subdomain if present
    """
    domain = normalize_domain(url_or_email)
    
    return {
        'domain': domain,
        'is_free_email': is_free_email_domain(domain) if domain else False,
        'is_valid': domain is not None,
    }


# =============================================================================
# TEXT NORMALIZATION (Unicode-aware)
# =============================================================================

def normalize_text_unicode(text: str) -> Dict:
    """
    Comprehensive Unicode text normalization.
    
    Returns dict with multiple representations:
    - norm: lowercase original
    - ascii: ASCII approximation (for matching)
    - has_non_latin: boolean
    """
    if not text or pd.isna(text):
        return {'norm': '', 'ascii': '', 'has_non_latin': False}
    
    text = str(text).strip()
    
    # NFD decomposition (separate base chars from diacritics)
    nfd = unicodedata.normalize('NFD', text)
    
    # Remove combining marks (diacritics)
    no_diacritics = ''.join(
        c for c in nfd 
        if unicodedata.category(c) != 'Mn'
    )
    
    # Check for non-Latin characters
    has_non_latin = bool(re.search(r'[^\x00-\x7F]', no_diacritics))
    
    # Full ASCII transliteration if available
    if HAS_UNIDECODE and has_non_latin:
        ascii_text = unidecode(text)
    else:
        ascii_text = no_diacritics
    
    return {
        'norm': text.lower(),
        'ascii': ascii_text.lower(),
        'has_non_latin': has_non_latin,
    }


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy matching.
    
    - Lowercase
    - Remove diacritics
    - Remove punctuation
    - Normalize whitespace
    """
    if not text or pd.isna(text):
        return ''
    
    text = str(text).strip().lower()
    
    # Remove diacritics
    nfd = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text for matching."""
    if not text:
        return []
    text = normalize_for_matching(text)
    return [t for t in text.split() if len(t) >= 2]


# =============================================================================
# DATE PARSING (Flexible)
# =============================================================================

def parse_date_flexible(date_str: str) -> Dict:
    """
    Parse dates with multiple format support.
    
    Handles:
    - ISO format: 2023-01-15
    - Year only: 2023
    - Quarter: Q1 2022
    - Month-year: March 2020
    - Various null values
    
    Returns:
        {'year': int|None, 'date': str|None, 'confidence': float}
    """
    if not date_str or pd.isna(date_str):
        return {'year': None, 'date': None, 'confidence': 0}
    
    date_str = str(date_str).strip()
    
    # Handle null-ish values
    if date_str.lower() in ['n/a', 'na', '-', 'unknown', 'none', '', 'null']:
        return {'year': None, 'date': None, 'confidence': 0}
    
    # Year only (common case)
    if re.match(r'^\d{4}$', date_str):
        year = int(date_str)
        if 1800 <= year <= 2100:
            return {'year': year, 'date': None, 'confidence': 0.8}
    
    # Quarter format: Q1 2022
    q_match = re.match(r'^Q([1-4])\s*(\d{4})$', date_str, re.I)
    if q_match:
        quarter, year = int(q_match.group(1)), int(q_match.group(2))
        return {'year': year, 'date': None, 'confidence': 0.7}
    
    # Try pandas date parsing
    try:
        parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed):
            return {
                'year': parsed.year,
                'date': parsed.strftime('%Y-%m-%d'),
                'confidence': 0.9
            }
    except:
        pass
    
    # Extract any 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return {'year': int(year_match.group()), 'date': None, 'confidence': 0.5}
    
    return {'year': None, 'date': None, 'confidence': 0}


# =============================================================================
# LEGAL SUFFIX HANDLING
# =============================================================================

# Comprehensive legal suffixes by country
LEGAL_SUFFIXES = [
    # English
    r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|llp|plc|co)\b\.?',
    # German
    r'\b(gmbh|ag|kg|ohg|gbr|ug|e\.?v\.?|mbh)\b\.?',
    # French
    r'\b(sa|sarl|sas|sasu|eurl|snc|sca)\b\.?',
    # Italian
    r'\b(spa|s\.p\.a|srl|s\.r\.l|sapa|snc|sas)\b\.?',
    # Spanish
    r'\b(sl|s\.l|sa|s\.a|slne|slu)\b\.?',
    # Dutch/Belgian
    r'\b(bv|b\.v|nv|n\.v|vof|cv)\b\.?',
    # Nordic
    r'\b(ab|as|a/s|aps|oy|oyj)\b\.?',
    # Other
    r'\b(pty|pte|bhd|sdn|kk|kabushiki)\b\.?',
]

LEGAL_SUFFIX_PATTERN = re.compile(
    '|'.join(LEGAL_SUFFIXES), 
    re.IGNORECASE
)


def strip_legal_suffix(name: str) -> str:
    """Remove legal suffixes from company name."""
    if not name:
        return ''
    
    # Normalize first
    name = str(name).strip()
    
    # Remove suffixes (iterate in case of multiple)
    for _ in range(3):  # Max 3 passes
        new_name = LEGAL_SUFFIX_PATTERN.sub('', name).strip()
        new_name = re.sub(r'[,\.]+$', '', new_name).strip()  # Trailing punctuation
        if new_name == name:
            break
        name = new_name
    
    return name


# =============================================================================
# SIMILARITY FUNCTIONS (Single Implementations)
# =============================================================================

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaro-Winkler similarity.
    Uses RapidFuzz if available, otherwise falls back to basic Jaro.
    """
    if not s1 or not s2:
        return 0.0
    
    s1 = normalize_for_matching(s1)
    s2 = normalize_for_matching(s2)
    
    if s1 == s2:
        return 1.0
    
    try:
        from rapidfuzz import distance
        return distance.JaroWinkler.similarity(s1, s2)
    except ImportError:
        pass
    
    # Basic Jaro implementation
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(0, match_distance)
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3
    
    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + prefix * 0.1 * (1 - jaro)


def token_jaccard(s1: str, s2: str) -> float:
    """Compute Jaccard similarity on token sets."""
    tokens1 = set(tokenize(s1))
    tokens2 = set(tokenize(s2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


# =============================================================================
# KNOWN REBRANDS
# =============================================================================

KNOWN_REBRANDS = {
    # old_name_normalized: new_name_normalized
    'facebook': 'meta platforms',
    'twitter': 'x corp',
    'google': 'alphabet',  # Parent company
    'fiat chrysler': 'stellantis',
    'raytheon': 'rtx corporation',
    'hewlett packard': 'hp',
    'philip morris': 'altria',
}


def get_rebrand_aliases(name: str) -> Set[str]:
    """Get known rebrand aliases for a company name."""
    name_norm = normalize_for_matching(name)
    aliases = {name_norm}
    
    # Check if this name is a known old name
    if name_norm in KNOWN_REBRANDS:
        aliases.add(KNOWN_REBRANDS[name_norm])
    
    # Check if this name is a known new name
    for old, new in KNOWN_REBRANDS.items():
        if name_norm == new or name_norm in new or new in name_norm:
            aliases.add(old)
    
    return aliases
