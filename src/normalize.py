"""
normalize.py - Name Normalization Module

Multi-view name normalization for entity resolution.
Handles: legal suffix stripping, diacritics, tokenization, fingerprinting.
"""

import re
import unicodedata
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LEGAL SUFFIX PATTERNS (Multi-language)
# =============================================================================

# Common legal suffixes by country/language
LEGAL_SUFFIXES = {
    # English
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'llc', 'llp', 'lp', 'plc', 'pvt',
    
    # German
    'gmbh', 'ag', 'kg', 'ohg', 'gbr', 'ug', 'mbh', 'e v', 'ev',
    
    # French
    'sa', 'sarl', 'sas', 'sasu', 'snc', 'sca', 'sci', 'eurl',
    
    # Italian
    'spa', 's p a', 'srl', 's r l', 'sas', 'snc', 'sapa',
    
    # Spanish
    'sl', 's l', 'sa', 'slu', 'sau', 'sc', 'coop',
    
    # Dutch/Belgian
    'bv', 'b v', 'nv', 'n v', 'vof', 'cv', 'cvba',
    
    # Nordic
    'ab', 'a b', 'as', 'a s', 'asa', 'aps', 'a/s', 'oy', 'oyj',
    
    # Eastern European
    'sp z o o', 'spzoo', 'zrt', 'kft', 'bt', 'sro', 's r o', 'as', 'sia', 'uab',
    
    # Other
    'pty', 'pty ltd', 'pte', 'pte ltd', 'bhd', 'sdn bhd',
    'holding', 'holdings', 'group', 'international', 'intl',
}

# Pattern to match legal suffixes at end of name
LEGAL_SUFFIX_PATTERN = re.compile(
    r'\s*[,.]?\s*(' + '|'.join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r')\s*\.?\s*$',
    re.IGNORECASE
)


# =============================================================================
# GENERIC NAME DETECTION
# =============================================================================

# Names that are too generic to match reliably without other signals
GENERIC_TOKENS = {
    'global', 'international', 'services', 'solutions', 'systems',
    'consulting', 'partners', 'associates', 'group', 'holdings',
    'technologies', 'technology', 'tech', 'digital', 'media',
    'capital', 'ventures', 'investments', 'management', 'enterprise',
    'industries', 'trading', 'commerce', 'logistics', 'software',
    'europe', 'european', 'asia', 'pacific', 'america', 'americas',
    'first', 'one', 'new', 'next', 'pioneer', 'united',
}


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def remove_diacritics(text: str) -> str:
    """Remove diacritics (accents) from text: é → e, ü → u, etc."""
    if not text:
        return text
    # Normalize to NFD (decomposed form), then remove combining chars
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip edges."""
    if not text:
        return text
    return ' '.join(text.split())


def remove_punctuation(text: str, keep_chars: str = '') -> str:
    """Remove punctuation, optionally keeping specific chars."""
    if not text:
        return text
    # Keep alphanumeric, spaces, and specified chars
    pattern = r'[^a-zA-Z0-9\s' + re.escape(keep_chars) + r']'
    return re.sub(pattern, ' ', text)


def strip_legal_suffix(name: str) -> str:
    """
    Remove common legal suffixes from company name.
    Iteratively removes suffixes until none remain.
    """
    if not name:
        return name
    
    result = name.strip()
    iterations = 0
    max_iterations = 5  # Prevent infinite loops
    
    while iterations < max_iterations:
        match = LEGAL_SUFFIX_PATTERN.search(result)
        if match:
            result = result[:match.start()].strip()
            iterations += 1
        else:
            break
    
    # Also strip trailing punctuation
    result = re.sub(r'[\s,.\-]+$', '', result)
    
    return result.strip() if result else name


def tokenize_name(name: str) -> List[str]:
    """Tokenize name into list of tokens."""
    if not name:
        return []
    # Split on whitespace and filter empty tokens
    return [t for t in name.lower().split() if t]


def compute_fingerprint(name: str) -> str:
    """
    Compute name fingerprint: sorted unique tokens joined.
    This is order-independent for matching "ABC Inc" with "Inc ABC".
    """
    tokens = tokenize_name(name)
    # Remove very short tokens (likely initials/articles)
    tokens = [t for t in tokens if len(t) > 1]
    # Sort and dedupe
    return ' '.join(sorted(set(tokens)))


def extract_acronym(name: str) -> str:
    """Extract acronym from name (first letter of each word)."""
    tokens = tokenize_name(name)
    if not tokens:
        return ''
    # Take first letter of each token that starts with a letter
    return ''.join(t[0] for t in tokens if t and t[0].isalpha()).upper()


def is_generic_name(name: str) -> bool:
    """
    Check if name is too generic to match reliably.
    A name is generic if:
    - It's very short (< 3 chars after cleaning)
    - It consists only of generic tokens
    """
    if not name:
        return True
    
    tokens = tokenize_name(strip_legal_suffix(name.lower()))
    
    # Too short
    if len(tokens) == 0:
        return True
    
    # All tokens are generic
    non_generic = [t for t in tokens if t not in GENERIC_TOKENS and len(t) > 2]
    
    return len(non_generic) == 0


def normalize_name(name: str) -> Dict[str, any]:
    """
    Full multi-view name normalization.
    
    Returns dict with:
        - name_norm: lowercase, no punct, no diacritics
        - name_legal_stripped: without legal suffixes
        - name_tokens: list of tokens
        - name_fingerprint: sorted token join
        - name_prefix8: first 8 chars of legal_stripped
        - name_acronym: first letters
        - name_is_generic: too generic flag
    """
    if not name or (isinstance(name, float) and str(name) == 'nan'):
        return {
            'name_norm': '',
            'name_legal_stripped': '',
            'name_tokens': [],
            'name_fingerprint': '',
            'name_prefix8': '',
            'name_acronym': '',
            'name_is_generic': True,
        }
    
    name = str(name)
    
    # Step 1: Basic normalization
    name_norm = remove_diacritics(name)
    name_norm = remove_punctuation(name_norm)
    name_norm = normalize_whitespace(name_norm)
    name_norm = name_norm.lower()
    
    # Step 2: Strip legal suffixes
    name_legal = strip_legal_suffix(name_norm)
    
    # Step 3: Tokenize
    tokens = tokenize_name(name_legal)
    
    # Step 4: Fingerprint
    fingerprint = compute_fingerprint(name_legal)
    
    # Step 5: Prefix (first 8 chars, useful for blocking)
    prefix8 = name_legal.replace(' ', '')[:8] if name_legal else ''
    
    # Step 6: Acronym
    acronym = extract_acronym(name_legal)
    
    # Step 7: Generic check
    is_generic = is_generic_name(name)
    
    return {
        'name_norm': name_norm,
        'name_legal_stripped': name_legal,
        'name_tokens': tokens,
        'name_fingerprint': fingerprint,
        'name_prefix8': prefix8,
        'name_acronym': acronym,
        'name_is_generic': is_generic,
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def normalize_name_column(names: 'pd.Series', chunk_size: int = 500000) -> 'pd.DataFrame':
    """
    VECTORIZED name normalization using pandas string operations.
    50-100x faster than apply() for large datasets.
    """
    import pandas as pd
    import numpy as np
    
    n_rows = len(names)
    logger.info(f"Normalizing {n_rows:,} names (vectorized)...")
    
    # Convert to string and lowercase
    s = names.fillna('').astype(str).str.lower().str.strip()
    
    # Step 1: Remove diacritics (vectorized with numpy)
    # This is the only slow part - use chunk processing
    def remove_diacritics_vec(series):
        import unicodedata
        return series.apply(lambda x: ''.join(
            c for c in unicodedata.normalize('NFKD', x) if not unicodedata.combining(c)
        ) if x else '')
    
    # Process in chunks for diacritics (the only non-vectorizable part)
    if n_rows > chunk_size:
        from tqdm import tqdm
        chunks = []
        for i in tqdm(range(0, n_rows, chunk_size), desc="Removing diacritics", unit="chunk"):
            chunks.append(remove_diacritics_vec(s.iloc[i:i+chunk_size]))
        name_norm = pd.concat(chunks, ignore_index=True)
    else:
        name_norm = remove_diacritics_vec(s)
    
    # Step 2: Remove punctuation (vectorized)
    name_norm = name_norm.str.replace(r'[^\w\s]', ' ', regex=True)
    name_norm = name_norm.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Step 3: Strip legal suffixes (vectorized regex)
    # Build pattern for all legal suffixes
    suffixes_pattern = r'\s*[,.]?\s*(' + '|'.join([
        'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
        'ltd', 'limited', 'llc', 'llp', 'lp', 'plc', 'pvt',
        'gmbh', 'ag', 'kg', 'gbr', 'ug', 'mbh',
        'sa', 'sarl', 'sas', 'sasu', 'snc', 'eurl',
        'spa', 'srl', 'snc',
        'sl', 'sau', 'coop',
        'bv', 'nv', 'vof', 'cv',
        'ab', 'as', 'asa', 'aps', 'oy', 'oyj',
        'holding', 'holdings', 'group', 'international'
    ]) + r')\s*\.?\s*$'
    
    name_legal = name_norm.str.replace(suffixes_pattern, '', regex=True, case=False)
    name_legal = name_legal.str.replace(r'[\s,.\-]+$', '', regex=True).str.strip()
    
    # Step 4: Prefix (first 8 chars, no spaces)
    name_prefix8 = name_legal.str.replace(' ', '').str[:8]
    
    # Step 5: Fingerprint (sorted unique tokens)
    # This is vectorized but needs list comprehension for sorting
    def compute_fingerprint_vec(series):
        return series.apply(lambda x: ' '.join(sorted(set(x.split()))) if x else '')
    
    if n_rows > chunk_size:
        chunks = []
        for i in tqdm(range(0, n_rows, chunk_size), desc="Computing fingerprints", unit="chunk"):
            chunks.append(compute_fingerprint_vec(name_legal.iloc[i:i+chunk_size]))
        name_fingerprint = pd.concat(chunks, ignore_index=True)
    else:
        name_fingerprint = compute_fingerprint_vec(name_legal)
    
    # Step 6: Acronym (first letter of each word)
    name_acronym = name_legal.str.split().apply(
        lambda tokens: ''.join(t[0].upper() for t in tokens if t) if tokens else ''
    )
    
    # Step 7: Generic name detection (vectorized)
    generic_tokens = {'global', 'international', 'services', 'solutions', 'systems',
                     'consulting', 'partners', 'associates', 'group', 'holdings',
                     'technologies', 'technology', 'tech', 'digital', 'media'}
    
    def is_generic_vec(series):
        def check(name):
            if not name or len(name) < 3:
                return True
            tokens = [t for t in name.split() if len(t) > 2 and t not in generic_tokens]
            return len(tokens) == 0
        return series.apply(check)
    
    name_is_generic = is_generic_vec(name_legal)
    
    # Build result DataFrame
    result = pd.DataFrame({
        'name_norm': name_norm,
        'name_legal_stripped': name_legal,
        'name_fingerprint': name_fingerprint,
        'name_prefix8': name_prefix8,
        'name_acronym': name_acronym,
        'name_is_generic': name_is_generic,
    })
    
    logger.info(f"Normalization complete: {n_rows:,} rows")
    
    return result


if __name__ == '__main__':
    # Test cases
    test_names = [
        "KJELL GROUP AB",
        "Delivery Hero SE",
        "Société Générale S.A.",
        "MÜLLER GmbH & Co. KG",
        "Global Services International Holdings Ltd.",
        "A",  # Too short
        "",   # Empty
    ]
    
    print("=== Name Normalization Test ===\n")
    for name in test_names:
        result = normalize_name(name)
        print(f"Input: '{name}'")
        print(f"  norm: '{result['name_norm']}'")
        print(f"  legal_stripped: '{result['name_legal_stripped']}'")
        print(f"  fingerprint: '{result['name_fingerprint']}'")
        print(f"  prefix8: '{result['name_prefix8']}'")
        print(f"  acronym: '{result['name_acronym']}'")
        print(f"  is_generic: {result['name_is_generic']}")
        print()
