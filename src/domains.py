"""
domains.py - Domain Extraction Module

Extract and normalize domains from URLs and email addresses.
Handles: protocol stripping, www removal, eTLD+1 extraction, free email detection.
"""

import re
from typing import Dict, Optional, Set
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FREE EMAIL DOMAINS (Blacklist for matching)
# =============================================================================

FREE_EMAIL_DOMAINS = {
    # Major providers
    'gmail.com', 'googlemail.com', 'yahoo.com', 'yahoo.co.uk', 'yahoo.fr', 
    'yahoo.de', 'yahoo.it', 'yahoo.es', 'outlook.com', 'hotmail.com', 
    'hotmail.co.uk', 'hotmail.fr', 'hotmail.de', 'hotmail.it', 'live.com',
    'msn.com', 'icloud.com', 'me.com', 'mac.com', 'aol.com',
    
    # European providers
    'gmx.de', 'gmx.net', 'gmx.at', 'gmx.ch', 'web.de', 't-online.de',
    'freenet.de', 'arcor.de', 'orange.fr', 'laposte.net', 'sfr.fr',
    'free.fr', 'wanadoo.fr', 'libero.it', 'virgilio.it', 'tin.it',
    'alice.it', 'tiscali.it', 'telefonica.net', 'terra.es',
    
    # Other common
    'protonmail.com', 'proton.me', 'tutanota.com', 'zoho.com',
    'yandex.com', 'yandex.ru', 'mail.ru', 'rambler.ru',
    'qq.com', '163.com', '126.com', 'sina.com',
    
    # Generic
    'email.com', 'mail.com', 'inbox.com', 'fastmail.com',
}

# Domains that should be ignored for matching (too common, not company-specific)
IGNORE_DOMAINS = FREE_EMAIL_DOMAINS | {
    'linkedin.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'youtube.com', 'github.com', 'gitlab.com', 'bitbucket.org',
    'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
    'wordpress.com', 'blogspot.com', 'medium.com', 'wix.com',
    'squarespace.com', 'weebly.com', 'godaddy.com',
}


# =============================================================================
# TLD HANDLING
# =============================================================================

# Known multi-part TLDs (country-code second-level domains)
MULTI_PART_TLDS = {
    # UK
    'co.uk', 'org.uk', 'me.uk', 'ltd.uk', 'plc.uk', 'ac.uk',
    # Australia
    'com.au', 'net.au', 'org.au', 'edu.au', 'gov.au',
    # Other
    'co.nz', 'co.za', 'com.br', 'co.jp', 'co.kr', 'co.in',
    'com.mx', 'com.ar', 'com.cn', 'com.hk', 'com.sg',
    'com.tr', 'com.ua', 'com.pl', 'co.il', 'com.eg',
}


def extract_etld1(url_or_email: str) -> Dict[str, any]:
    """
    Extract eTLD+1 (registrable domain) from URL or email.
    
    Examples:
        'https://www.example.co.uk/page' → 'example.co.uk'
        'contact@sales.company.com' → 'company.com'
        'https://subdomain.example.com:8080/path' → 'example.com'
    
    Returns dict with:
        - etld1: registrable domain (e.g., 'example.com')
        - host: full hostname (e.g., 'www.example.com')
        - is_free_email: True if domain is a free email provider
        - is_ignored: True if domain should be ignored for matching
        - is_missing: True if extraction failed
    """
    if not url_or_email or (isinstance(url_or_email, float) and str(url_or_email) == 'nan'):
        return {
            'etld1': None,
            'host': None,
            'is_free_email': False,
            'is_ignored': False,
            'is_missing': True,
        }
    
    url_or_email = str(url_or_email).strip().lower()
    
    # Detect if it's an email
    if '@' in url_or_email:
        # Extract domain from email
        try:
            domain = url_or_email.split('@')[1].strip()
            host = domain
        except:
            return {
                'etld1': None,
                'host': None,
                'is_free_email': False,
                'is_ignored': False,
                'is_missing': True,
            }
    else:
        # It's a URL - normalize it
        url = url_or_email
        
        # Add scheme if missing
        if not url.startswith(('http://', 'https://', '//')):
            url = 'https://' + url
        
        try:
            parsed = urlparse(url)
            host = parsed.netloc or parsed.path.split('/')[0]
            
            # Remove port
            host = host.split(':')[0]
            
            # Remove www prefix
            if host.startswith('www.'):
                host = host[4:]
                
        except:
            # Fallback: simple extraction
            host = re.sub(r'^(https?://)?((www|ww2|ww3)\.)?', '', url)
            host = host.split('/')[0].split(':')[0].split('?')[0]
    
    if not host:
        return {
            'etld1': None,
            'host': None,
            'is_free_email': False,
            'is_ignored': False,
            'is_missing': True,
        }
    
    # Extract eTLD+1
    etld1 = _extract_registrable_domain(host)
    
    # Check if free email
    is_free = etld1 in FREE_EMAIL_DOMAINS if etld1 else False
    is_ignored = etld1 in IGNORE_DOMAINS if etld1 else False
    
    return {
        'etld1': etld1,
        'host': host,
        'is_free_email': is_free,
        'is_ignored': is_ignored,
        'is_missing': etld1 is None,
    }


def _extract_registrable_domain(host: str) -> Optional[str]:
    """
    Extract the registrable domain (eTLD+1) from a hostname.
    Handles multi-part TLDs like .co.uk.
    """
    if not host:
        return None
    
    parts = host.split('.')
    
    if len(parts) < 2:
        return None
    
    # Check for known multi-part TLDs
    if len(parts) >= 3:
        possible_tld = '.'.join(parts[-2:])
        if possible_tld in MULTI_PART_TLDS:
            if len(parts) >= 3:
                return '.'.join(parts[-3:])
    
    # Standard case: last two parts
    return '.'.join(parts[-2:])


def normalize_domain(domain: str) -> Optional[str]:
    """Normalize a domain for comparison."""
    if not domain:
        return None
    
    domain = str(domain).strip().lower()
    
    # Remove protocol
    domain = re.sub(r'^(https?://)?', '', domain)
    
    # Remove www
    domain = re.sub(r'^(www\d?\.)', '', domain)
    
    # Remove path/query
    domain = domain.split('/')[0].split('?')[0].split('#')[0]
    
    # Remove port
    domain = domain.split(':')[0]
    
    return domain if domain else None


def domains_match(domain1: str, domain2: str) -> bool:
    """
    Check if two domains match (are the same registrable domain).
    """
    if not domain1 or not domain2:
        return False
    
    d1 = extract_etld1(domain1)
    d2 = extract_etld1(domain2)
    
    if d1['is_missing'] or d2['is_missing']:
        return False
    
    return d1['etld1'] == d2['etld1']


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def extract_domain_column(urls: 'pd.Series', chunk_size: int = 500000) -> 'pd.DataFrame':
    """
    VECTORIZED domain extraction using pandas string ops.
    10-50x faster than apply() for large datasets.
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
    
    n_rows = len(urls)
    logger.info(f"Extracting domains from {n_rows:,} URLs (vectorized)...")
    
    # Fill NaN and convert to string
    s = urls.fillna('').astype(str).str.strip().str.lower()
    
    # Step 1: Remove protocol (vectorized)
    s = s.str.replace(r'^https?://', '', regex=True)
    
    # Step 2: Remove www. prefix (vectorized)
    s = s.str.replace(r'^www\.', '', regex=True)
    
    # Step 3: Handle email addresses (extract domain part)
    is_email = s.str.contains('@', na=False)
    s = s.where(~is_email, s.str.split('@').str[-1])
    
    # Step 4: Remove path/query (keep only host)
    s = s.str.split('/').str[0]
    s = s.str.split('?').str[0]
    s = s.str.split(':').str[0]  # Remove port
    
    # Step 5: Check for free email domains (vectorized)
    is_free_email = s.isin(FREE_EMAIL_DOMAINS)
    
    # Step 6: Check for ignored domains (vectorized)  
    is_ignored = s.isin(IGNORE_DOMAINS)
    
    # Step 7: Handle empty/invalid
    is_missing = (s == '') | (s.isna()) | (~s.str.contains('.', na=False))
    
    # Build result DataFrame
    result = pd.DataFrame({
        'etld1': s.where(~is_missing, ''),
        'host': s.where(~is_missing, ''),
        'is_free_email': is_free_email,
        'is_ignored': is_ignored,
        'is_missing': is_missing,
    })
    
    logger.info(f"Domain extraction complete: {n_rows:,} rows")
    
    return result


if __name__ == '__main__':
    # Test cases
    test_inputs = [
        'https://www.example.com/page',
        'http://subdomain.example.co.uk:8080/path?query',
        'www.company.de',
        'contact@sales.example.com',
        'info@gmail.com',
        'example.com',
        '',
        None,
    ]
    
    print("=== Domain Extraction Test ===\n")
    for inp in test_inputs:
        result = extract_etld1(inp)
        print(f"Input: '{inp}'")
        print(f"  etld1: '{result['etld1']}'")
        print(f"  host: '{result['host']}'")
        print(f"  is_free_email: {result['is_free_email']}")
        print(f"  is_missing: {result['is_missing']}")
        print()
