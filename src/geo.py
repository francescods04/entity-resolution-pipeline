"""
geo.py - Geographic Parsing Module

Parse and normalize location data from Crunchbase and Orbis.
Handles: country code standardization, city normalization, location string parsing.
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# COUNTRY CODE MAPPING
# =============================================================================

# Common country name → ISO 3166-1 alpha-2 mapping
COUNTRY_TO_ISO = {
    # English names
    'austria': 'AT', 'belgium': 'BE', 'bulgaria': 'BG', 'croatia': 'HR',
    'cyprus': 'CY', 'czech republic': 'CZ', 'czechia': 'CZ',
    'denmark': 'DK', 'estonia': 'EE', 'finland': 'FI', 'france': 'FR',
    'germany': 'DE', 'greece': 'GR', 'hungary': 'HU', 'ireland': 'IE',
    'italy': 'IT', 'latvia': 'LV', 'lithuania': 'LT', 'luxembourg': 'LU',
    'malta': 'MT', 'netherlands': 'NL', 'the netherlands': 'NL', 'holland': 'NL',
    'poland': 'PL', 'portugal': 'PT', 'romania': 'RO', 'slovakia': 'SK',
    'slovenia': 'SI', 'spain': 'ES', 'sweden': 'SE',
    
    # Non-EU European
    'united kingdom': 'GB', 'uk': 'GB', 'great britain': 'GB', 'england': 'GB',
    'scotland': 'GB', 'wales': 'GB', 'northern ireland': 'GB',
    'switzerland': 'CH', 'norway': 'NO', 'iceland': 'IS',
    'liechtenstein': 'LI', 'monaco': 'MC', 'andorra': 'AD', 'san marino': 'SM',
    
    # Other major countries
    'united states': 'US', 'usa': 'US', 'us': 'US', 'america': 'US',
    'canada': 'CA', 'australia': 'AU', 'new zealand': 'NZ',
    'china': 'CN', 'japan': 'JP', 'south korea': 'KR', 'korea': 'KR',
    'india': 'IN', 'singapore': 'SG', 'hong kong': 'HK',
    'israel': 'IL', 'turkey': 'TR', 'russia': 'RU', 'ukraine': 'UA',
    'brazil': 'BR', 'mexico': 'MX', 'argentina': 'AR',
    'south africa': 'ZA', 'egypt': 'EG', 'nigeria': 'NG',
    'united arab emirates': 'AE', 'uae': 'AE', 'saudi arabia': 'SA',
}

# Region keywords that can help identify country
REGION_TO_COUNTRY = {
    'europe': None,  # Not specific enough
    'european union': None,
    'eu': None,
    'western europe': None,
    'eastern europe': None,
    'nordic': None,
    'scandinavia': None,
    'asia pacific': None,
    'emea': None,
    'americas': None,
    'latin america': None,
}

# Valid ISO codes (for validation)
VALID_ISO_CODES = set(COUNTRY_TO_ISO.values()) | {
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
    'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK',
    'SI', 'ES', 'SE', 'GB', 'CH', 'NO', 'IS', 'LI', 'MC', 'AD', 'SM', 'VA',
    'US', 'CA', 'AU', 'NZ', 'CN', 'JP', 'KR', 'IN', 'SG', 'HK', 'TW', 'MY',
    'ID', 'TH', 'VN', 'PH', 'IL', 'TR', 'RU', 'UA', 'BY', 'KZ', 'GE', 'AM',
    'BR', 'MX', 'AR', 'CL', 'CO', 'PE', 'VE', 'EC', 'UY', 'PY', 'BO',
    'ZA', 'EG', 'NG', 'KE', 'MA', 'TN', 'GH', 'TZ', 'ET',
    'AE', 'SA', 'QA', 'KW', 'BH', 'OM', 'JO', 'LB', 'IR', 'IQ', 'PK',
}


# =============================================================================
# LOCATION PARSING
# =============================================================================

def parse_crunchbase_location(location_str: str) -> Dict[str, Optional[str]]:
    """
    Parse Crunchbase HQ Location string.
    
    Format: "City, Region, Country, Continent"
    Example: "Berlin, Berlin, Germany, Europe"
    Example: "San Francisco, California, United States, North America"
    
    Returns dict with:
        - country_iso: ISO 3166-1 alpha-2 code
        - country_name: original country name
        - city_norm: normalized city name
        - region: state/region if present
    """
    if not location_str or (isinstance(location_str, float) and str(location_str) == 'nan'):
        return {
            'country_iso': None,
            'country_name': None,
            'city_norm': None,
            'region': None,
        }
    
    location_str = str(location_str).strip()
    
    # Split by comma
    parts = [p.strip() for p in location_str.split(',')]
    parts = [p for p in parts if p]  # Remove empty
    
    if not parts:
        return {
            'country_iso': None,
            'country_name': None,
            'city_norm': None,
            'region': None,
        }
    
    country_iso = None
    country_name = None
    city = None
    region = None
    
    # Work backwards to find country (usually second-to-last or last before continent)
    for i in range(len(parts) - 1, -1, -1):
        part_lower = parts[i].lower().strip()
        
        # Skip region keywords (continent-level)
        if part_lower in REGION_TO_COUNTRY:
            continue
        
        # Check if it's a known country
        if part_lower in COUNTRY_TO_ISO:
            country_iso = COUNTRY_TO_ISO[part_lower]
            country_name = parts[i]
            
            # City is typically first part
            if i > 0:
                city = parts[0]
            
            # Region is second part if we have at least 3 parts before continent
            if i > 1:
                region = parts[1]
            
            break
        
        # Check if it looks like a 2-letter ISO code
        if len(part_lower) == 2 and part_lower.upper() in VALID_ISO_CODES:
            country_iso = part_lower.upper()
            country_name = parts[i]
            if i > 0:
                city = parts[0]
            if i > 1:
                region = parts[1]
            break
    
    # Normalize city
    city_norm = normalize_city(city) if city else None
    
    return {
        'country_iso': country_iso,
        'country_name': country_name,
        'city_norm': city_norm,
        'region': region,
    }


def normalize_city(city: str) -> Optional[str]:
    """
    Normalize city name for comparison.
    - Lowercase
    - Remove punctuation
    - Handle common variants
    """
    if not city:
        return None
    
    city = str(city).strip().lower()
    
    # Remove common prefixes/suffixes
    city = re.sub(r'^(city of|greater|metro)\s+', '', city)
    city = re.sub(r'\s+(city|metro|metropolitan area)$', '', city)
    
    # Remove punctuation
    city = re.sub(r'[^\w\s]', '', city)
    
    # Normalize whitespace
    city = ' '.join(city.split())
    
    # Common normalizations
    city_mappings = {
        'new york': 'new york',
        'nyc': 'new york',
        'manhattan': 'new york',
        'brooklyn': 'new york',
        'la': 'los angeles',
        'sf': 'san francisco',
        'london': 'london',
        'munich': 'munich',
        'muenchen': 'munich',
        'münchen': 'munich',
        'cologne': 'cologne',
        'koeln': 'cologne',
        'köln': 'cologne',
        'milan': 'milan',
        'milano': 'milan',
        'rome': 'rome',
        'roma': 'rome',
        'vienna': 'vienna',
        'wien': 'vienna',
        'warsaw': 'warsaw',
        'warszawa': 'warsaw',
        'prague': 'prague',
        'praha': 'prague',
        'copenhagen': 'copenhagen',
        'kobenhavn': 'copenhagen',
        'stockholm': 'stockholm',
        'helsinki': 'helsinki',
        'brussels': 'brussels',
        'bruxelles': 'brussels',
        'amsterdam': 'amsterdam',
        'the hague': 'the hague',
        'den haag': 'the hague',
        'geneva': 'geneva',
        'geneve': 'geneva',
        'zurich': 'zurich',
        'zuerich': 'zurich',
        'zürich': 'zurich',
    }
    
    return city_mappings.get(city, city)


def standardize_country_code(code: str) -> Optional[str]:
    """
    Standardize country code to ISO 3166-1 alpha-2.
    Handles: 2-letter codes, 3-letter codes, full names.
    """
    if not code or (isinstance(code, float) and str(code) == 'nan'):
        return None
    
    code = str(code).strip()
    
    # Already a valid 2-letter code
    if len(code) == 2 and code.upper() in VALID_ISO_CODES:
        return code.upper()
    
    # Check country name mapping
    code_lower = code.lower()
    if code_lower in COUNTRY_TO_ISO:
        return COUNTRY_TO_ISO[code_lower]
    
    # Try removing common prefixes
    for prefix in ['the ', 'republic of ', 'kingdom of ']:
        if code_lower.startswith(prefix):
            trimmed = code_lower[len(prefix):]
            if trimmed in COUNTRY_TO_ISO:
                return COUNTRY_TO_ISO[trimmed]
    
    return None


def countries_match(country1: str, country2: str) -> bool:
    """Check if two country codes/names represent the same country."""
    if not country1 or not country2:
        return False
    
    iso1 = standardize_country_code(country1)
    iso2 = standardize_country_code(country2)
    
    if iso1 and iso2:
        return iso1 == iso2
    
    return False


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def parse_location_column(locations: 'pd.Series') -> 'pd.DataFrame':
    """
    VECTORIZED location parsing for Crunchbase HQ locations.
    Format: "City, Region, Country, Continent"
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
    
    n_rows = len(locations)
    logger.info(f"Parsing {n_rows:,} locations (vectorized)...")
    
    # Fill NaN and convert to string
    s = locations.fillna('').astype(str).str.strip()
    
    # Split by comma
    parts = s.str.split(',', expand=True)
    
    # Extract components (handle variable number of parts)
    n_cols = parts.shape[1] if parts is not None else 0
    
    city = parts[0].str.strip() if n_cols > 0 else pd.Series([''] * n_rows)
    region = parts[1].str.strip() if n_cols > 1 else pd.Series([''] * n_rows)
    country_name = parts[2].str.strip() if n_cols > 2 else pd.Series([''] * n_rows)
    # Continent in parts[3] - we don't need it
    
    # Normalize country to ISO (vectorized via mapping)
    country_lower = country_name.str.lower()
    country_iso = country_lower.map(COUNTRY_TO_ISO).fillna('')
    
    # Normalize city (vectorized)
    city_norm = city.str.lower().str.strip()
    city_norm = city_norm.str.replace(r'[^\w\s]', '', regex=True)
    city_norm = city_norm.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Build result DataFrame
    result = pd.DataFrame({
        'country_iso': country_iso,
        'country_name': country_name,
        'city_norm': city_norm,
        'region': region,
    })
    
    logger.info(f"Location parsing complete: {n_rows:,} rows")
    
    return result


if __name__ == '__main__':
    # Test cases
    test_locations = [
        "Berlin, Berlin, Germany, Europe",
        "San Francisco, California, United States, North America",
        "London, England, United Kingdom, Europe",
        "Amsterdam, Noord-Holland, The Netherlands, Europe",
        "Paris, Île-de-France, France, Europe",
        "Stockholm, Stockholm, Sweden, Europe",
        "",
        None,
    ]
    
    print("=== Location Parsing Test ===\n")
    for loc in test_locations:
        result = parse_crunchbase_location(loc)
        print(f"Input: '{loc}'")
        print(f"  country_iso: '{result['country_iso']}'")
        print(f"  city_norm: '{result['city_norm']}'")
        print(f"  region: '{result['region']}'")
        print()
