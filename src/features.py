"""
features.py - Pair Feature Computation

Computes similarity features for candidate pairs to feed into the matching model.

FEATURE CATEGORIES:
------------------
1. String Similarity (name): Jaro-Winkler, token Jaccard, RapidFuzz ratio
2. Domain/Contact: exact match, family overlap
3. Geographic: country match, city similarity
4. Temporal: founded year vs incorporation year
5. Semantic: embedding cosine similarity (computed separately on GPU)
6. Structural: Orbis corporate role, family size

HANDLING NAME COLLISIONS:
------------------------
For ambiguous names (e.g., "B & C SRL"), we compute additional disambiguation
features that help the model distinguish between multiple Orbis entities.
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle

import pandas as pd
import numpy as np

# String similarity
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    
logger = logging.getLogger(__name__)


# =============================================================================
# STRING SIMILARITY FEATURES
# =============================================================================

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity (0-1 scale)."""
    if not s1 or not s2:
        return 0.0
    
    s1 = str(s1).lower().strip()
    s2 = str(s2).lower().strip()
    
    if s1 == s2:
        return 1.0
    
    if HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(s1, s2)
    else:
        # Fallback simple implementation
        return _simple_jaro_winkler(s1, s2)


def _simple_jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Simple Jaro-Winkler implementation as fallback."""
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Match window
    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(0, match_distance)
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    
    # Winkler adjustment
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + prefix * prefix_weight * (1 - jaro)


def token_jaccard(s1: str, s2: str) -> float:
    """Compute token set Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    
    tokens1 = set(_tokenize(str(s1)))
    tokens2 = set(_tokenize(str(s2)))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def rapidfuzz_ratio(s1: str, s2: str) -> float:
    """Compute RapidFuzz token sort ratio (0-100 scale)."""
    if not s1 or not s2:
        return 0.0
    
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(str(s1), str(s2))
    else:
        # Fallback: simple token Jaccard * 100
        return token_jaccard(s1, s2) * 100


def prefix_match(s1: str, s2: str, length: int = 8) -> bool:
    """Check if first N characters match."""
    if not s1 or not s2:
        return False
    
    clean1 = re.sub(r'[^a-zA-Z0-9]', '', str(s1).lower())[:length]
    clean2 = re.sub(r'[^a-zA-Z0-9]', '', str(s2).lower())[:length]
    
    return clean1 == clean2 and len(clean1) >= length


def acronym_match(s1: str, s2: str) -> bool:
    """Check if acronyms match."""
    def get_acronym(s):
        tokens = _tokenize(str(s))
        return ''.join(t[0] for t in tokens if t).upper()
    
    acr1 = get_acronym(s1)
    acr2 = get_acronym(s2)
    
    return acr1 == acr2 and len(acr1) >= 2


def _tokenize(text: str) -> List[str]:
    """Tokenize text for comparisons."""
    if not text:
        return []
    text = str(text).lower()
    tokens = re.split(r'[^a-zA-Z0-9]+', text)
    return [t for t in tokens if len(t) >= 2]


# =============================================================================
# DOMAIN/CONTACT FEATURES
# =============================================================================

def domain_exact_match(d1: str, d2: str) -> bool:
    """Check if domains match exactly (after normalization)."""
    if not d1 or not d2:
        return False
    
    def norm(d):
        d = str(d).lower().strip()
        d = re.sub(r'^https?://', '', d)
        d = re.sub(r'^www\d?\.', '', d)
        d = d.split('/')[0]
        return d
    
    return norm(d1) == norm(d2)


def domain_in_family(cb_domain: str, orbis_domains: str) -> bool:
    """Check if CB domain appears in Orbis family domains (pipe-separated)."""
    if not cb_domain or not orbis_domains:
        return False
    
    cb_norm = _normalize_domain(cb_domain)
    orbis_list = str(orbis_domains).split('|')
    
    for od in orbis_list:
        if _normalize_domain(od) == cb_norm:
            return True
    
    return False


def _normalize_domain(d: str) -> str:
    """
    Extract eTLD+1 (registrable domain) for comparison.
    
    Correctly handles subdomains:
        store.husarion.com → husarion.com
        www.onepark.fr     → onepark.fr
        example.co.uk      → example.co.uk
    """
    if not d:
        return ''
    d = str(d).strip()
    if not d:
        return ''
    
    # Primary: tldextract for proper eTLD+1
    try:
        import tldextract
        ext = tldextract.extract(d)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        if ext.domain:
            return ext.domain
        return ''
    except Exception:
        pass
    
    # Fallback: manual extraction (strip protocol/www/path, then eTLD+1)
    d = d.lower()
    d = re.sub(r'^https?://', '', d)
    d = re.sub(r'^www\d?\.', '', d)
    d = d.split('/')[0].split(':')[0].split('?')[0]
    
    # Handle known multi-part TLDs
    parts = d.split('.')
    if len(parts) >= 3:
        two_part_tld = '.'.join(parts[-2:])
        MULTI_PART_TLDS = {
            'co.uk', 'org.uk', 'co.nz', 'co.za', 'co.jp', 'co.kr', 'co.in',
            'co.il', 'com.au', 'com.br', 'com.cn', 'com.hk', 'com.sg',
            'com.mx', 'com.ar', 'com.tr', 'com.ua', 'com.pl', 'com.eg',
        }
        if two_part_tld in MULTI_PART_TLDS:
            return '.'.join(parts[-3:])
    
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return d


# =============================================================================
# GEOGRAPHIC FEATURES
# =============================================================================

def country_match(c1: str, c2: str) -> bool:
    """Check if country codes match."""
    if not c1 or not c2:
        return False
    return str(c1).upper().strip() == str(c2).upper().strip()


def city_similarity(city1: str, city2: str) -> float:
    """Compute city name similarity (0-1 scale)."""
    if not city1 or not city2:
        return 0.0
    
    city1 = str(city1).lower().strip()
    city2 = str(city2).lower().strip()
    
    if city1 == city2:
        return 1.0
    
    # Token overlap
    tokens1 = set(city1.split())
    tokens2 = set(city2.split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    overlap = len(tokens1 & tokens2)
    total = len(tokens1 | tokens2)
    
    return overlap / total if total > 0 else 0.0


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def year_difference(year1: Optional[int], year2: Optional[int]) -> Optional[int]:
    """Compute absolute year difference."""
    if year1 is None or year2 is None:
        return None
    
    try:
        return abs(int(year1) - int(year2))
    except:
        return None


def year_compatibility_score(year1: Optional[int], year2: Optional[int]) -> float:
    """
    Compute year compatibility score (0-1 scale).
    - Same year: 1.0
    - 1-2 year diff: 0.8
    - 3-5 year diff: 0.5
    - >5 year diff: 0.2
    - Missing: 0.5 (neutral)
    """
    diff = year_difference(year1, year2)
    
    if diff is None:
        return 0.5  # Neutral
    
    if diff == 0:
        return 1.0
    elif diff <= 2:
        return 0.8
    elif diff <= 5:
        return 0.5
    else:
        return 0.2


# =============================================================================
# STRUCTURAL FEATURES (Orbis corporate graph)
# =============================================================================

def is_guo(entity_role: str) -> bool:
    """Check if entity is Global Ultimate Owner."""
    return str(entity_role).upper().strip() == 'GUO'


def is_subsidiary(entity_role: str) -> bool:
    """Check if entity is a subsidiary."""
    return str(entity_role).upper().strip() in ['SUBSIDIARY', 'SUB']


def is_branch(entity_role: str) -> bool:
    """Check if entity is a branch."""
    return str(entity_role).upper().strip() == 'BRANCH'


def family_size_log(family_size: Optional[int]) -> float:
    """Log-transformed family size for feature."""
    if family_size is None or family_size <= 0:
        return 0.0
    return np.log1p(family_size)


# =============================================================================
# LEARNED DISAMBIGUATION (SOTA: NO MAGIC NUMBERS)
# =============================================================================

class LearnedDisambiguator:
    """
    Logistic Regression-based disambiguation scorer.
    
    Replaces hardcoded heuristic weights with a learned probabilistic model.
    This is scientifically defensible for research-grade entity resolution.
    
    Usage:
        # Training
        disambiguator = LearnedDisambiguator()
        disambiguator.fit(features_df, labels)
        disambiguator.save('disambiguator.pkl')
        
        # Inference
        disambiguator = LearnedDisambiguator.load('disambiguator.pkl')
        score = disambiguator.score(feature_dict)
    """
    
    FEATURE_COLUMNS = [
        'domain_match', 'country_match', 'city_sim', 'year_compat'
    ]
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self._is_fitted = False
    
    def _extract_features(self, row: Dict) -> np.ndarray:
        """Extract disambiguation features from a row dict."""
        features = []
        
        # Domain match (binary)
        domain_match = 1.0 if domain_exact_match(
            row.get('cb_domain', ''), 
            row.get('orbis_domain', row.get('orbis_website', ''))
        ) else 0.0
        features.append(domain_match)
        
        # Country match (binary)
        country_m = 1.0 if country_match(
            row.get('cb_country_iso', ''), 
            row.get('orbis_country', '')
        ) else 0.0
        features.append(country_m)
        
        # City similarity (continuous 0-1)
        city_sim = city_similarity(
            row.get('cb_city_norm', ''), 
            row.get('orbis_city', '')
        )
        features.append(city_sim)
        
        # Year compatibility (continuous 0-1)
        year_compat = year_compatibility_score(
            row.get('cb_founded_year'),
            row.get('orbis_incorp_year')
        )
        features.append(year_compat)
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'LearnedDisambiguator':
        """
        Fit the disambiguation model.
        
        Args:
            X: DataFrame with columns: domain_match, country_match, city_sim, year_compat
            y: Binary labels (1=correct match, 0=incorrect)
        
        Returns:
            self
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        # Ensure correct columns
        X_arr = X[self.FEATURE_COLUMNS].values if isinstance(X, pd.DataFrame) else X
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
        
        self._is_fitted = True
        
        # Log learned weights
        logger.info("LearnedDisambiguator fitted:")
        for i, col in enumerate(self.FEATURE_COLUMNS):
            logger.info(f"  {col}: coef={self.model.coef_[0][i]:.3f}")
        
        return self
    
    def score(self, row: Dict) -> float:
        """
        Compute disambiguation probability for a candidate pair.
        
        Args:
            row: Dict with keys cb_domain, orbis_domain, cb_country_iso, etc.
        
        Returns:
            Probability of correct match (0-1 scale)
        """
        if not self._is_fitted:
            raise RuntimeError("LearnedDisambiguator not fitted. Call fit() first.")
        
        X = self._extract_features(row)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict_proba(X_scaled)[0, 1])
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self._is_fitted,
            }, f)
        logger.info(f"Saved LearnedDisambiguator to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LearnedDisambiguator':
        """Load model from disk."""
        disambiguator = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        disambiguator.model = data['model']
        disambiguator.scaler = data['scaler']
        disambiguator._is_fitted = data['is_fitted']
        logger.info(f"Loaded LearnedDisambiguator from {path}")
        return disambiguator


# Global instance (lazy-loaded)
_GLOBAL_DISAMBIGUATOR: Optional[LearnedDisambiguator] = None


def get_disambiguator(model_path: Optional[str] = None) -> LearnedDisambiguator:
    """Get or create global disambiguator instance."""
    global _GLOBAL_DISAMBIGUATOR
    
    if _GLOBAL_DISAMBIGUATOR is None:
        if model_path and Path(model_path).exists():
            _GLOBAL_DISAMBIGUATOR = LearnedDisambiguator.load(model_path)
        else:
            _GLOBAL_DISAMBIGUATOR = LearnedDisambiguator()
    
    return _GLOBAL_DISAMBIGUATOR


# =============================================================================
# DEPRECATED: OLD HEURISTIC DISAMBIGUATION (KEPT FOR BACKWARD COMPATIBILITY)
# =============================================================================

def compute_disambiguation_score(
    cb_row: Dict,
    orbis_row: Dict,
) -> float:
    """
    [DEPRECATED] Compute disambiguation score using hardcoded weights.
    
    .. deprecated:: 2.0
        Use `LearnedDisambiguator.score()` instead. This function uses
        magic numbers that are not scientifically defensible.
    
    Higher score = more likely to be the correct match among
    multiple Orbis entities with the same name.
    """
    warnings.warn(
        "compute_disambiguation_score() is deprecated and uses magic numbers. "
        "Use LearnedDisambiguator.score() for rigorous scoring.",
        DeprecationWarning,
        stacklevel=2
    )
    
    score = 0.0
    
    # Domain match is strongest disambiguator
    if domain_exact_match(cb_row.get('cb_domain'), orbis_row.get('orbis_domain')):
        score += 100.0
    
    # Country match
    if country_match(cb_row.get('cb_country_iso'), orbis_row.get('orbis_country')):
        score += 30.0
    else:
        score -= 50.0  # Strong penalty for country mismatch
    
    # City match
    city_sim = city_similarity(cb_row.get('cb_city_norm'), orbis_row.get('orbis_city'))
    score += city_sim * 20.0
    
    # Year compatibility
    year_score = year_compatibility_score(
        cb_row.get('cb_founded_year'),
        orbis_row.get('orbis_incorp_year')
    )
    score += (year_score - 0.5) * 20.0  # Center around neutral
    
    return score


# =============================================================================
# INDUSTRY & DESCRIPTION OVERLAP (P1: Deal sheet exploitation)
# =============================================================================

def industry_token_overlap(cb_industries: str, orbis_trade_desc: str) -> float:
    """
    Token Jaccard between CB industry keywords and Orbis trade description.
    
    CB industries: pipe-separated (e.g., "Software|SaaS|Cloud Computing")
    Orbis trade desc: free text (e.g., "Development of cloud software solutions")
    """
    if not cb_industries or not orbis_trade_desc:
        return 0.0
    if pd.isna(cb_industries) or pd.isna(orbis_trade_desc):
        return 0.0
    
    # Tokenize CB industries (split on pipe, then tokenize each)
    cb_tokens = set()
    for industry in str(cb_industries).split('|'):
        for token in industry.lower().split():
            token = re.sub(r'[^a-z0-9]', '', token)
            if len(token) >= 3:
                cb_tokens.add(token)
    
    # Tokenize Orbis trade description
    orbis_tokens = set()
    for token in str(orbis_trade_desc).lower().split():
        token = re.sub(r'[^a-z0-9]', '', token)
        if len(token) >= 3:
            orbis_tokens.add(token)
    
    if not cb_tokens or not orbis_tokens:
        return 0.0
    
    intersection = len(cb_tokens & orbis_tokens)
    union = len(cb_tokens | orbis_tokens)
    return intersection / union if union > 0 else 0.0


def description_token_overlap(cb_desc: str, orbis_trade_desc: str) -> float:
    """
    Token Jaccard between CB company description and Orbis trade description.
    Both are free text — measures semantic domain similarity.
    """
    if not cb_desc or not orbis_trade_desc:
        return 0.0
    if pd.isna(cb_desc) or pd.isna(orbis_trade_desc):
        return 0.0
    
    def _tokenize(text):
        tokens = set()
        for token in str(text).lower().split():
            token = re.sub(r'[^a-z0-9]', '', token)
            if len(token) >= 3:
                tokens.add(token)
        return tokens
    
    cb_tokens = _tokenize(cb_desc)
    orbis_tokens = _tokenize(orbis_trade_desc)
    
    if not cb_tokens or not orbis_tokens:
        return 0.0
    
    intersection = len(cb_tokens & orbis_tokens)
    union = len(cb_tokens | orbis_tokens)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# FULL FEATURE COMPUTATION
# =============================================================================

def compute_pair_features(
    cb_row: Dict,
    orbis_row: Dict,
    alias_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute all features for a candidate pair.
    
    Args:
        cb_row: Crunchbase company record (dict)
        orbis_row: Orbis company record (dict)
        alias_names: Optional list of known aliases for the CB company
    
    Returns:
        Dict of feature_name → value
    """
    features = {}
    
    # Get key fields
    cb_name = cb_row.get('cb_name', '')
    orbis_name = orbis_row.get('orbis_name', '')
    
    # 1. Name similarity features
    features['name_jw'] = jaro_winkler_similarity(cb_name, orbis_name)
    features['name_token_jaccard'] = token_jaccard(cb_name, orbis_name)
    features['name_rapidfuzz_ratio'] = rapidfuzz_ratio(cb_name, orbis_name)
    features['name_prefix_match'] = prefix_match(cb_name, orbis_name)
    features['name_acronym_match'] = acronym_match(cb_name, orbis_name)
    
    # Alias matching (best similarity among all known aliases)
    features['alias_best_sim'] = 0.0
    if alias_names:
        for alias in alias_names:
            sim = jaro_winkler_similarity(alias, orbis_name)
            if sim > features['alias_best_sim']:
                features['alias_best_sim'] = sim
    
    # 2. Domain features
    features['domain_exact'] = domain_exact_match(
        cb_row.get('cb_domain'), 
        orbis_row.get('orbis_website')
    )
    features['email_domain_exact'] = domain_exact_match(
        cb_row.get('cb_email_domain'),
        orbis_row.get('orbis_email')
    )
    features['domain_in_family'] = domain_in_family(
        cb_row.get('cb_domain'),
        orbis_row.get('orbis_website')  # Orbis may have multiple websites
    )
    
    # 3. Geographic features
    features['country_match'] = country_match(
        cb_row.get('cb_country_iso'),
        orbis_row.get('orbis_country')
    )
    features['city_sim'] = city_similarity(
        cb_row.get('cb_city_norm'),
        orbis_row.get('orbis_city')
    )
    
    # 4. Temporal features
    features['year_diff'] = year_difference(
        cb_row.get('cb_founded_year'),
        orbis_row.get('orbis_incorp_year')
    )
    features['year_compat'] = year_compatibility_score(
        cb_row.get('cb_founded_year'),
        orbis_row.get('orbis_incorp_year')
    )
    
    # 5. Structural features
    entity_role = orbis_row.get('entity_role', '')
    features['is_guo'] = is_guo(entity_role)
    features['is_subsidiary'] = is_subsidiary(entity_role)
    features['is_branch'] = is_branch(entity_role)
    features['family_size_log'] = family_size_log(orbis_row.get('family_size'))
    
    # 6. Disambiguation score
    features['disambiguation_score'] = compute_disambiguation_score(cb_row, orbis_row)
    
    # 7. Meta features
    features['is_generic_name'] = cb_row.get('cb_name_is_generic', False)
    features['is_free_email'] = cb_row.get('cb_is_free_email', False)
    
    # 8. Industry & description overlap (P1: Deal sheet data)
    features['industry_token_overlap'] = industry_token_overlap(
        cb_row.get('cb_industries', ''),
        orbis_row.get('orbis_trade_desc', '')
    )
    features['desc_token_overlap'] = description_token_overlap(
        cb_row.get('cb_description', ''),
        orbis_row.get('orbis_trade_desc', '')
    )
    features['has_trade_desc'] = bool(
        orbis_row.get('orbis_trade_desc') and 
        not pd.isna(orbis_row.get('orbis_trade_desc', ''))
    )
    
    return features



# =============================================================================
# EMBEDDING HELPERS
# =============================================================================

def _safe_load_embedding(file_path: Path) -> Optional[np.ndarray]:
    """
    Load embeddings from .npy or raw memmap format.
    
    Handles both standard numpy arrays and raw memmap files created by
    compute_embeddings_streaming (which saves raw float16 + .meta.json).
    """
    import json
    
    if not file_path.exists():
        return None
    
    # Check file isn't empty or too small
    file_size = file_path.stat().st_size
    if file_size < 100:  # Too small to be real embeddings
        logger.warning(f"Embedding file too small ({file_size} bytes): {file_path.name}")
        return None
    
    # Check for memmap metadata first (streaming mode)
    meta_path = Path(str(file_path) + '.meta.json')
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            shape = tuple(meta['shape'])
            dtype = meta.get('dtype', 'float16')
            logger.info(f"Loading memmap: {file_path.name} shape={shape}")
            return np.memmap(str(file_path), dtype=dtype, mode='r', shape=shape)
        except Exception as e:
            logger.warning(f"Failed to load via meta.json: {e}")
    
    # Standard .npy format
    try:
        arr = np.load(str(file_path), mmap_mode='r')
        logger.info(f"Loaded .npy: {file_path.name} shape={arr.shape}")
        return arr
    except (ValueError, Exception):
        pass
    
    # Fallback: infer shape from file size
    bytes_per_elem = np.dtype(np.float16).itemsize
    
    # Try companion index parquet for row count
    index_parquet = file_path.with_name(file_path.stem + '_index.parquet')
    if index_parquet.exists():
        try:
            idx_df = pd.read_parquet(index_parquet, columns=['id'])
            n_rows = len(idx_df)
            dim = file_size // (n_rows * bytes_per_elem)
            del idx_df
        except Exception:
            n_rows, dim = None, None
    else:
        n_rows, dim = None, None
    
    if n_rows is None:
        for dim in [384, 768, 1024]:
            n_rows = file_size // (dim * bytes_per_elem)
            if n_rows * dim * bytes_per_elem == file_size:
                break
        else:
            dim = 384
            n_rows = file_size // (dim * bytes_per_elem)
    
    shape = (n_rows, dim)
    logger.info(f"Inferred memmap: {file_path.name} shape={shape}")
    
    # Save metadata for future loads
    try:
        with open(str(meta_path), 'w') as f:
            json.dump({'shape': list(shape), 'dtype': 'float16'}, f)
    except Exception:
        pass
    
    try:
        return np.memmap(str(file_path), dtype=np.float16, mode='r', shape=shape)
    except Exception as e:
        logger.error(f"Failed to memmap {file_path.name}: {e}")
        return None


def load_embedding_resources(embeddings_dir: str) -> Optional[Dict]:
    """
    Load embedding arrays (mmap) and ID maps for fast lookup.
    
    Handles both standard .npy and raw memmap formats transparently.
    
    Args:
        embeddings_dir: Directory containing .npy and index .parquet files
        
    Returns:
        Dict with keys: 'cb_emb', 'orbis_emb', 'cb_id_map', 'orbis_id_map'
    """
    emb_path = Path(embeddings_dir)
    if not emb_path.exists():
        return None
        
    resources = {}
    
    try:
        # 1. Load Arrays (handles both .npy and raw memmap)
        cb_emb = _safe_load_embedding(emb_path / 'cb_desc_emb.npy')
        if cb_emb is not None:
            resources['cb_emb'] = cb_emb
            
        # Try desc first, fall back to name embeddings (GPU step may produce either)
        orbis_emb = _safe_load_embedding(emb_path / 'orbis_desc_emb.npy')
        if orbis_emb is None:
            orbis_emb = _safe_load_embedding(emb_path / 'orbis_name_emb.npy')
        if orbis_emb is not None:
            resources['orbis_emb'] = orbis_emb
            
        # 2. Load Maps (ID -> Index)
        if (emb_path / 'cb_desc_emb_index.parquet').exists():
            df = pd.read_parquet(emb_path / 'cb_desc_emb_index.parquet', columns=['id', 'idx'])
            resources['cb_id_map'] = dict(zip(df['id'], df['idx']))
            
        # Try desc index first, fall back to name index
        orbis_idx_path = emb_path / 'orbis_desc_emb_index.parquet'
        if not orbis_idx_path.exists():
            orbis_idx_path = emb_path / 'orbis_name_emb_index.parquet'
        if orbis_idx_path.exists():
            df = pd.read_parquet(orbis_idx_path, columns=['id', 'idx'])
            resources['orbis_id_map'] = dict(zip(df['id'], df['idx']))
            
        logger.info(f"Loaded embedding resources: {list(resources.keys())}")
        return resources
    except Exception as e:
        logger.warning(f"Failed to load embedding resources: {e}")
        return None


def compute_features_batch(
    candidates: pd.DataFrame,
    cb_data: pd.DataFrame,
    orbis_data: pd.DataFrame,
    alias_registry = None,
    embedding_resources: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compute features for all candidate pairs using VECTORIZED operations.
    
    Args:
        candidates: DataFrame with cb_id, bvd_id
        cb_data: CB data buffer
        orbis_data: Orbis data buffer
        alias_registry: Optional registry
        embedding_resources: Dict with mmap arrays and ID maps
        
    Returns:
        DataFrame with feature columns
    """
    if candidates.empty:
        return pd.DataFrame()
        
    # Merge data to get full rows
    # We need CB columns and Orbis columns in the same DF
    df = candidates[['cb_id', 'bvd_id']].copy()
    
    df = df.merge(cb_data, on='cb_id', how='left')
    df = df.merge(orbis_data, on='bvd_id', how='left')
    
    # --- 1. String Features (Vectorized) ---
    # We'll use list comprehensions or apply which is fast enough for string metrics 
    # (specialized C libraries like rapidfuzz release the GIL)
    
    logger.info(f"Computing features for {len(df)} pairs...")
    
    # Fill NAs
    df['cb_name'] = df['cb_name'].fillna('').astype(str)
    df['orbis_name'] = df['orbis_name'].fillna('').astype(str)
    
    # ===========================================================================
    # PARALLEL STRING SIMILARITY (P1 OPTIMIZATION)
    # RapidFuzz releases GIL, so parallelization gives 4-8x speedup
    # ===========================================================================
    n_pairs = len(df)
    n_jobs = min(8, max(1, n_pairs // 10000))  # Use more cores for larger datasets
    
    if n_pairs > 50000 and n_jobs > 1:
        logger.info(f"Using parallel processing with {n_jobs} workers for {n_pairs:,} pairs")
        try:
            from joblib import Parallel, delayed
            
            
            # Split into chunks
            chunk_size = (n_pairs + n_jobs - 1) // n_jobs
            
            def compute_chunk_jw(names_a, names_b):
                if HAS_RAPIDFUZZ:
                    from rapidfuzz import distance
                    return [distance.JaroWinkler.similarity(s1, s2) for s1, s2 in zip(names_a, names_b)]
                else:
                    return [jaro_winkler_similarity(s1, s2) for s1, s2 in zip(names_a, names_b)]
            
            def compute_chunk_ratio(names_a, names_b):
                if HAS_RAPIDFUZZ:
                    from rapidfuzz import fuzz
                    return [fuzz.token_sort_ratio(s1, s2) for s1, s2 in zip(names_a, names_b)]
                else:
                    return [rapidfuzz_ratio(s1, s2) for s1, s2 in zip(names_a, names_b)]
            
            # Parallel JW similarity
            chunks = [(df['cb_name'].iloc[i:i+chunk_size].tolist(), 
                      df['orbis_name'].iloc[i:i+chunk_size].tolist()) 
                     for i in range(0, n_pairs, chunk_size)]
            
            jw_results = Parallel(n_jobs=n_jobs)(
                delayed(compute_chunk_jw)(a, b) for a, b in chunks
            )
            df['name_jw'] = np.concatenate(jw_results)
            
            ratio_results = Parallel(n_jobs=n_jobs)(
                delayed(compute_chunk_ratio)(a, b) for a, b in chunks
            )
            df['name_rapidfuzz_ratio'] = np.concatenate(ratio_results)
            
        except ImportError:
            logger.warning("joblib not installed, falling back to sequential")
            if HAS_RAPIDFUZZ:
                from rapidfuzz import fuzz, distance
                df['name_jw'] = [distance.JaroWinkler.similarity(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]
                df['name_rapidfuzz_ratio'] = [fuzz.token_sort_ratio(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]
            else:
                df['name_jw'] = list(map(lambda x: jaro_winkler_similarity(x[0], x[1]), zip(df['cb_name'], df['orbis_name'])))
                df['name_rapidfuzz_ratio'] = list(map(lambda x: rapidfuzz_ratio(x[0], x[1]), zip(df['cb_name'], df['orbis_name'])))
    else:
        # Sequential for small datasets
        if HAS_RAPIDFUZZ:
            from rapidfuzz import fuzz, distance
            df['name_jw'] = [distance.JaroWinkler.similarity(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]
            df['name_rapidfuzz_ratio'] = [fuzz.token_sort_ratio(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]
        else:
            df['name_jw'] = list(map(lambda x: jaro_winkler_similarity(x[0], x[1]), zip(df['cb_name'], df['orbis_name'])))
            df['name_rapidfuzz_ratio'] = list(map(lambda x: rapidfuzz_ratio(x[0], x[1]), zip(df['cb_name'], df['orbis_name'])))

    # Simple Token Jaccard
    df['name_token_jaccard'] = [token_jaccard(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]
    
    # Prefix match
    df['name_prefix_match'] = [prefix_match(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]
    
    # Acronym match
    df['name_acronym_match'] = [acronym_match(s1, s2) for s1, s2 in zip(df['cb_name'], df['orbis_name'])]

    # --- 2. Domain Features ---
    # Need to handle normalization properly
    def vec_domain_exact(d1, d2):
        if pd.isna(d1) or pd.isna(d2): return False
        return _normalize_domain(d1) == _normalize_domain(d2)
    
    df['domain_exact'] = [vec_domain_exact(d1, d2) for d1, d2 in zip(df['cb_domain'], df['orbis_website'])]
    df['email_domain_exact'] = [vec_domain_exact(d1, d2) for d1, d2 in zip(df.get('cb_email_domain', pd.Series([None]*len(df))), df['orbis_email'])]
    
    # --- 3. Geo Features ---
    df['country_match'] = (df['cb_country_iso'].fillna('').astype(str).str.upper() == df['orbis_country'].fillna('').astype(str).str.upper())
    
    # City Sim
    df['city_sim'] = [city_similarity(c1, c2) for c1, c2 in zip(df.get('cb_city_norm', pd.Series(['']*len(df))), df.get('orbis_city', pd.Series(['']*len(df))))]
    
    # --- 4. Temporal ---
    # Vectorized year diff
    df['year_diff'] = (df['cb_founded_year'] - df['orbis_incorp_year']).abs()
    
    # Vectorized year compatibility
    conditions = [
        df['year_diff'].isna(),
        df['year_diff'] == 0,
        df['year_diff'] <= 2,
        df['year_diff'] <= 5
    ]
    choices = [0.5, 1.0, 0.8, 0.5]
    df['year_compat'] = np.select(conditions, choices, default=0.2)
    
    # --- 5. Structural ---
    df['is_guo'] = df['entity_role'].astype(str).str.upper() == 'GUO'
    df['is_subsidiary'] = df['entity_role'].astype(str).str.upper().isin(['SUBSIDIARY', 'SUB'])
    df['is_branch'] = df['entity_role'].astype(str).str.upper() == 'BRANCH'
    df['family_size_log'] = np.log1p(df['family_size'].fillna(0))
    
    # --- 6. Meta ---
    df['is_generic_name'] = df.get('cb_name_is_generic', False)
    df['is_free_email'] = df.get('cb_is_free_email', False)
    
    # --- 8. Industry & Description Overlap (P1) ---
    if 'cb_industries' in df.columns and 'orbis_trade_desc' in df.columns:
        df['industry_token_overlap'] = [
            industry_token_overlap(i, t) 
            for i, t in zip(df['cb_industries'].fillna(''), df['orbis_trade_desc'].fillna(''))
        ]
        df['desc_token_overlap'] = [
            description_token_overlap(d, t)
            for d, t in zip(df.get('cb_description', pd.Series(['']*len(df))).fillna(''), df['orbis_trade_desc'].fillna(''))
        ]
        df['has_trade_desc'] = df['orbis_trade_desc'].notna() & (df['orbis_trade_desc'].astype(str).str.len() > 0)
    else:
        df['industry_token_overlap'] = 0.0
        df['desc_token_overlap'] = 0.0
        df['has_trade_desc'] = False
    
    # --- 7. Embeddings ---
    df['desc_embedding_cos'] = 0.0
    if embedding_resources and 'cb_emb' in embedding_resources and 'orbis_emb' in embedding_resources:
        try:
            # Map IDs to indices
            # If map lookup fails (key error), we get NaN -> fill with -1 or discard
            # We use map().fillna(-1)
            cb_indices = df['cb_id'].map(embedding_resources['cb_id_map']).fillna(-1).astype(int)
            orbis_indices = df['bvd_id'].map(embedding_resources['orbis_id_map']).fillna(-1).astype(int)
            
            # Mask valid
            valid_mask = (cb_indices != -1) & (orbis_indices != -1)
            
            if valid_mask.any():
                # Gather vectors (mmap read)
                # Note: mmap slicing with a list of indices is slightly slower than contiguous, 
                # but much faster than loop.
                v1 = embedding_resources['cb_emb'][cb_indices[valid_mask]]
                v2 = embedding_resources['orbis_emb'][orbis_indices[valid_mask]]
                
                # Compute Cosine: dot(u, v) / (norm(u)*norm(v))
                # SentenceTransformers are usually normalized? Let's assume yes or normalize.
                # Just to be safe, we compute full cosine.
                
                # Norms
                norm1 = np.linalg.norm(v1, axis=1)
                norm2 = np.linalg.norm(v2, axis=1)
                
                dot = np.sum(v1 * v2, axis=1)
                
                # Avoid div by zero
                denom = norm1 * norm2
                denom[denom == 0] = 1e-9
                
                sims = dot / denom
                
                # Assign back
                # We need to assign to the rows where valid_mask is True
                df.loc[valid_mask, 'desc_embedding_cos'] = sims
                
        except Exception as e:
            logger.error(f"Vectorized embedding compute failed: {e}")
    
    # Disambiguation Score (Learned)
    disambiguator = get_disambiguator()
    if disambiguator._is_fitted:
        X = np.column_stack([
            df['domain_exact'].astype(float),
            df['country_match'].astype(float),
            df['city_sim'].astype(float),
            df['year_compat'].astype(float)
        ])
        X_scaled = disambiguator.scaler.transform(X)
        df['disambiguation_score'] = disambiguator.model.predict_proba(X_scaled)[:, 1]
    else:
        df['disambiguation_score'] = 0.5

    df['alias_best_sim'] = 0.0
    if alias_registry is not None:
        # Compute alias similarity: check if orbis_name matches any CB alias
        try:
            alias_sims = []
            for _, row in df[['cb_id', 'orbis_name']].iterrows():
                cb_id = row['cb_id']
                orbis_name = str(row['orbis_name']).lower().strip()
                best_sim = 0.0
                aliases = alias_registry.get_aliases(cb_id) if hasattr(alias_registry, 'get_aliases') else []
                for alias in aliases:
                    alias_lower = str(alias).lower().strip()
                    if alias_lower and orbis_name:
                        # Token overlap ratio
                        tokens_a = set(alias_lower.split())
                        tokens_b = set(orbis_name.split())
                        if tokens_a and tokens_b:
                            overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
                            best_sim = max(best_sim, overlap)
                alias_sims.append(best_sim)
            df['alias_best_sim'] = alias_sims
            n_with_alias = sum(1 for s in alias_sims if s > 0)
            logger.info(f"Computed alias_best_sim: {n_with_alias} pairs with alias signal")
        except Exception as e:
            logger.warning(f"Alias similarity failed: {e}")

    # --- TF-IDF Name Similarity (CPU-friendly semantic feature) ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos_sim
        
        all_names = pd.concat([df['cb_name'], df['orbis_name']]).fillna('').tolist()
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50000)
        tfidf.fit(all_names)
        
        cb_vecs = tfidf.transform(df['cb_name'].fillna('').tolist())
        orbis_vecs = tfidf.transform(df['orbis_name'].fillna('').tolist())
        
        # Row-wise cosine similarity
        tfidf_sims = np.array([
            sklearn_cos_sim(cb_vecs[i:i+1], orbis_vecs[i:i+1])[0, 0]
            for i in range(len(df))
        ])
        df['name_tfidf_cos'] = tfidf_sims
        logger.info(f"Computed TF-IDF char-ngram similarity for {len(df)} pairs")
    except Exception as e:
        logger.warning(f"TF-IDF feature failed: {e}")
        df['name_tfidf_cos'] = 0.0

    # Keep only feature columns + IDs
    feature_cols = [
        'name_jw', 'name_token_jaccard', 'name_rapidfuzz_ratio', 'name_prefix_match', 'name_acronym_match',
        'name_tfidf_cos',
        'alias_best_sim',
        'domain_exact', 'email_domain_exact',
        'country_match', 'city_sim',
        'year_diff', 'year_compat',
        'desc_embedding_cos', # Added
        'is_guo', 'is_subsidiary', 'is_branch', 'family_size_log',
        'disambiguation_score',
        'is_generic_name', 'is_free_email'
    ]
    
    # Ensure all columns exist
    for col in feature_cols:
         if col not in df.columns:
             df[col] = 0.0
             
    # Keep IDs + names (needed by rerank) + features
    id_cols = ['cb_id', 'bvd_id']
    name_cols = [c for c in ['cb_name', 'orbis_name'] if c in df.columns]
    final_cols = id_cols + name_cols + feature_cols
    return df[final_cols]


if __name__ == '__main__':
    # Test feature computation
    cb = {
        'cb_name': 'Deliveroo',
        'cb_domain': 'deliveroo.com',
        'cb_country_iso': 'GB',
        'cb_city_norm': 'london',
        'cb_founded_year': 2013,
    }
    
    orbis = {
        'orbis_name': 'DELIVEROO HOLDINGS PLC',
        'orbis_website': 'www.deliveroo.com',
        'orbis_country': 'GB',
        'orbis_city': 'London',
        'orbis_incorp_year': 2013,
        'entity_role': 'GUO',
        'family_size': 15,
    }
    
    features = compute_pair_features(cb, orbis)
    
    print("=== Feature Test ===")
    for k, v in features.items():
        print(f"  {k}: {v}")
