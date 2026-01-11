"""
config.py - Pipeline Configuration

Centralized configuration for the entity resolution pipeline.
Loads settings from YAML config files.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Paths (will be overwritten by actual paths)
    'paths': {
        'project_root': '.',
        'raw_crunchbase': 'data/raw/crunchbase',
        'raw_orbis': 'data/raw/orbis',
        'interim': 'data/interim',
        'outputs': 'data/outputs',
        'models': 'data/models',
    },
    
    # Blocking configuration
    'blocking': {
        'max_candidates_per_cb': 300,
        'ann_topk_name': 100,
        'ann_topk_desc': 50,
        'rare_token_threshold': 10000,
    },
    
    # Embedding configuration
    'embeddings': {
        'enabled': True,
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 512,
        'dtype': 'float16',
        'device': 'cpu',  # 'cuda' for GPU
    },
    
    # FAISS configuration
    'faiss': {
        'use_gpu': False,
        'nprobe': 16,
        'metric': 'inner_product',
    },
    
    # Tier thresholds
    'tiers': {
        'A': 0.98,
        'B': 0.93,
        'C': 0.75,
    },
    
    # Feature toggles
    'features': {
        'enable_investor_checks': True,
        'enable_semantic_embeddings': True,
        'enable_family_expansion': True,
    },
    
    # Model configuration
    'model': {
        'type': 'gradient_boosting',
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'calibration': 'isotonic',
    },
    
    # Processing
    'processing': {
        'chunk_size': 50000,
        'max_workers': 4,
        'memory_limit_gb': 12,  # For 16GB machine, leave headroom
    },
    
    # Random seed
    'random_seed': 42,
    
    # Logging
    'logging': {
        'level': 'INFO',
        'save_timing': True,
    },
}


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

class Config:
    """Configuration manager for the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to YAML config file, or None for defaults
        """
        self._config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self._load_yaml(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.info("Using default configuration")
    
    def _load_yaml(self, path: str) -> None:
        """Load and merge YAML config."""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        self._merge_recursive(self._config, yaml_config)
    
    def _merge_recursive(self, base: Dict, override: Dict) -> None:
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_recursive(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Example: config.get('blocking.max_candidates_per_cb')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set config value using dot notation.
        
        Example: config.set('embeddings.device', 'cuda')
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Get full config as dict."""
        return self._config.copy()
    
    def save(self, path: str) -> None:
        """Save current config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
        logger.info(f"Saved config to {path}")


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_project_paths(config: Config) -> Dict[str, Path]:
    """
    Get resolved project paths from config.
    
    Returns dict with:
        - project_root
        - raw_crunchbase
        - raw_orbis
        - interim
        - cb_clean
        - orbis_clean
        - embeddings
        - candidates
        - features
        - outputs
        - matches
        - review
        - reports
        - models
    """
    root = Path(config.get('paths.project_root', '.'))
    
    paths = {
        'project_root': root,
        'raw_crunchbase': root / config.get('paths.raw_crunchbase', 'data/raw/crunchbase'),
        'raw_orbis': root / config.get('paths.raw_orbis', 'data/raw/orbis'),
        'interim': root / config.get('paths.interim', 'data/interim'),
        'outputs': root / config.get('paths.outputs', 'data/outputs'),
        'models': root / config.get('paths.models', 'data/models'),
    }
    
    # Derived paths
    paths['cb_clean'] = paths['interim'] / 'cb_clean'
    paths['orbis_clean'] = paths['interim'] / 'orbis_clean'
    paths['embeddings'] = paths['interim'] / 'embeddings'
    paths['candidates'] = paths['interim'] / 'candidates'
    paths['features'] = paths['interim'] / 'features'
    paths['indexes'] = paths['interim'] / 'indexes'
    
    paths['matches'] = paths['outputs'] / 'matches'
    paths['review'] = paths['outputs'] / 'review'
    paths['reports'] = paths['outputs'] / 'reports'
    
    return paths


def ensure_directories(paths: Dict[str, Path]) -> None:
    """Create directories if they don't exist."""
    for name, path in paths.items():
        if not path.suffix:  # Skip file paths
            path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

_config: Optional[Config] = None


def init_config(config_path: Optional[str] = None) -> Config:
    """Initialize global config."""
    global _config
    _config = Config(config_path)
    return _config


def get_config() -> Config:
    """Get global config (initializes with defaults if not already)."""
    global _config
    if _config is None:
        _config = Config()
    return _config


if __name__ == '__main__':
    # Test config
    config = Config()
    
    print("=== Default Config ===")
    print(f"Max candidates: {config.get('blocking.max_candidates_per_cb')}")
    print(f"Tier A threshold: {config.get('tiers.A')}")
    print(f"Model type: {config.get('model.type')}")
    
    # Test path resolution
    paths = get_project_paths(config)
    print("\n=== Paths ===")
    for name, path in paths.items():
        print(f"  {name}: {path}")
