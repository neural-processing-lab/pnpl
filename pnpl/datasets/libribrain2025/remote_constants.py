"""
Remote constants management for the LibriBrain dataset.

This module provides functionality to fetch constants from a remote JSON source
and cache them locally, with fallback to hardcoded constants if the remote
source is unavailable.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Configure logging
logger = logging.getLogger(__name__)

# Default constants (fallback values)
DEFAULT_CONSTANTS = {
    "PHONEMES": ['aa', 'ae', 'ah', 'ao', 'aw', 'ax-h', 'ax', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih',
                'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'sil', 'h#', 'epi', 'pau'],
    "PHONATION_BY_PHONEME": {'aa': 'v', 'ae': 'v', 'ah': 'v', 'ao': 'v', 'aw': 'v', 'ax-h': 'v', 'ax': 'v', 'axr': 'v', 'ay': 'v', 'b': 'v', 'bcl': 'uv', 'ch': 'uv', 'd': 'v', 'dcl': 'uv', 'dh': 'v', 'dx': 'v', 'eh': 'v', 'el': 'v', 'em': 'v', 'en': 'v', 'eng': 'v', 'er': 'v', 'ey': 'v', 'f': 'uv', 'g': 'v', 'gcl': 'uv', 'hh': 'uv', 'hv': 'v', 'ih': 'v', 'ix': 'v',
                            'iy': 'v', 'jh': 'v', 'k': 'uv', 'kcl': 'uv', 'l': 'v', 'm': 'v', 'n': 'v', 'ng': 'v', 'nx': 'v', 'ow': 'v', 'oy': 'v', 'p': 'uv', 'pcl': 'uv', 'q': 'uv', 'r': 'v', 's': 'uv', 'sh': 'uv', 't': 'uv', 'tcl': 'uv', 'th': 'v', 'uh': 'v', 'uw': 'v', 'ux': 'v', 'v': 'v', 'w': 'v', 'y': 'v', 'z': 'v', 'zh': 'v', 'sil': 's', 'h#': 's', 'epi': 's', 'pau': 's'},
    "RUN_KEYS": [('0', '1', 'Sherlock1', '1'),
                ('0', '2', 'Sherlock1', '1'),
                ('0', '3', 'Sherlock1', '1'),
                ('0', '4', 'Sherlock1', '1'),
                ('0', '5', 'Sherlock1', '1'),
                ('0', '6', 'Sherlock1', '1'),
                ('0', '7', 'Sherlock1', '1'),
                ('0', '8', 'Sherlock1', '1'),
                ('0', '9', 'Sherlock1', '1'),
                ('0', '10', 'Sherlock1', '1'),
                ('0', '11', 'Sherlock1', '2'),
                ('0', '12', 'Sherlock1', '2'),
                ('0', '1', 'Sherlock2', '1'),
                ('0', '2', 'Sherlock2', '1'),
                ('0', '3', 'Sherlock2', '1'),
                ('0', '4', 'Sherlock2', '1'),
                ('0', '5', 'Sherlock2', '1'),
                ('0', '6', 'Sherlock2', '1'),
                ('0', '7', 'Sherlock2', '1'),
                ('0', '8', 'Sherlock2', '1'),
                ('0', '9', 'Sherlock2', '1'),
                ('0', '10', 'Sherlock2', '1'),
                ('0', '11', 'Sherlock2', '1'),
                ('0', '12', 'Sherlock2', '1'),
                ('0', '1', 'Sherlock3', '1'),
                ('0', '2', 'Sherlock3', '1'),
                ('0', '3', 'Sherlock3', '1'),
                ('0', '4', 'Sherlock3', '1'),
                ('0', '5', 'Sherlock3', '1'),
                ('0', '6', 'Sherlock3', '1'),
                ('0', '7', 'Sherlock3', '1'),
                ('0', '8', 'Sherlock3', '1'),
                ('0', '9', 'Sherlock3', '1'),
                ('0', '10', 'Sherlock3', '1'),
                ('0', '11', 'Sherlock3', '1'),
                ('0', '12', 'Sherlock3', '1'),
                ('0', '1', 'Sherlock4', '1'),
                ('0', '2', 'Sherlock4', '1'),
                ('0', '3', 'Sherlock4', '1'),
                ('0', '4', 'Sherlock4', '1'),
                ('0', '5', 'Sherlock4', '1'),
                ('0', '6', 'Sherlock4', '1'),
                ('0', '7', 'Sherlock4', '1'),
                # ('0', '8', 'Sherlock4', '1'),
                ('0', '9', 'Sherlock4', '1'),
                ('0', '10', 'Sherlock4', '1'),
                ('0', '11', 'Sherlock4', '1'),
                ('0', '12', 'Sherlock4', '1'),
                ('0', '1', 'Sherlock5', '1'),
                ('0', '2', 'Sherlock5', '1'),
                ('0', '3', 'Sherlock5', '1'),
                ('0', '4', 'Sherlock5', '1'),
                ('0', '5', 'Sherlock5', '1'),
                ('0', '6', 'Sherlock5', '1'),
                ('0', '7', 'Sherlock5', '1'),
                ('0', '8', 'Sherlock5', '1'),
                ('0', '9', 'Sherlock5', '1'),
                ('0', '10', 'Sherlock5', '1'),
                ('0', '11', 'Sherlock5', '1'),
                ('0', '12', 'Sherlock5', '1'),
                ('0', '13', 'Sherlock5', '1'),
                ('0', '14', 'Sherlock5', '1'),
                ('0', '15', 'Sherlock5', '1'),
                ('0', '1', 'Sherlock6', '1'),
                ('0', '2', 'Sherlock6', '1'),
                ('0', '3', 'Sherlock6', '1'),
                ('0', '4', 'Sherlock6', '1'),
                ('0', '5', 'Sherlock6', '1'),
                ('0', '6', 'Sherlock6', '1'),
                ('0', '7', 'Sherlock6', '1'),
                ('0', '8', 'Sherlock6', '1'),
                ('0', '9', 'Sherlock6', '1'),
                ('0', '10', 'Sherlock6', '1'),
                ('0', '11', 'Sherlock6', '1'),
                ('0', '12', 'Sherlock6', '1'),
                ('0', '13', 'Sherlock6', '1'),
                ('0', '14', 'Sherlock6', '1'),
                ('0', '1', 'Sherlock7', '1'),
                ('0', '2', 'Sherlock7', '1'),
                ('0', '3', 'Sherlock7', '1'),
                ('0', '4', 'Sherlock7', '1'),
                ('0', '5', 'Sherlock7', '1'),
                ('0', '6', 'Sherlock7', '1'),
                ('0', '7', 'Sherlock7', '1'),
                ('0', '8', 'Sherlock7', '1'),
                ('0', '9', 'Sherlock7', '1'),
                ('0', '10', 'Sherlock7', '1'),
                ('0', '11', 'Sherlock7', '1'),
                ('0', '12', 'Sherlock7', '1'),
                ('0', '13', 'Sherlock7', '1'),
                ('0', '14', 'Sherlock7', '1'),
                ],
    "VALIDATION_RUN_KEYS": [
        ('0', '11', 'Sherlock1', '2'),
    ],
    "TEST_RUN_KEYS": [
        ('0', '12', 'Sherlock1', '2')
    ],
    "PHONEME_CLASSES": 39,
    "SPEECH_OUTPUT_DIM": 1,
    "PHONEME_HOLDOUT_PREDICTIONS": 2382,
    "SPEECH_HOLDOUT_PREDICTIONS": 560638
}


class RemoteConstantsManager:
    """
    Manager for fetching and caching constants from remote JSON sources.
    
    By default, fetches constants from the official LibriBrain2025 competition URL.
    
    Environment variables:
    - PNPL_REMOTE_CONSTANTS_URL: URL to fetch constants from (default: https://neural-processing-lab.github.io/2025-libribrain-competition/constants.json)
    - PNPL_REMOTE_CONSTANTS_DISABLED: Set to "true" to disable remote fetching
    - PNPL_CACHE_DIR: Directory to store cached constants (default: ~/.pnpl/cache)
    - PNPL_CACHE_TIMEOUT: Cache timeout in seconds (default: 86400 = 24 hours)
    """
    
    def __init__(self):
        # Cache for the session
        self._cached_constants = None
        
    @property
    def remote_url(self):
        # Default URL for LibriBrain2025 constants
        default_url = "https://neural-processing-lab.github.io/2025-libribrain-competition/constants.json"
        return os.environ.get('PNPL_REMOTE_CONSTANTS_URL', default_url)
        
    @property
    def disabled(self):
        return os.environ.get('PNPL_REMOTE_CONSTANTS_DISABLED', '').lower() == 'true'
        
    @property
    def cache_dir(self):
        default_cache_dir = self._get_default_cache_dir()
        cache_dir = Path(os.environ.get('PNPL_CACHE_DIR', default_cache_dir))
        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _get_default_cache_dir(self):
        """Get platform-appropriate default cache directory."""
        import platform
        
        if platform.system() == 'Windows':
            # Use AppData/Local on Windows
            appdata = os.environ.get('LOCALAPPDATA')
            if appdata:
                return Path(appdata) / 'pnpl' / 'cache'
            else:
                # Fallback to home directory
                return Path.home() / 'AppData' / 'Local' / 'pnpl' / 'cache'
        else:
            # Unix-like systems (macOS, Linux)
            return Path.home() / '.pnpl' / 'cache'
        
    @property
    def cache_timeout(self):
        return int(os.environ.get('PNPL_CACHE_TIMEOUT', '86400'))  # 24 hours
        
    @property
    def cache_file(self):
        return self.cache_dir / 'libribrain2025_constants.json'
        
    @property
    def cache_meta_file(self):
        return self.cache_dir / 'libribrain2025_constants_meta.json'
        
    def _fetch_remote_constants(self) -> Optional[Dict[str, Any]]:
        """Fetch constants from remote URL."""
        if not self.remote_url or self.disabled:
            return None
            
        if not HAS_REQUESTS:
            logger.warning("requests library not available, cannot fetch remote constants")
            return None
            
        try:
            logger.info(f"Fetching constants from {self.remote_url}")
            
            # Handle file:// URLs specially
            if self.remote_url.startswith('file://'):
                file_path = self.remote_url[7:]  # Remove 'file://' prefix
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                response = requests.get(self.remote_url, timeout=30)
                response.raise_for_status()
                data = response.json()
            
            # Validate the JSON structure
            if not isinstance(data, dict):
                logger.error("Remote constants must be a JSON object")
                return None
                
            # Convert RUN_KEYS from list of lists to list of tuples if needed
            if 'RUN_KEYS' in data and isinstance(data['RUN_KEYS'], list):
                data['RUN_KEYS'] = [tuple(item) if isinstance(item, list) else item for item in data['RUN_KEYS']]
                
            if 'VALIDATION_RUN_KEYS' in data and isinstance(data['VALIDATION_RUN_KEYS'], list):
                data['VALIDATION_RUN_KEYS'] = [tuple(item) if isinstance(item, list) else item for item in data['VALIDATION_RUN_KEYS']]
                
            if 'TEST_RUN_KEYS' in data and isinstance(data['TEST_RUN_KEYS'], list):
                data['TEST_RUN_KEYS'] = [tuple(item) if isinstance(item, list) else item for item in data['TEST_RUN_KEYS']]
            
            logger.info("Successfully fetched remote constants")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch remote constants: {e}")
            return None
            
    def _save_cache(self, constants: Dict[str, Any]) -> None:
        """Save constants to cache."""
        try:
            # Save constants
            with open(self.cache_file, 'w') as f:
                json.dump(constants, f, indent=2)
                
            # Save metadata
            meta = {
                'timestamp': time.time(),
                'source_url': self.remote_url
            }
            with open(self.cache_meta_file, 'w') as f:
                json.dump(meta, f, indent=2)
                
            logger.info("Constants cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to cache constants: {e}")
            
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load constants from cache if valid."""
        try:
            if not self.cache_file.exists() or not self.cache_meta_file.exists():
                return None
                
            # Check cache metadata
            with open(self.cache_meta_file, 'r') as f:
                meta = json.load(f)
                
            # Check if cache is expired
            if time.time() - meta['timestamp'] > self.cache_timeout:
                logger.info("Cache expired, will fetch fresh constants")
                return None
                
            # Load constants
            with open(self.cache_file, 'r') as f:
                constants = json.load(f)
                
            # Convert RUN_KEYS from list of lists to list of tuples if needed
            if 'RUN_KEYS' in constants and isinstance(constants['RUN_KEYS'], list):
                constants['RUN_KEYS'] = [tuple(item) if isinstance(item, list) else item for item in constants['RUN_KEYS']]
                
            if 'VALIDATION_RUN_KEYS' in constants and isinstance(constants['VALIDATION_RUN_KEYS'], list):
                constants['VALIDATION_RUN_KEYS'] = [tuple(item) if isinstance(item, list) else item for item in constants['VALIDATION_RUN_KEYS']]
                
            if 'TEST_RUN_KEYS' in constants and isinstance(constants['TEST_RUN_KEYS'], list):
                constants['TEST_RUN_KEYS'] = [tuple(item) if isinstance(item, list) else item for item in constants['TEST_RUN_KEYS']]
                
            logger.info("Loaded constants from cache")
            return constants
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
            
    def get_constants(self) -> Dict[str, Any]:
        """
        Get constants, trying remote source first, then cache, then defaults.
        
        Returns:
            Dict containing all constants
        """
        # Return cached constants if already loaded in this session
        if self._cached_constants is not None:
            return self._cached_constants
            
        constants = None
        
        # Try to load from cache first
        constants = self._load_cache()
        
        # If cache is expired or doesn't exist, try to fetch fresh constants
        if constants is None:
            remote_constants = self._fetch_remote_constants()
            if remote_constants is not None:
                constants = remote_constants
                self._save_cache(constants)
            else:
                # If remote fetch fails, try cache again (even if expired)
                constants = self._load_cache()
                
        # Fall back to default constants if all else fails
        if constants is None:
            logger.info("Using default constants")
            constants = DEFAULT_CONSTANTS.copy()
        else:
            # Merge with defaults to ensure all keys are present
            merged_constants = DEFAULT_CONSTANTS.copy()
            merged_constants.update(constants)
            constants = merged_constants
            
        # Cache for the session
        self._cached_constants = constants
        
        return constants
        
    def force_refresh(self) -> Dict[str, Any]:
        """Force refresh constants from remote source."""
        # Clear session cache
        self._cached_constants = None
        
        # Try to fetch fresh constants
        remote_constants = self._fetch_remote_constants()
        if remote_constants is not None:
            self._save_cache(remote_constants)
            constants = remote_constants
        else:
            # Fall back to defaults
            constants = DEFAULT_CONSTANTS.copy()
            
        # Merge with defaults
        merged_constants = DEFAULT_CONSTANTS.copy()
        merged_constants.update(constants)
        
        # Cache for the session
        self._cached_constants = merged_constants
        
        return merged_constants


# Global instance
_manager = RemoteConstantsManager()

def get_constants() -> Dict[str, Any]:
    """Get constants using the global manager."""
    return _manager.get_constants()

def force_refresh_constants() -> Dict[str, Any]:
    """Force refresh constants from remote source."""
    return _manager.force_refresh() 