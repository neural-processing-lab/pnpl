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
from typing import Dict, Any, Optional
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# Default constants (fallback values)
DEFAULT_CONSTANTS = {
    "PHONEMES": ['aa', 'ae', 'ah', 'ao', 'aw', 'ax-h', 'ax', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih',
                'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'sil', 'h#', 'epi', 'pau'],
    "PHONEME_LABELS_SORTED": ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'],
    "PHONATION_BY_PHONEME": {'aa': 'v', 'ae': 'v', 'ah': 'v', 'ao': 'v', 'aw': 'v', 'ax-h': 'v', 'ax': 'v', 'axr': 'v', 'ay': 'v', 'b': 'v', 'bcl': 'uv', 'ch': 'uv', 'd': 'v', 'dcl': 'uv', 'dh': 'v', 'dx': 'v', 'eh': 'v', 'el': 'v', 'em': 'v', 'en': 'v', 'eng': 'v', 'er': 'v', 'ey': 'v', 'f': 'uv', 'g': 'v', 'gcl': 'uv', 'hh': 'uv', 'hv': 'v', 'ih': 'v', 'ix': 'v',
                            'iy': 'v', 'jh': 'v', 'k': 'uv', 'kcl': 'uv', 'l': 'v', 'm': 'v', 'n': 'v', 'ng': 'v', 'nx': 'v', 'ow': 'v', 'oy': 'v', 'p': 'uv', 'pcl': 'uv', 'q': 'uv', 'r': 'v', 's': 'uv', 'sh': 'uv', 't': 'uv', 'tcl': 'uv', 'th': 'v', 'uh': 'v', 'uw': 'v', 'ux': 'v', 'v': 'v', 'w': 'v', 'y': 'v', 'z': 'v', 'zh': 'v', 'sil': 's', 'h#': 's', 'epi': 's', 'pau': 's'},
    "RUN_KEYS": [
        ('0', '1', 'Sherlock1', '1'), ('0', '2', 'Sherlock1', '1'), ('0', '3', 'Sherlock1', '1'),
        ('0', '4', 'Sherlock1', '1'), ('0', '5', 'Sherlock1', '1'), ('0', '6', 'Sherlock1', '1'),
        ('0', '7', 'Sherlock1', '1'), ('0', '8', 'Sherlock1', '1'), ('0', '9', 'Sherlock1', '1'),
        ('0', '10', 'Sherlock1', '1'), ('0', '11', 'Sherlock1', '2'), ('0', '12', 'Sherlock1', '2'),
        ('0', '1', 'Sherlock2', '1'), ('0', '2', 'Sherlock2', '1'), ('0', '3', 'Sherlock2', '1'),
        ('0', '4', 'Sherlock2', '1'), ('0', '5', 'Sherlock2', '1'), ('0', '6', 'Sherlock2', '1'),
        ('0', '7', 'Sherlock2', '1'), ('0', '8', 'Sherlock2', '1'), ('0', '9', 'Sherlock2', '1'),
        ('0', '10', 'Sherlock2', '1'), ('0', '11', 'Sherlock2', '1'), ('0', '12', 'Sherlock2', '1'),
        ('0', '1', 'Sherlock3', '1'), ('0', '2', 'Sherlock3', '1'), ('0', '3', 'Sherlock3', '1'),
        ('0', '4', 'Sherlock3', '1'), ('0', '5', 'Sherlock3', '1'), ('0', '6', 'Sherlock3', '1'),
        ('0', '7', 'Sherlock3', '1'), ('0', '8', 'Sherlock3', '1'), ('0', '9', 'Sherlock3', '1'),
        ('0', '10', 'Sherlock3', '1'), ('0', '11', 'Sherlock3', '1'), ('0', '12', 'Sherlock3', '1'),
        ('0', '1', 'Sherlock4', '1'), ('0', '3', 'Sherlock4', '1'),
        ('0', '4', 'Sherlock4', '1'), ('0', '5', 'Sherlock4', '1'), ('0', '6', 'Sherlock4', '1'),
        ('0', '7', 'Sherlock4', '1'), ('0', '8', 'Sherlock4', '1'), ('0', '10', 'Sherlock4', '1'),
        ('0', '11', 'Sherlock4', '1'), ('0', '12', 'Sherlock4', '1'),
        ('0', '1', 'Sherlock5', '1'), ('0', '2', 'Sherlock5', '1'), ('0', '3', 'Sherlock5', '1'),
        ('0', '4', 'Sherlock5', '1'), ('0', '5', 'Sherlock5', '1'), ('0', '6', 'Sherlock5', '1'),
        ('0', '7', 'Sherlock5', '1'), ('0', '8', 'Sherlock5', '1'), ('0', '9', 'Sherlock5', '1'),
        ('0', '10', 'Sherlock5', '1'), ('0', '11', 'Sherlock5', '1'), ('0', '12', 'Sherlock5', '1'),
        ('0', '13', 'Sherlock5', '1'), ('0', '14', 'Sherlock5', '1'), ('0', '15', 'Sherlock5', '1'),
        ('0', '1', 'Sherlock6', '1'), ('0', '2', 'Sherlock6', '1'),
        ('0', '4', 'Sherlock6', '1'), ('0', '5', 'Sherlock6', '1'), ('0', '6', 'Sherlock6', '1'),
        ('0', '7', 'Sherlock6', '1'), ('0', '8', 'Sherlock6', '1'), ('0', '9', 'Sherlock6', '1'),
        ('0', '11', 'Sherlock6', '1'), ('0', '12', 'Sherlock6', '1'), ('0', '13', 'Sherlock6', '1'),
        ('0', '1', 'Sherlock7', '1'), ('0', '2', 'Sherlock7', '1'), ('0', '3', 'Sherlock7', '1'),
        ('0', '4', 'Sherlock7', '1'), ('0', '5', 'Sherlock7', '1'), ('0', '6', 'Sherlock7', '1'),
        ('0', '7', 'Sherlock7', '1'), ('0', '8', 'Sherlock7', '1'), ('0', '9', 'Sherlock7', '1'),
        ('0', '10', 'Sherlock7', '1'), ('0', '11', 'Sherlock7', '1'), ('0', '12', 'Sherlock7', '1'),
        ('0', '13', 'Sherlock7', '1'), ('0', '14', 'Sherlock7', '1'),
    ],
    "VALIDATION_RUN_KEYS": [('0', '11', 'Sherlock1', '2')],
    "TEST_RUN_KEYS": [('0', '12', 'Sherlock1', '2')],
    "PHONEME_CLASSES": 39,
    "SPEECH_OUTPUT_DIM": 1,
    "PHONEME_HOLDOUT_PREDICTIONS": 6862,
    "SPEECH_HOLDOUT_PREDICTIONS": 1043
}


class RemoteConstantsManager:
    """Manager for fetching and caching constants from remote JSON sources."""
    
    def __init__(self):
        self._cached_constants = None
        
    @property
    def remote_url(self):
        default_url = "https://neural-processing-lab.github.io/2025-libribrain-competition/constants.json"
        return os.environ.get('PNPL_REMOTE_CONSTANTS_URL', default_url)
        
    @property
    def disabled(self):
        return os.environ.get('PNPL_REMOTE_CONSTANTS_DISABLED', '').lower() == 'true'
        
    @property
    def cache_dir(self):
        default_cache_dir = self._get_default_cache_dir()
        cache_dir = Path(os.environ.get('PNPL_CACHE_DIR', default_cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _get_default_cache_dir(self):
        import platform
        if platform.system() == 'Windows':
            appdata = os.environ.get('LOCALAPPDATA')
            if appdata:
                return Path(appdata) / 'pnpl' / 'cache'
            return Path.home() / 'AppData' / 'Local' / 'pnpl' / 'cache'
        return Path.home() / '.pnpl' / 'cache'
        
    @property
    def cache_timeout(self):
        return int(os.environ.get('PNPL_CACHE_TIMEOUT', '86400'))
        
    @property
    def cache_file(self):
        return self.cache_dir / 'libribrain_constants.json'
        
    @property
    def cache_meta_file(self):
        return self.cache_dir / 'libribrain_constants_meta.json'
        
    def _fetch_remote_constants(self) -> Optional[Dict[str, Any]]:
        if not self.remote_url or self.disabled:
            return None
        if not HAS_REQUESTS:
            return None
            
        try:
            if self.remote_url.startswith('file://'):
                with open(self.remote_url[7:], 'r') as f:
                    data = json.load(f)
            else:
                response = requests.get(self.remote_url, timeout=30)
                response.raise_for_status()
                data = response.json()
            
            if not isinstance(data, dict):
                return None
                
            # Convert list to tuples for run keys
            for key in ['RUN_KEYS', 'VALIDATION_RUN_KEYS', 'TEST_RUN_KEYS']:
                if key in data and isinstance(data[key], list):
                    data[key] = [tuple(item) if isinstance(item, list) else item for item in data[key]]
            
            return data
        except Exception:
            return None
            
    def _save_cache(self, constants: Dict[str, Any]) -> None:
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(constants, f, indent=2)
            meta = {'timestamp': time.time(), 'source_url': self.remote_url}
            with open(self.cache_meta_file, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
            
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.cache_file.exists() or not self.cache_meta_file.exists():
                return None
            with open(self.cache_meta_file, 'r') as f:
                meta = json.load(f)
            if time.time() - meta['timestamp'] > self.cache_timeout:
                return None
            with open(self.cache_file, 'r') as f:
                constants = json.load(f)
            for key in ['RUN_KEYS', 'VALIDATION_RUN_KEYS', 'TEST_RUN_KEYS']:
                if key in constants and isinstance(constants[key], list):
                    constants[key] = [tuple(item) if isinstance(item, list) else item for item in constants[key]]
            return constants
        except Exception:
            return None
            
    def get_constants(self) -> Dict[str, Any]:
        if self._cached_constants is not None:
            return self._cached_constants
            
        constants = self._load_cache()
        if constants is None:
            remote = self._fetch_remote_constants()
            if remote is not None:
                constants = remote
                self._save_cache(constants)
            else:
                constants = self._load_cache()
                
        if constants is None:
            constants = DEFAULT_CONSTANTS.copy()
        else:
            merged = DEFAULT_CONSTANTS.copy()
            merged.update(constants)
            constants = merged
            
        self._cached_constants = constants
        return constants
        
    def force_refresh(self) -> Dict[str, Any]:
        self._cached_constants = None
        remote = self._fetch_remote_constants()
        if remote is not None:
            self._save_cache(remote)
            constants = remote
        else:
            constants = DEFAULT_CONSTANTS.copy()
        merged = DEFAULT_CONSTANTS.copy()
        merged.update(constants)
        self._cached_constants = merged
        return merged


_manager = RemoteConstantsManager()

def get_constants() -> Dict[str, Any]:
    return _manager.get_constants()

def force_refresh_constants() -> Dict[str, Any]:
    return _manager.force_refresh()
