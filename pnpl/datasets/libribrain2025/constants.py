"""
Constants for LibriBrain2025 dataset.

This module provides constants that can be updated remotely without re-releasing
the package. Constants are fetched from a remote JSON source and cached locally.

Environment variables:
- PNPL_REMOTE_CONSTANTS_URL: URL to fetch constants from
- PNPL_REMOTE_CONSTANTS_DISABLED: Set to "true" to disable remote fetching
- PNPL_CACHE_DIR: Directory to store cached constants (default: ~/.pnpl/cache)
- PNPL_CACHE_TIMEOUT: Cache timeout in seconds (default: 86400 = 24 hours)
"""

from .remote_constants import get_constants, _manager


class ConstantsAccessor:
    """Dynamic accessor for constants that can be refreshed."""
    
    def __init__(self):
        self._cached_constants = None
        self._manager_cache_id = None
    
    def _get_constants(self):
        """Get constants, refreshing if the remote manager cache has changed."""
        # Check if the remote manager's cache has changed
        current_cache_id = id(_manager._cached_constants)
        
        if self._cached_constants is None or self._manager_cache_id != current_cache_id:
            self._cached_constants = get_constants()
            self._manager_cache_id = current_cache_id
            
        return self._cached_constants
    
    def get_constant(self, name: str):
        """Get a constant by name."""
        constants = self._get_constants()
        
        if name == 'SPEECH_CLASSES':
            # Legacy compatibility
            return constants['SPEECH_OUTPUT_DIM']
        
        if name in constants:
            return constants[name]
        
        raise AttributeError(f"No constant named '{name}'")


# Create the global accessor
_accessor = ConstantsAccessor()

# List of available constants for __all__
__all__ = [
    'PHONEMES', 'PHONATION_BY_PHONEME', 'RUN_KEYS', 'VALIDATION_RUN_KEYS',
    'TEST_RUN_KEYS', 'PHONEME_CLASSES', 'SPEECH_OUTPUT_DIM',
    'PHONEME_HOLDOUT_PREDICTIONS', 'SPEECH_HOLDOUT_PREDICTIONS', 'SPEECH_CLASSES',
    'refresh_module_constants'
]


def __getattr__(name: str):
    """
    Dynamic attribute access for constants.
    This allows the module to provide updated constants on each access.
    """
    if name in __all__[:-1]:  # Exclude 'refresh_module_constants' from dynamic lookup
        return _accessor.get_constant(name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def refresh_module_constants():
    """
    Force refresh of module-level constants.
    Call this after updating remote constants to refresh cached values.
    """
    _accessor._cached_constants = None
    _accessor._manager_cache_id = None


# Note: Module-level constants are dynamically provided by __getattr__
# This allows them to be updated when remote constants change
# IDEs might not recognize these attributes, but they work at runtime
