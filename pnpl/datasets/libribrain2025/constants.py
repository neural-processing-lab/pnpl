"""
Constants for LibriBrain dataset.

This module provides constants that can be updated remotely without re-releasing
the package. Constants are fetched from a remote JSON source and cached locally.

Environment variables:
- PNPL_REMOTE_CONSTANTS_URL: URL to fetch constants from
- PNPL_REMOTE_CONSTANTS_DISABLED: Set to "true" to disable remote fetching
- PNPL_CACHE_DIR: Directory to store cached constants (default: ~/.pnpl/cache)
- PNPL_CACHE_TIMEOUT: Cache timeout in seconds (default: 86400 = 24 hours)
"""

from .remote_constants import _manager, get_constants


class ConstantsAccessor:
    """Dynamic accessor for constants that can be refreshed."""

    def __init__(self):
        self._cached_constants = None
        self._manager_cache_id = None

    def _get_constants(self):
        """Get constants, refreshing if the remote manager cache has changed."""
        current_cache_id = id(_manager._cached_constants)

        if self._cached_constants is None or self._manager_cache_id != current_cache_id:
            self._cached_constants = get_constants()
            self._manager_cache_id = current_cache_id

        return self._cached_constants

    def get_constant(self, name: str):
        """Get a constant by name."""
        constants = self._get_constants()

        if name == "SPEECH_CLASSES":
            return constants["SPEECH_OUTPUT_DIM"]

        if name in constants:
            return constants[name]

        raise AttributeError(f"No constant named '{name}'")


_accessor = ConstantsAccessor()

__all__ = [
    "PHONEMES",
    "PHONEME_LABELS_SORTED",
    "PHONATION_BY_PHONEME",
    "RUN_KEYS",
    "VALIDATION_RUN_KEYS",
    "TEST_RUN_KEYS",
    "PHONEME_CLASSES",
    "SPEECH_OUTPUT_DIM",
    "PHONEME_HOLDOUT_PREDICTIONS",
    "SPEECH_HOLDOUT_PREDICTIONS",
    "SPEECH_CLASSES",
    "refresh_module_constants",
]


def __getattr__(name: str):
    """Dynamic attribute access for constants."""
    if name in __all__[:-1]:
        return _accessor.get_constant(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def refresh_module_constants():
    """Force refresh of module-level constants."""
    _accessor._cached_constants = None
    _accessor._manager_cache_id = None
