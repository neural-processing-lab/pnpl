"""
Utility functions for managing remote constants.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .remote_constants import force_refresh_constants, get_constants, _manager


def set_remote_constants_url(url: str) -> None:
    """
    Set the remote constants URL for the current session.
    
    Args:
        url: The URL to fetch constants from
    """
    os.environ['PNPL_REMOTE_CONSTANTS_URL'] = url
    # Clear the cached constants to force refresh
    _manager._cached_constants = None
    # Refresh the constants module
    from . import constants
    constants.refresh_module_constants()


def disable_remote_constants() -> None:
    """
    Disable remote constants fetching for the current session.
    """
    os.environ['PNPL_REMOTE_CONSTANTS_DISABLED'] = 'true'
    # Clear the cached constants to force refresh
    _manager._cached_constants = None
    # Refresh the constants module
    from . import constants
    constants.refresh_module_constants()


def enable_remote_constants() -> None:
    """
    Enable remote constants fetching for the current session.
    """
    os.environ.pop('PNPL_REMOTE_CONSTANTS_DISABLED', None)
    # Clear the cached constants to force refresh
    _manager._cached_constants = None
    # Refresh the constants module
    from . import constants
    constants.refresh_module_constants()


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state.
    
    Returns:
        Dict containing cache information
    """
    cache_info = {
        'cache_dir': str(_manager.cache_dir),
        'cache_file_exists': _manager.cache_file.exists(),
        'cache_meta_file_exists': _manager.cache_meta_file.exists(),
        'remote_url': _manager.remote_url,
        'disabled': _manager.disabled,
        'cache_timeout': _manager.cache_timeout,
        'cache_timestamp': None,
        'cache_source_url': None
    }
    
    if _manager.cache_meta_file.exists():
        try:
            with open(_manager.cache_meta_file, 'r') as f:
                meta = json.load(f)
            cache_info['cache_timestamp'] = meta.get('timestamp')
            cache_info['cache_source_url'] = meta.get('source_url')
        except Exception:
            cache_info['cache_timestamp'] = None
            cache_info['cache_source_url'] = None
    
    return cache_info


def clear_cache() -> None:
    """
    Clear the cached constants files.
    """
    if _manager.cache_file.exists():
        _manager.cache_file.unlink()
    if _manager.cache_meta_file.exists():
        _manager.cache_meta_file.unlink()
    # Clear the session cache
    _manager._cached_constants = None
    # Refresh the constants module
    from . import constants
    constants.refresh_module_constants()


def refresh_constants() -> Dict[str, Any]:
    """
    Force refresh constants from the remote source.
    
    Returns:
        Dict containing the refreshed constants
    """
    result = force_refresh_constants()
    # Refresh the constants module
    from . import constants
    constants.refresh_module_constants()
    return result


def validate_constants_json(constants_dict: Dict[str, Any]) -> bool:
    """
    Validate that a constants dictionary has the expected structure.
    
    Args:
        constants_dict: Dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = {
        'PHONEMES', 'PHONATION_BY_PHONEME', 'RUN_KEYS', 'VALIDATION_RUN_KEYS',
        'TEST_RUN_KEYS', 'PHONEME_CLASSES', 'SPEECH_OUTPUT_DIM',
        'PHONEME_HOLDOUT_PREDICTIONS', 'SPEECH_HOLDOUT_PREDICTIONS'
    }
    
    if not isinstance(constants_dict, dict):
        return False
        
    # Check that all required keys are present
    if not required_keys.issubset(constants_dict.keys()):
        return False
        
    # Validate data types
    if not isinstance(constants_dict['PHONEMES'], list):
        return False
        
    if not isinstance(constants_dict['PHONATION_BY_PHONEME'], dict):
        return False
        
    if not isinstance(constants_dict['RUN_KEYS'], list):
        return False
        
    if not isinstance(constants_dict['VALIDATION_RUN_KEYS'], list):
        return False
        
    if not isinstance(constants_dict['TEST_RUN_KEYS'], list):
        return False
        
    if not isinstance(constants_dict['PHONEME_CLASSES'], int):
        return False
        
    if not isinstance(constants_dict['SPEECH_OUTPUT_DIM'], int):
        return False
        
    if not isinstance(constants_dict['PHONEME_HOLDOUT_PREDICTIONS'], int):
        return False
        
    if not isinstance(constants_dict['SPEECH_HOLDOUT_PREDICTIONS'], int):
        return False
    
    return True


def create_constants_json_file(output_path: str, constants_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Create a constants JSON file from the current constants or provided dictionary.
    
    Args:
        output_path: Path to write the JSON file
        constants_dict: Optional dictionary of constants to write. If None, uses current constants.
    """
    if constants_dict is None:
        constants_dict = get_constants()
        
    # Validate the constants
    if not validate_constants_json(constants_dict):
        raise ValueError("Invalid constants dictionary")
        
    # Convert tuples to lists for JSON serialization
    json_dict = constants_dict.copy()
    for key in ['RUN_KEYS', 'VALIDATION_RUN_KEYS', 'TEST_RUN_KEYS']:
        if key in json_dict:
            json_dict[key] = [list(item) if isinstance(item, tuple) else item for item in json_dict[key]]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


def print_status() -> None:
    """
    Print the current status of the remote constants system.
    """
    cache_info = get_cache_info()
    constants = get_constants()
    
    print("PNPL LibriBrain2025 Remote Constants Status")
    print("=" * 50)
    
    # Check if URL is explicitly set to empty string (offline mode)
    env_url = os.environ.get('PNPL_REMOTE_CONSTANTS_URL')
    if env_url == '':
        print("Remote URL: Offline mode (explicitly disabled)")
    elif env_url is None:
        print(f"Remote URL: {cache_info['remote_url']} (default)")
    else:
        print(f"Remote URL: {cache_info['remote_url']}")
    
    print(f"Remote fetching disabled: {cache_info['disabled']}")
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Cache timeout: {cache_info['cache_timeout']} seconds")
    print(f"Cache file exists: {cache_info['cache_file_exists']}")
    print(f"Cache meta file exists: {cache_info['cache_meta_file_exists']}")
    
    if cache_info.get('cache_timestamp'):
        import time
        cache_age = time.time() - cache_info['cache_timestamp']
        print(f"Cache age: {cache_age:.0f} seconds")
        print(f"Cache source URL: {cache_info.get('cache_source_url', 'Unknown')}")
    
    print(f"\nCurrent constants summary:")
    print(f"  PHONEMES: {len(constants['PHONEMES'])} phonemes")
    print(f"  RUN_KEYS: {len(constants['RUN_KEYS'])} run keys")
    print(f"  VALIDATION_RUN_KEYS: {len(constants['VALIDATION_RUN_KEYS'])} validation keys")
    print(f"  TEST_RUN_KEYS: {len(constants['TEST_RUN_KEYS'])} test keys")
    print(f"  PHONEME_CLASSES: {constants['PHONEME_CLASSES']}")
    print(f"  SPEECH_OUTPUT_DIM: {constants['SPEECH_OUTPUT_DIM']}")
    print(f"  PHONEME_HOLDOUT_PREDICTIONS: {constants['PHONEME_HOLDOUT_PREDICTIONS']}")
    print(f"  SPEECH_HOLDOUT_PREDICTIONS: {constants['SPEECH_HOLDOUT_PREDICTIONS']}")


# Environment variable configuration examples (for documentation)
EXAMPLE_ENV_VARS = """
# Example environment variables for remote constants configuration:

# Default behavior (uses official LibriBrain2025 competition URL)
# No need to set PNPL_REMOTE_CONSTANTS_URL - it defaults to:
# https://neural-processing-lab.github.io/2025-libribrain-competition/constants.json

# Override with custom remote constants URL
export PNPL_REMOTE_CONSTANTS_URL="https://example.com/libribrain2025_constants.json"

# Use local file
export PNPL_REMOTE_CONSTANTS_URL="file:///path/to/constants.json"

# Disable remote constants fetching (offline mode)
export PNPL_REMOTE_CONSTANTS_URL=""

# Alternative: disable via flag
export PNPL_REMOTE_CONSTANTS_DISABLED="true"

# Set custom cache directory
export PNPL_CACHE_DIR="/path/to/custom/cache"

# Set cache timeout (in seconds, default 86400 = 24 hours)
export PNPL_CACHE_TIMEOUT="3600"
""" 