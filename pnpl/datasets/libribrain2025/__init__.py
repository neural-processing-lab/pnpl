"""Namespace-enabled subpackage for LibriBrain2025 datasets.

Extends the module search path so private overlays can contribute files
like `word_dataset.py` without modifying the public package.
"""

try:  # pkgutil-style namespace for nested subpackage
    import pkgutil as _pkgutil  # stdlib
    __path__ = _pkgutil.extend_path(__path__, __name__)
except Exception:
    pass
