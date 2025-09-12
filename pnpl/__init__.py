"""pnpl public base package.

This package is installable on its own from PyPI. It also supports an
optional private overlay distribution that contributes additional
subpackages under the same top-level namespace.
"""

# 1) Make `pnpl` a namespace so multiple distributions can contribute.
try:  # pkgutil-style namespace (works when __init__.py is present)
    import pkgutil as _pkgutil  # stdlib
    __path__ = _pkgutil.extend_path(__path__, __name__)
except Exception:
    # If anything goes wrong, keep default package path.
    pass

# 2) Optional lazy access to overlay exports at the top-level. We avoid
# importing the overlay eagerly to keep `import pnpl` lightweight.
def __getattr__(name):  # pragma: no cover - import-time hook
    try:
        from importlib import import_module
        mod = import_module("pnpl._private_exports")
    except Exception as _e:
        raise AttributeError(name) from _e
    try:
        return getattr(mod, name)
    except AttributeError as _e:
        raise

def __dir__():  # pragma: no cover - import-time hook
    standard = []
    try:
        from importlib import import_module
        mod = import_module("pnpl._private_exports")
        return sorted(set(list(standard) + [
            n for n in dir(mod) if not n.startswith("_")
        ]))
    except Exception:
        return sorted(standard)
