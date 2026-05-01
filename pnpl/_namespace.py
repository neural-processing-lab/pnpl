"""Helpers for package composition under the shared ``pnpl`` namespace."""

from __future__ import annotations

import os
import pkgutil
import sys
from collections.abc import Iterable
from pathlib import Path


def _normalize(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _editable_overlay_paths(package_name: str) -> list[str]:
    package_parts = package_name.split(".")
    package_depth = len(package_parts)
    discovered: list[str] = []
    seen: set[str] = set()

    for module in list(sys.modules.values()):
        mapping = getattr(module, "MAPPING", None)
        namespaces = getattr(module, "NAMESPACES", None)
        if not isinstance(mapping, dict) or not isinstance(namespaces, dict):
            continue

        candidates: list[tuple[str, list[str]]] = []
        candidates.extend(
            (name, [path])
            for name, path in mapping.items()
            if isinstance(name, str) and isinstance(path, str)
        )
        candidates.extend(
            (name, [path for path in paths if isinstance(path, str)])
            for name, paths in namespaces.items()
            if isinstance(name, str) and isinstance(paths, list)
        )

        for name, paths in candidates:
            if name == package_name:
                derived = paths
            elif name.startswith(f"{package_name}."):
                remaining = len(name.split(".")) - package_depth
                derived = [
                    str(Path(path).parents[remaining - 1]) if remaining else path
                    for path in paths
                    if remaining <= len(Path(path).parts)
                ]
            else:
                continue

            for path in derived:
                if not os.path.isdir(path):
                    continue
                normalized = _normalize(path)
                if normalized in seen:
                    continue
                seen.add(normalized)
                discovered.append(path)

    return discovered


def extend_overlay_path(path: Iterable[str], package_name: str) -> list[str]:
    """Extend a package path and prefer later-installed portions.

    ``pkgutil.extend_path`` merges package portions across distributions, but it
    keeps the currently imported package directory first. For the package
    composition model used by PNPL, later-installed namespace portions should
    win for same-named submodules while still falling back to the local
    implementation when no override exists.
    """

    local_paths = list(path)
    merged_paths = list(pkgutil.extend_path(local_paths, package_name))
    merged_paths.extend(_editable_overlay_paths(package_name))
    if len(merged_paths) <= len(local_paths):
        return merged_paths

    local_norms = {_normalize(entry) for entry in local_paths}
    overlay_paths: list[str] = []
    public_paths: list[str] = []
    seen: set[str] = set()

    for entry in merged_paths:
        normalized = _normalize(entry)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized in local_norms:
            public_paths.append(entry)
        else:
            overlay_paths.append(entry)

    return overlay_paths + public_paths
