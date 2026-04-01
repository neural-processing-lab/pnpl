import importlib
import importlib.util as iu
import sys
import types


def _disable_overlay_editable(monkeypatch):
    def _is_pnpl_editable_module(name: str) -> bool:
        return name.startswith("__editable___") and "pnpl" in name

    def _is_pnpl_editable_path(entry: str) -> bool:
        return entry.startswith("__editable__.") and "pnpl" in entry

    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if not _is_pnpl_editable_module(getattr(finder, "__module__", ""))
        ],
    )
    monkeypatch.setattr(
        sys,
        "path_hooks",
        [
            hook
            for hook in sys.path_hooks
            if not _is_pnpl_editable_module(getattr(hook, "__module__", ""))
        ],
    )
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if not _is_pnpl_editable_path(entry)],
    )
    for name in list(sys.modules):
        if _is_pnpl_editable_module(name):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_namespace_paths_exist():
    for modname in ['pnpl', 'pnpl.datasets', 'pnpl.tasks', 'pnpl.preprocessing']:
        spec = iu.find_spec(modname)
        assert spec is not None
        if spec.submodule_search_locations is not None:
            # namespace or package with search locations
            assert len(list(spec.submodule_search_locations)) >= 1

    spec = iu.find_spec('pnpl.datasets.libribrain2025')
    assert spec is not None and spec.submodule_search_locations


def test_overlay_modules_are_discovered_without_registration(tmp_path, monkeypatch):
    overlay_root = tmp_path / "pnpl" / "tasks"
    overlay_root.mkdir(parents=True)
    (overlay_root / "private_overlay.py").write_text("VALUE = 'overlay'\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))

    for modname in ["pnpl.tasks.private_overlay", "pnpl.tasks", "pnpl"]:
        sys.modules.pop(modname, None)

    module = importlib.import_module("pnpl.tasks.private_overlay")
    assert module.VALUE == "overlay"


def test_overlay_modules_can_override_public_modules(tmp_path, monkeypatch):
    _disable_overlay_editable(monkeypatch)
    overlay_root = tmp_path / "pnpl" / "datasets" / "libribrain2025"
    overlay_root.mkdir(parents=True)
    (overlay_root / "remote_constants.py").write_text(
        "OVERLAY_SENTINEL = 'internal'\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    for modname in [
        "pnpl.datasets.libribrain2025.remote_constants",
        "pnpl.datasets.libribrain2025",
        "pnpl.datasets",
        "pnpl",
    ]:
        sys.modules.pop(modname, None)

    libribrain = importlib.import_module("pnpl.datasets.libribrain2025")
    assert str(overlay_root) == list(libribrain.__path__)[0]

    module = importlib.import_module("pnpl.datasets.libribrain2025.remote_constants")
    assert module.OVERLAY_SENTINEL == "internal"


def test_editable_overlay_paths_are_included(monkeypatch, tmp_path):
    _disable_overlay_editable(monkeypatch)
    public_path = tmp_path / "public" / "pnpl" / "datasets"
    public_path.mkdir(parents=True)
    overlay_path = tmp_path / "overlay" / "pnpl" / "datasets"
    overlay_path.mkdir(parents=True)

    finder = types.ModuleType("__editable___pnpl_overlay_test_finder")
    finder.MAPPING = {"pnpl.datasets": str(overlay_path)}
    finder.NAMESPACES = {"pnpl": []}
    monkeypatch.setitem(sys.modules, finder.__name__, finder)

    from pnpl._namespace import extend_overlay_path

    merged = extend_overlay_path([str(public_path)], "pnpl.datasets")
    assert str(overlay_path) in merged
    assert merged.index(str(overlay_path)) < merged.index(str(public_path))
    assert str(public_path) in merged


def test_libribrain_module_aliases_import():
    for modname in [
        'pnpl.datasets.libribrain2025.phoneme_dataset',
        'pnpl.datasets.libribrain2025.speech_dataset',
        'pnpl.datasets.libribrain2025.sentence_dataset',
        'pnpl.datasets.libribrain2025.word_dataset',
    ]:
        spec = iu.find_spec(modname)
        assert spec is not None
