import importlib.util as iu
import sys


def test_public_imports_are_lightweight():
    # Snapshot loaded modules before importing pnpl
    before = set(sys.modules)

    import pnpl  # noqa: F401
    import pnpl.datasets  # noqa: F401

    # Ensure heavy deps were not dragged in by just importing package roots
    heavy = {"torch", "mne_bids", "h5py"}
    newly_loaded = set(sys.modules) - before
    assert heavy.isdisjoint(newly_loaded)


def test_find_public_subpackages_without_importing_them():
    # We can discover public subpackages without importing heavy modules
    spec = iu.find_spec('pnpl.datasets.libribrain2025')
    assert spec is not None and spec.submodule_search_locations

    # The convenience names are listed via __dir__ without import
    import pnpl.datasets as ds
    listed = dir(ds)
    for name in ['LibriBrainPhoneme', 'LibriBrainSpeech', 'GroupedDataset', 'LibriBrainCompetitionHoldout']:
        assert name in listed
