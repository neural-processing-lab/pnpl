import importlib.util as iu


def test_namespace_paths_exist():
    for modname in ['pnpl', 'pnpl.datasets']:
        spec = iu.find_spec(modname)
        assert spec is not None
        if spec.submodule_search_locations is not None:
            # namespace or package with search locations
            assert len(list(spec.submodule_search_locations)) >= 1

    spec = iu.find_spec('pnpl.datasets.libribrain2025')
    assert spec is not None and spec.submodule_search_locations

