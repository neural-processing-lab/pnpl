import types
import sys


def test_datasets_overlay_hook_via_private_exports(monkeypatch):
    import pnpl.datasets as ds

    mod = types.ModuleType('pnpl.datasets._private_exports')
    mod.Foo = object()
    monkeypatch.setitem(sys.modules, 'pnpl.datasets._private_exports', mod)

    # dir should now list Foo and getattr should return it lazily
    assert 'Foo' in dir(ds)
    assert getattr(ds, 'Foo') is mod.Foo


def test_top_level_overlay_hook_via_private_exports(monkeypatch):
    import pnpl

    mod = types.ModuleType('pnpl._private_exports')
    mod.Bar = object()
    monkeypatch.setitem(sys.modules, 'pnpl._private_exports', mod)

    # dir should now list Bar and getattr should return it lazily
    assert 'Bar' in dir(pnpl)
    assert getattr(pnpl, 'Bar') is mod.Bar

