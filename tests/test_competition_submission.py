"""Offline tests for submit_to_kaggle output/credential handling (no network)."""

import subprocess

import pytest

from pnpl.competition import (
    PNPL_2026_COMPETITIONS,
    KaggleSubmissionResult,
    SubmissionError,
    resolve_competition,
    submit_to_kaggle,
)
from pnpl.competition import submission as sub


# The real kaggle success output: an "outdated version" warning, then the
# "Successfully submitted ..." line, with an upload progress bar on stderr.
SUCCESS_STDOUT = (
    "Warning: Looks like you're using an outdated `kaggle` version "
    "(installed: 2.0.2), please consider upgrading to the latest version "
    "(2.2.2)\nSuccessfully submitted to PNPL Competition 2026"
)
SUCCESS_STDERR = "\n  0%|          | 0.00/427k [00:00<?, ?B/s]\n100%|##########| 427k/427k\n"


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "kaggle_cfg"
    empty.mkdir()
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(empty))
    return empty


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "submission.csv"
    p.write_text("index,a\n0,1.0\n")
    return p


def _fake_run(returncode, stdout="", stderr=""):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0] if args else [], returncode, stdout, stderr)
    return run


# -- credential resolution -------------------------------------------------

def test_have_credentials(monkeypatch, no_credentials):
    assert sub._have_kaggle_credentials(None) is False
    assert sub._have_kaggle_credentials("KGAT_xyz") is True
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_abc")
    assert sub._have_kaggle_credentials(None) is True


def test_have_credentials_username_key(monkeypatch, no_credentials):
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    assert sub._have_kaggle_credentials(None) is False   # key missing
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert sub._have_kaggle_credentials(None) is True


def test_have_credentials_json_file(no_credentials):
    assert sub._have_kaggle_credentials(None) is False
    (no_credentials / "kaggle.json").write_text("{}")
    assert sub._have_kaggle_credentials(None) is True


def test_missing_credentials_fails_fast(no_credentials, csv_file, monkeypatch):
    # Must not even try to shell out.
    monkeypatch.setattr(sub.subprocess, "run", _fake_run(0, SUCCESS_STDOUT))
    with pytest.raises(SubmissionError) as e:
        submit_to_kaggle(csv_file, "pnpl-competition-2026")
    msg = str(e.value)
    assert "No Kaggle credentials found" in msg
    assert "KAGGLE_API_TOKEN" in msg and "kaggle.json" in msg


# -- output parsing --------------------------------------------------------

def test_summarize_drops_version_warning():
    detail = sub._summarize_output(SUCCESS_STDOUT + "\n" + SUCCESS_STDERR, success=True)
    assert detail == "Successfully submitted to PNPL Competition 2026"
    assert "outdated" not in detail.lower()


def test_diagnose_auth_failure():
    out = "401 - Unauthorized: invalid API token"
    assert "could not authenticate" in sub._diagnose_failure(out, 1)


def test_diagnose_auth_failure_missing_json():
    out = "OSError: Could not find kaggle.json. Make sure it's located in ~/.kaggle"
    assert "could not authenticate" in sub._diagnose_failure(out, 1)


def test_diagnose_generic_failure():
    out = "Warning: Looks like you're using an outdated version\nBadRequest: file too large"
    assert sub._diagnose_failure(out, 1) == "BadRequest: file too large"


# -- result object ---------------------------------------------------------

def test_result_repr_and_bool():
    ok = KaggleSubmissionResult(success=True, competition="c", file="/tmp/submission_deep.csv",
                                message="m", detail="Successfully submitted to X", process=None)
    assert bool(ok) is True
    r = repr(ok)
    assert "submission_deep.csv" in r and "→ c" in r and "Successfully submitted to X" in r

    bad = KaggleSubmissionResult(success=False, competition="c", file="/tmp/s.csv",
                                 message="m", detail="nope", process=None)
    assert bool(bad) is False and "failed" in repr(bad).lower()


# -- end to end (subprocess mocked) ---------------------------------------

def test_submit_success(monkeypatch, csv_file):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_fake")
    monkeypatch.setattr(sub.subprocess, "run", _fake_run(0, SUCCESS_STDOUT, SUCCESS_STDERR))
    res = submit_to_kaggle(csv_file, "pnpl-competition-2026", "hi",
                           kaggle_bin="kaggle", verbose=False)
    assert res and res.success and res.returncode == 0
    assert res.detail == "Successfully submitted to PNPL Competition 2026"


def test_submit_failure_raises(monkeypatch, csv_file):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_fake")
    monkeypatch.setattr(sub.subprocess, "run", _fake_run(1, "", "401 Unauthorized"))
    with pytest.raises(SubmissionError, match="could not authenticate"):
        submit_to_kaggle(csv_file, "c", kaggle_bin="kaggle")


def test_submit_failure_no_check_returns_result(monkeypatch, csv_file):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_fake")
    monkeypatch.setattr(sub.subprocess, "run", _fake_run(1, "", "BadRequest: too big"))
    res = submit_to_kaggle(csv_file, "c", kaggle_bin="kaggle", check=False, verbose=False)
    assert res.success is False
    assert "too big" in res.detail


def test_submit_exit0_without_marker_is_failure(monkeypatch, csv_file):
    # kaggle sometimes exits 0 but prints an error instead of the success line.
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_fake")
    monkeypatch.setattr(sub.subprocess, "run", _fake_run(0, "Some error, not submitted", ""))
    res = submit_to_kaggle(csv_file, "c", kaggle_bin="kaggle", check=False, verbose=False)
    assert res.success is False


# -- competition shorthand resolution --------------------------------------

def test_resolve_competition_shorthand():
    assert resolve_competition("deep") == "pnpl-competition-2026-deep"
    assert resolve_competition("broad") == "pnpl-competition-2026-broad"
    assert resolve_competition("DEEP") == PNPL_2026_COMPETITIONS["deep"]
    assert resolve_competition("  Broad ") == PNPL_2026_COMPETITIONS["broad"]


def test_resolve_competition_url():
    base = "https://www.kaggle.com/competitions/"
    assert resolve_competition(base + "pnpl-competition-2026-deep/") == "pnpl-competition-2026-deep"
    assert resolve_competition(base + "pnpl-competition-2026-broad") == "pnpl-competition-2026-broad"
    assert resolve_competition(base + "some-comp/overview") == "some-comp"


def test_resolve_competition_passthrough():
    # A full slug (an explicit override) is returned unchanged.
    assert resolve_competition("pnpl-internal-testing") == "pnpl-internal-testing"
    assert resolve_competition("pnpl-competition-2026") == "pnpl-competition-2026"


def test_submit_resolves_shorthand_in_cli(monkeypatch, csv_file):
    # submit_to_kaggle maps "deep" -> the slug and passes THAT to `kaggle ... -c <slug>`.
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_fake")
    seen = {}

    def record(*args, **kwargs):
        seen["cmd"] = list(args[0]) if args else list(kwargs["args"])
        return subprocess.CompletedProcess(seen["cmd"], 0, SUCCESS_STDOUT, SUCCESS_STDERR)

    monkeypatch.setattr(sub.subprocess, "run", record)
    res = submit_to_kaggle(csv_file, competition="deep", kaggle_bin="kaggle", check=False)
    assert res.success and res.competition == "pnpl-competition-2026-deep"
    assert seen["cmd"][seen["cmd"].index("-c") + 1] == "pnpl-competition-2026-deep"
