"""``run_all.sh`` must hand its scripts absolute paths.

Each deliverable runs with its own directory as the working directory, so
``--fits evaluations/fourth.fits`` given at the repo root resolved against
``results/`` and every script raised FileNotFoundError on a path that plainly
existed. ``tab:unit-test-bias`` failed the same way but reported all four rungs
as ``\\pending``, which is the worse shape of the bug: it reads as an unfinished
campaign rather than a broken driver.

Run with: python -m pytest results/test_run_all_paths.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "run_all.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="run_all.sh absent")


def _dry_run(*args, cwd=None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(cwd or REPO),
    )


def test_relative_fits_becomes_absolute(tmp_path):
    runs = tmp_path / "evaluations"
    runs.mkdir()
    (runs / "fourth.fits").write_bytes(b"")

    result = _dry_run("--fits", "evaluations/fourth.fits", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--fits evaluations/fourth.fits" not in result.stdout, (
        "a relative --fits reaches the scripts, which run from results/ and "
        "will not find it"
    )
    assert str(runs / "fourth.fits") in result.stdout


def test_relative_runs_becomes_absolute(tmp_path):
    runs = tmp_path / "evaluations"
    runs.mkdir()
    for name in ("first", "second", "third", "fourth"):
        (runs / f"{name}.fits").write_bytes(b"")

    result = _dry_run("--runs", "evaluations", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"--runs {runs}" in result.stdout


def test_absolute_paths_are_left_alone(tmp_path):
    fits = tmp_path / "one.fits"
    fits.write_bytes(b"")
    result = _dry_run("--fits", str(fits), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert str(fits) in result.stdout
    assert f"{tmp_path}{tmp_path}" not in result.stdout  # not doubled


def test_a_missing_fits_fails_before_anything_runs(tmp_path):
    """One clear message, not seven identical tracebacks."""
    runs = tmp_path / "evaluations"
    runs.mkdir()
    for name in ("first", "fourth"):
        (runs / f"{name}.fits").write_bytes(b"")

    result = _dry_run("--go", "--fits", "evaluations/typo.fits", cwd=tmp_path)
    assert result.returncode == 1
    assert "--fits does not exist" in result.stderr
    assert "first.fits" in result.stderr and "fourth.fits" in result.stderr
    assert "=== tab:response-diag ===" not in result.stdout


def test_a_missing_runs_directory_is_refused(tmp_path):
    result = _dry_run("--go", "--runs", "nope", cwd=tmp_path)
    assert result.returncode == 1
    assert "--runs is not a directory" in result.stderr


def test_the_cut_reaches_every_deliverable_that_takes_one(tmp_path):
    """fig:snr_size and tab:unit-test-bias must not be left on a different sample."""
    runs = tmp_path / "evaluations"
    runs.mkdir()
    (runs / "fourth.fits").write_bytes(b"")

    result = _dry_run("--fits", "evaluations/fourth.fits", "--runs", "evaluations",
                      "--cut", "both", "--min-resolution", "1.0", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        for label in ("fig:snr_size", "tab:unit-test-bias",
                      "tab:response-diag", "fig:prediction-residuals",
                      "fig:response_snr"):
            if label in line:
                assert "--cut both" in line, f"{label} runs without the cut: {line}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash absent")
def test_the_driver_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
