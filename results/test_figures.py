"""Things about the figures that broke once and would break again.

``fig:response_snr`` takes its binning and its axes from
``plots_from_fits.ipynb`` cells 22 and 26 (see the house rule in README).
Those are copies, so they can drift from the notebook without anything
complaining. These check the properties the notebook's version has, and that
the figure is drawn on the whole population, never the cut one.

Run with: python -m pytest results/test_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from response_vs_snr import _as_sorted, _bin_response, style_log_x  # noqa: E402


# --------------------------------------------------------------------------
# 2. the binning still behaves the way the notebook's does
# --------------------------------------------------------------------------

def test_bins_hold_equal_counts_not_equal_widths():
    """The notebook bins on percentiles, so a lognormal S/N tail cannot own a bin."""
    rng = np.random.default_rng(1)
    s2n = rng.lognormal(3.0, 0.8, 20_000)
    response = np.full_like(s2n, 0.9)

    centres, medians, errors = _bin_response(response, s2n, n_bins=10)
    assert len(centres) == len(medians) == len(errors) == 10
    assert np.allclose(medians, 0.9)
    assert np.all(np.diff(centres) > 0)


def test_nonpositive_and_nonfinite_are_dropped_before_binning():
    """S/N <= 0 would break the log axis; NaN responses would poison a median."""
    s2n = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    response = np.array([1.0, 1.0, np.nan, 0.5, 0.5, 0.5])
    centres, medians, _ = _bin_response(response, s2n, n_bins=2)
    assert np.all(np.isfinite(centres)) and np.all(np.isfinite(medians))
    assert centres.min() > 0


def test_as_sorted_orders_by_x_and_drops_nan():
    x, y, e = _as_sorted([3.0, 1.0, np.nan, 2.0], [0.3, 0.1, 0.9, 0.2],
                         [0.03, 0.01, 0.09, 0.02])
    assert list(x) == [1.0, 2.0, 3.0]
    assert list(y) == [0.1, 0.2, 0.3]
    assert list(e) == [0.01, 0.02, 0.03]




def test_figure_3_has_no_cut_option_and_is_measured_uncut(tmp_path, monkeypatch):
    """The figure motivates the cut, so it must show what the cut removes."""
    import response_vs_snr
    from make_fixture import build

    path = tmp_path / "fourth.fits"
    build(3000, seed=7).writeto(path)
    with pytest.raises(SystemExit):
        response_vs_snr.main(["--fits", str(path), "--cut", "metacal"])

    seen = {}

    def fake_draw(per_object, estimators, **kwargs):
        seen.update(per_object)
        return kwargs["out_path"]

    monkeypatch.setattr(response_vs_snr, "draw", fake_draw)
    monkeypatch.setitem(sys.modules, "superbit_lensing", type(sys)("superbit_lensing"))
    monkeypatch.setitem(sys.modules, "superbit_lensing.plotter",
                        type(sys)("superbit_lensing.plotter"))
    response_vs_snr.main(["--fits", str(path), "--out", str(tmp_path / "r.pdf")])
    # every object, including the ones s2n > 10 would drop
    assert seen["ngmix"]["n_objects"] == 3000
    assert (seen["ngmix"]["s2n"] < 10).any()


# --------------------------------------------------------------------------
# 4. a PSF source may be a directory, because the configs use one
# --------------------------------------------------------------------------

def _psf_properties():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "psf"))
    import psf_properties

    return psf_properties


def test_a_directory_of_psf_models_resolves_to_one_of_them(tmp_path):
    """paths.psfex_model_file is a file OR a directory, and UT1-UT4 use a directory.

    ShearNet accepts both and draws a different one of the 50 SuperBIT models
    per object; this script wants a single model, and used to raise
    FileNotFoundError on a directory that plainly existed.
    """
    for name in ("b.psf", "a.psf", "c.psf"):
        (tmp_path / name).write_bytes(b"")

    resolve = _psf_properties().resolve_psf_model
    assert resolve(tmp_path).name == "a.psf", "sorted, not filesystem order"
    assert resolve(tmp_path, index=2).name == "c.psf"
    assert resolve(tmp_path / "b.psf").name == "b.psf", "a file still passes through"


def test_a_directory_with_no_models_says_so(tmp_path):
    resolve = _psf_properties().resolve_psf_model
    with pytest.raises(FileNotFoundError, match="no .psf files"):
        resolve(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve(tmp_path / "absent")


def test_the_x_axis_is_log_with_one_two_five_ticks():
    fig, ax = plt.subplots()
    style_log_x(ax, 5.0, 2000.0)
    assert ax.get_xscale() == "log"

    ticks = ax.xaxis.get_major_locator()()
    assert len(ticks) <= 8, "the notebook caps the labelled ticks at eight"
    mantissas = {round(t / 10 ** np.floor(np.log10(t)), 6) for t in ticks}
    assert mantissas <= {1.0, 2.0, 5.0}, f"unexpected tick mantissas: {mantissas}"
    plt.close(fig)
