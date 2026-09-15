"""Two things about the figures that broke once and would break again.

1. ``fig:prediction-residuals`` shipped as a 6 MB PDF that no viewer would open.
   ``plot_comparison`` draws one vector path per point, which is right at the
   sizes it was written for and fatal at 2x10^5 per panel: 18 MB of uncompressed
   drawing commands, and ~30 s to render a single page even in a fast C
   renderer. Nothing about the figure was wrong -- the slopes in it are the ones
   the paper quotes -- so there was no failure to notice, only a file that would
   not open.

2. ``fig:response_snr`` takes its binning and its axes from
   ``plots_from_fits.ipynb`` cells 22 and 26 (see the house rule in README).
   Those are copies, so they can drift from the notebook without anything
   complaining. These check the properties the notebook's version has.

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
from matplotlib.collections import PathCollection  # noqa: E402

from prediction_residuals import _rasterize_scatter  # noqa: E402
from response_vs_snr import _as_sorted, _bin_response, style_log_x  # noqa: E402


# --------------------------------------------------------------------------
# 1. the point cloud, and only the point cloud, is rasterized
# --------------------------------------------------------------------------

def test_only_the_scatter_is_rasterized():
    """Text and the tolerance band must stay vector, or the page stops scaling."""
    fig, ax = plt.subplots()
    scatter = ax.scatter(np.arange(10), np.arange(10))
    band = ax.fill_between([0, 9], [-1, -1], [1, 1])
    line, = ax.plot([0, 9], [0, 0])

    assert _rasterize_scatter([ax]) == 1
    assert scatter.get_rasterized() is True
    assert not band.get_rasterized()
    assert not line.get_rasterized()
    plt.close(fig)


def test_it_walks_a_2d_axes_grid():
    """plot_comparison returns a list, but subplots can hand back an array."""
    fig, axes = plt.subplots(2, 2)
    for ax in axes.ravel():
        ax.scatter([0, 1], [0, 1])
    assert _rasterize_scatter(axes) == 4
    plt.close(fig)


def test_the_page_stops_growing_with_the_sample(tmp_path):
    """The invariant, not a size threshold.

    A vector scatter costs one path operator per point, so the page grows
    without bound as the catalogue does -- which is how a 2x10^5-object run
    produced 18 MB of drawing commands. A rasterized one costs the same however
    many points land in it. Quadrupling N should barely move the raster and
    should visibly move the vector; the ratio between them at any single N is
    just a function of figure size and dpi, so asserting on it would be
    asserting on nothing.
    """
    rng = np.random.default_rng(0)

    def page(n, rasterize):
        fig, ax = plt.subplots()
        ax.scatter(rng.normal(size=n), rng.normal(size=n), s=0.7, alpha=0.5)
        if rasterize:
            _rasterize_scatter([ax])
        out = tmp_path / f"{n}-{rasterize}.pdf"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out.stat().st_size

    small, large = 20_000, 80_000
    vector_growth = page(large, False) / page(small, False)
    raster_growth = page(large, True) / page(small, True)

    assert vector_growth > 2.0, (
        f"the vector page barely grew ({vector_growth:.2f}x for 4x the points) "
        "-- this test is no longer measuring what it thinks it is"
    )
    assert raster_growth < 1.3, (
        f"the rasterized page grew {raster_growth:.2f}x for 4x the points, so "
        "the scatter is reaching the PDF as vector paths -- if it is no longer "
        "a PathCollection, _rasterize_scatter silently does nothing"
    )


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


# --------------------------------------------------------------------------
# 3. the figure and the table quote alpha on the same shape
# --------------------------------------------------------------------------

def test_the_figure_and_the_tables_resolve_the_same_leakage_shape():
    """fig:psf_leakage and tab:unit-test-bias must not drift apart.

    ``run_all.sh`` passes no ``--shape``, so before this both estimators were
    fitted on ``raw`` while the tables quoted each on its own shape -- a figure
    and a caption describing different measurements, with nothing failing.
    """
    from paper_numbers import reported_leakage_shape
    from psf_leakage import _reported_shape

    for estimator in ("shearnet", "ngmix"):
        assert _reported_shape(estimator) == reported_leakage_shape(estimator)


def test_ngmix_leakage_excludes_rpsf():
    """The one deliberate divergence, pinned so it cannot revert quietly.

    ngmix's m and c come from the full metacal pipeline, but its alpha stops one
    step short: R^PSF is ~25x larger than the leakage it would correct and
    subtracting it flips the sign (alpha -0.43 rather than +0.017). The paper's
    frozen numbers and its limitations section both assume R^PSF is applied to
    neither estimator's leakage.
    """
    from paper_numbers import REPORTED_CORRECTION, reported_leakage_shape

    assert REPORTED_CORRECTION["ngmix"] == "metacal", "m and c stay on metacal"
    assert reported_leakage_shape("ngmix") == "noshear_rgamma"
    assert "rpsf" not in reported_leakage_shape("ngmix")
    assert "rpsf" not in reported_leakage_shape("shearnet")


def test_the_x_axis_is_log_with_one_two_five_ticks():
    fig, ax = plt.subplots()
    style_log_x(ax, 5.0, 2000.0)
    assert ax.get_xscale() == "log"

    ticks = ax.xaxis.get_major_locator()()
    assert len(ticks) <= 8, "the notebook caps the labelled ticks at eight"
    mantissas = {round(t / 10 ** np.floor(np.log10(t)), 6) for t in ticks}
    assert mantissas <= {1.0, 2.0, 5.0}, f"unexpected tick mantissas: {mantissas}"
    plt.close(fig)
