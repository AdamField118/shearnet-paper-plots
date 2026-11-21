"""The plots must describe the same selected galaxies as the leakage fits."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from psf_distribution import add_histograms, corner_comparison


def test_background_counts_include_tails_but_do_not_move_either_axis():
    pairs = np.array([[-.15, .12], [0, .03], [0, .03], [.06, -.2]])
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.set(xlim=(-.05, .05), ylim=(-.001, .003))
    add_histograms(axes, pairs, bins=3)
    for ax, values in zip(axes, pairs.T):
        histogram = ax.patches[0].get_data()
        counts, _ = np.histogram(values, histogram.edges)
        np.testing.assert_allclose(histogram.values, .70 * counts / counts.max())
        assert histogram.edges[0] <= min(values) <= max(values) <= histogram.edges[-1]
        assert ax.get_xlim() == (-.05, .05)
        assert ax.get_ylim() == (-.001, .003)
    plt.close(fig)


def test_paper_comparison_overlays_all_fits_without_autoscaling_to_histogram():
    from leakage_figure import Curve, paper_comparison

    x = np.linspace(-.04, .03, 10)
    def curve(slope):
        return Curve(x, slope * x, np.full_like(x, .0001),
                     np.column_stack((x[[0, -1]], slope * x[[0, -1]])), slope, 5.)

    curves = [curve(.005), curve(.001), curve(.5)]
    hist = (np.array([1, 50, 1]), np.linspace(-.4, .4, 4))
    fig = paper_comparison([curves, curves], [hist, hist], ["shearnet", "ngmix", "ngmix"])
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_yscale() == "linear"
        assert -.1 < ax.get_xlim()[0] < x.min() < x.max() < ax.get_xlim()[1] < .1
        assert ax.get_ylim()[0] < -20.1 and ax.get_ylim()[1] > 15.1
        assert ax.patches[0].get_data().edges[0] == -.4
        assert len(ax.containers) == len(curves)
        for bars, original in zip(ax.containers, curves):
            assert bars.lines[0].axes is ax
            np.testing.assert_allclose(bars.lines[0].get_ydata(), 1000 * original.median)
    plt.close(fig)


def test_corner_preserves_pairing_scale_and_marginal_alignment():
    x = np.linspace(-.13, .09, 44)
    samples = (np.column_stack((x, x)), np.column_stack((x[:22], -x[:22])))
    fig = corner_comparison(*samples, bins=9)
    fig.canvas.draw()
    for ax, pairs, sign in zip(fig.axes, samples, ("+1.00", "-1.00")):
        top, right = ax.child_axes
        assert sign in top.get_title(loc="right")
        marginal = top.patches[0].get_data()
        edges = marginal.edges
        joint = np.histogram2d(*pairs.T, bins=(edges, edges))[0] / len(pairs)
        np.testing.assert_allclose(marginal.values, joint.sum(axis=1))
        np.testing.assert_allclose(right.patches[0].get_data().values, joint.sum(axis=0))
        np.testing.assert_allclose(ax.collections[0].get_array().filled(0).reshape(9, 9), joint.T)
        np.testing.assert_allclose(top.bbox.intervalx, ax.bbox.intervalx)
        np.testing.assert_allclose(right.bbox.intervaly, ax.bbox.intervaly)
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    assert fig.axes[0].collections[0].norm is fig.axes[1].collections[0].norm
    plt.close(fig)


def test_corner_recovery_uses_the_leakage_sample_across_chunk_boundaries(tmp_path):
    from catalog import Catalog
    from make_fixture import build
    from psf_corner import selected_pairs
    from shear_stats import PAPER_CUT, Sample, leakage_inputs

    path = tmp_path / "run.fits"
    with build(509, seed=5) as hdus:
        # One bad station must remove its whole ring, even without the size/SNR cut.
        hdus["SHEARNET"].data["flags_1p"][511] = 1
        hdus.writeto(path)
    for cut in (None, PAPER_CUT):
        with Catalog(path) as catalog:
            sample = Sample(catalog, ["zero"], cut=cut)
            ref = leakage_inputs(sample, "ngmix", variant="noshear")
        expected = np.column_stack((ref["e1_psf"], ref["e2_psf"]))
        np.testing.assert_array_equal(selected_pairs(path, cut, chunk=31), expected)


def test_psf_comparison_uses_a_common_stretch_and_a_signed_linear_residual():
    import sys
    from pathlib import Path
    from matplotlib.colors import LogNorm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "psf"))
    from psf_em5 import comparison

    y, x = np.mgrid[-26:27, -26:27]
    fit = np.exp(-(x*x + y*y) / 18)
    model = fit.copy()
    model[25, 26] += .02
    model[27, 26] -= .03
    fig = comparison(model, fit)
    left, middle, residual = [ax.images[0] for ax in fig.axes[:3]]
    assert isinstance(left.norm, LogNorm) and left.norm is middle.norm
    assert not isinstance(residual.norm, LogNorm)
    np.testing.assert_allclose(residual.get_array(), (model-fit)[10:43, 10:43])
    assert residual.norm(0.) == .5
    plt.close(fig)
