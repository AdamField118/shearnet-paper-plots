"""The plots must describe the same selected galaxies as the leakage fits."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from psf_distribution import add_histograms, corner_comparison


def test_background_counts_include_tails_but_do_not_move_shear_axis():
    pairs = np.array([[-.15, .12], [0, .03], [0, .03], [.06, -.2]])
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.set(xlim=(-.05, .05), ylim=(-.001, .003))
    add_histograms(axes, pairs, bins=3)
    for ax, values in zip(axes, pairs.T):
        histogram = ax.patches[0].get_data()
        counts, _ = np.histogram(values, histogram.edges)
        np.testing.assert_allclose(histogram.values, .60 * counts / counts.max())
        assert ax.get_xlim()[0] <= min(values) <= max(values) <= ax.get_xlim()[1]
        assert ax.get_ylim() == (-.001, .003)
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
