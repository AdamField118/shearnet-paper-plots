"""PSF ellipticities of the objects used to measure leakage."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

LABELS = (r"$e^{\rm PSF}_1$", r"$e^{\rm PSF}_2$")


def _edges(values, bins):
    lo, hi = np.min(values), np.max(values)
    if lo == hi:
        pad = max(abs(lo) * .05, .001)
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, bins + 1)


def add_histograms(axes, pairs, bins=40):
    """Show the full PSF sample; histogram height is in axes, not shear, units."""
    for ax, values in zip(axes, pairs.T):
        counts, edges = np.histogram(values, bins=_edges(values, bins))
        ylim = ax.get_ylim()
        ax.stairs(.60 * counts / counts.max(), edges, fill=True,
                  transform=ax.get_xaxis_transform(), color="0.88", lw=0, zorder=-1)
        # Include the tails even if upstream fitted only the percentile-bin centres.
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(min(xmin, edges[0]), max(xmax, edges[-1]))
        ax.set_ylim(ylim)


def corner_comparison(fiducial, abell, bins=48):
    """Joint and marginal bin probabilities on identical scales in both fields."""
    samples = (fiducial, abell)
    for pairs in samples:
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or not np.isfinite(pairs).all():
            raise ValueError("Expected at least one finite pair of PSF ellipticity components")
    edges = _edges(np.concatenate(samples).ravel(), bins)
    mass = [np.histogram2d(*p.T, bins=(edges, edges))[0] / len(p) for p in samples]
    positive = np.concatenate([h[h > 0] for h in mass])
    norm = LogNorm(positive.min(), max(positive.max(), positive.min() * 1.01))
    marginal_max = max(h.sum(axis=i).max() for h in mass for i in (0, 1))
    fig, joints = plt.subplots(1, 2, figsize=(8.1, 4.3))
    fig.subplots_adjust(left=.09, right=.89, bottom=.29, top=.77, wspace=.60)
    for ax, pairs, h, label in zip(joints, samples, mass, ("Fiducial", "Abell 3411")):
        top = ax.inset_axes([0, 1.015, 1, .23])
        side = ax.inset_axes([1.015, 0, .23, 1])
        mesh = ax.pcolormesh(edges, edges, np.ma.masked_equal(h.T, 0), cmap="cividis",
                             norm=norm, rasterized=True)
        # Match scales in both coordinates and align the marginals to the joint panel.
        ax.set(xlabel=LABELS[0], ylabel=LABELS[1], xlim=(edges[0], edges[-1]),
               ylim=(edges[0], edges[-1]), aspect="equal")
        top.stairs(h.sum(axis=1), edges, fill=True, color=".6")
        side.stairs(h.sum(axis=0), edges, orientation="horizontal", fill=True, color=".6")
        top.set(xlim=ax.get_xlim(), ylim=(0, marginal_max * 1.05))
        side.set(ylim=ax.get_ylim(), xlim=(0, marginal_max * 1.05))
        for marginal in (top, side):
            marginal.set_axis_off()
        rho = np.corrcoef(pairs.T)[0, 1] if np.all(np.ptp(pairs, axis=0) > 0) else None
        stat = f"$r={rho:+.2f}$" if rho is not None else "$r$ undefined"
        top.set_title(label, loc="left", fontsize=12)
        print(f"{label}: N={len(pairs)}, Pearson r={rho if rho is not None else 'undefined'}")
        top.set_title(stat, loc="right", fontsize=10)
    cax = fig.add_axes([.30, .105, .40, .025])
    fig.colorbar(mesh, cax=cax, orientation="horizontal", label="Fraction of objects per joint bin")
    return fig
