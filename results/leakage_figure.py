"""Figure 5: compare the two adopted estimators and show ngmix before PSF subtraction."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from paper_colors import COLORS
from psf_distribution import LABELS, draw_histogram


@dataclass
class Curve:
    x: np.ndarray
    median: np.ndarray
    error: np.ndarray
    endpoints: np.ndarray
    alpha: float
    significance: float


def read_component(path, i):
    """Read the upstream bin measurements and fit; do not fit the medians again."""
    from astropy.table import Table

    tab = Table.read(path, hdu=f"E{i}PSF")
    x = np.asarray(tab["x_bin"])
    alpha, beta, err = (tab.meta[f"{key}{i}"] for key in ("AFULL", "BFULL", "AERR"))
    limits = np.array([x.min(), x.max()])
    return Curve(x, np.asarray(tab[f"y_bin_{i}"]), np.asarray(tab[f"yerr_bin_{i}"]),
                 np.column_stack((limits, alpha * limits + beta)), alpha, abs(alpha) / err)


def paper_comparison(components, histograms, estimators):
    """Overlay all treatments on the same linear shear axis for each component."""
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.05), sharey=True)
    fig.subplots_adjust(left=.105, right=.988, bottom=.17, top=.77, wspace=.14)
    styles = {"shearnet": ("ShearNet-D4", "o", "-"),
              "ngmix": (r"ngmix ($R^{\rm PSF}$ subtracted)", "^", "--")}
    for col, (ax, curves) in enumerate(zip(axes, components)):
        handles = []
        for n, (curve, est) in enumerate(zip(curves, estimators)):
            label, marker, ls = (styles[est] if n < 2
                                 else (r"ngmix (no $R^{\rm PSF}$)", "s", ":"))
            color = COLORS[est]
            ax.errorbar(curve.x, 1000 * curve.median, yerr=1000 * curve.error,
                        fmt=marker, mfc="white", mec=color, mew=.85, color=color,
                        ms=4, elinewidth=.8, capsize=1.7, zorder=5 - n)
            ax.plot(curve.endpoints[:, 0], 1000 * curve.endpoints[:, 1],
                    color=color, ls=ls, lw=1.25, zorder=2)
            label += rf": $\alpha_{col+1}={curve.alpha:.3f}$ ({curve.significance:.2f}$\sigma$)"
            handles.append(Line2D([], [], color=color, ls=ls, lw=1.25,
                                  marker=marker, mfc="white", ms=4, label=label))
        ax.axhline(0, color=".42", lw=.6, ls=":", zorder=1)
        ax.margins(x=.065, y=.07)
        ax.tick_params(labelsize=8, direction="in", top=True, right=True, length=3, width=.65)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel(LABELS[col], fontsize=9, labelpad=2)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.018),
                  frameon=False, borderaxespad=0, borderpad=0, fontsize=7.7,
                  labelspacing=.18, handlelength=2, handletextpad=.5)
        for spine in ax.spines.values():
            spine.set_linewidth(.65)
    axes[0].set_ylabel(r"Median centered shear ($10^{-3}$)", fontsize=9)
    # The background has no shear units; freeze the limits after plotting all error bars.
    for ax, (counts, edges) in zip(axes, histograms):
        draw_histogram(ax, counts, edges)
    return fig
