"""fig:response_snr -- the measured responses against signal-to-noise.

WHERE THE DRAWING COMES FROM
----------------------------
The house rule is to take a plotter from, in order, ``s-Sayan/LITB-III-plots``,
then ``superbit-collaboration/superbit-lensing``, then ShearNet's own
``research/shear_bias/plots_from_fits.ipynb``, and only then write one.

* LITB-III-plots has no response-versus-S/N figure. Its closest relative,
  ``sec5/fig19_shear_bias.ipynb``, is shear bias against cluster *radius*.
* ``superbit_lensing.plotter`` has no response-versus-S/N plotter either; its
  binning helpers all live inside :class:`PSFLeakagePanelMaker` and bin against
  PSF ellipticity.
* ``plots_from_fits.ipynb`` has exactly this figure, twice: cell 22 draws
  metacal :math:`R^{\\gamma}` against S/N and cell 26 draws metacal
  :math:`R^{\\rm PSF}` the same way.

So the drawing is that notebook's, copied rather than paraphrased:
:func:`_bin_response`, :func:`_as_sorted` and :func:`style_log_x` are its cells
verbatim, and :func:`draw` makes the same artist calls in the same order --
shaded +/-1 SE band, dashed connecting line, open-circle error bars, the 1/2/5
decade ticks.

Three things differ from the notebook, each because this is one paper figure
rather than two working PNGs:

1. The two panels share a figure instead of being saved separately.
2. The legend sits inside the axes. The notebook anchors it outside to the
   right, which in a 1x2 grid lands on top of the neighbouring panel.
3. An ensemble reference line is drawn, and the y-range is widened to include it.

WHY BOTH DIAGONALS
------------------
:math:`R_{11}` and :math:`R_{22}` are drawn separately because :math:`D_4`
equivariance constrains their difference not at all -- every group element acts
on :math:`(e_1, e_2)` as :math:`{\\rm diag}(\\pm 1, \\pm 1)`, so conjugating
:math:`R` by it zeroes the off-diagonals and says nothing about the diagonal.
Collapsing them to one curve would hide the one thing the architecture cannot
give you. (Cell 22 draws both; cell 26 draws only :math:`R^{\\rm PSF}_{11}`,
and is extended here to the second diagonal in cell 22's own style.)

The unit-response line on the left is the isotropic ensemble reference for
reduced ellipticity. It is not a bin-averaged per-object training derivative,
and these evaluation responses are not derivatives of the training renderer.
The right panel shows the metacalibration PSF response with zero as its reference.
The plotted central values are within-bin medians; error bars and bands use the
notebook's standard deviation divided by the square root of the bin count.

Styling is ``superbit_lensing.plotter.pub_rc``, as every other figure here.

    python response_vs_snr.py --fits ../evaluations/fourth.fits
    python response_vs_snr.py --fits ../evaluations/fourth.fits --nbins 15 --cut both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from evaluation_fits import Evaluation
from plotstyle import tex_available, warn_once
from paper_numbers import _pair_tables
from response_diagnostics import _matrix_column

#: Ensemble reference lines, not per-object training targets.
TARGETS = {"gamma": 1.0, "psf": 0.0}

from paper_colors import COLORS, RESPONSE_STYLES


# ---------------------------------------------------------------------------
# plots_from_fits.ipynb cell 22, verbatim. Do not paraphrase these.
# ---------------------------------------------------------------------------

def _bin_response(r_arr, snr_arr, n_bins=25):
    valid = np.isfinite(r_arr) & np.isfinite(snr_arr) & (snr_arr > 0)
    r, s = r_arr[valid], snr_arr[valid]
    edges = np.percentile(s, np.linspace(0, 100, n_bins + 1))
    idx = np.digitize(s, edges)
    centers, meds, errs = [], [], []
    for i in range(1, len(edges)):
        mask = idx == i
        if mask.sum() > 0:
            centers.append(np.median(s[mask]))
            meds.append(np.median(r[mask]))
            errs.append(np.std(r[mask]) / np.sqrt(mask.sum()))
    return np.array(centers), np.array(meds), np.array(errs)


def _as_sorted(x, y, e):
    x, y, e = np.asarray(x), np.asarray(y), np.asarray(e)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = x[m], y[m], e[m]
    s = np.argsort(x)
    return x[s], y[s], e[s]


def style_log_x(ax, xlo, xhi):
    import matplotlib.ticker as mticker

    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)

    decades = np.arange(
        np.floor(np.log10(xlo)),
        np.ceil(np.log10(xhi)) + 1,
    )
    ticks = np.sort(np.concatenate([
        multiplier * 10.0**decades
        for multiplier in (1, 2, 5)
    ]))
    ticks = ticks[(ticks >= xlo) & (ticks <= xhi)]

    # Keep at most eight labeled ticks.
    if len(ticks) > 8:
        ticks = ticks[::int(np.ceil(len(ticks) / 8))]

    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:g}")
    )
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=10,
        labelrotation=0,
        pad=6,
    )
    ax.tick_params(which="both", direction="out", top=False, right=False)


# ---------------------------------------------------------------------------


def response_arrays(evaluation, estimator, template, key, mask=None):
    """``(s2n, {'11': R11, '22': R22})`` per object, or ``None`` if absent.

    The response is the mean of the +gamma and -gamma measurements of the same
    galaxy, which is the same quantity the notebook reads off ``TAB_P`` alone
    with half the variance, and it is what ``tab:response-diag`` already uses.
    ``_matrix_column`` averages over the ring stations.
    """
    plus, minus = _pair_tables(evaluation, component=0)
    up, down = (_matrix_column(plus, template, estimator),
                _matrix_column(minus, template, estimator))
    if up is None or down is None or key not in plus.colnames:
        return None
    matrix = 0.5 * (up + down)
    s2n = np.asarray(plus[key], dtype=float)

    keep = np.isfinite(matrix).all(axis=(1, 2)) & np.isfinite(s2n)
    if mask is not None:
        keep &= mask
    if not keep.any():
        return None
    return s2n[keep], {"11": matrix[keep, 0, 0], "22": matrix[keep, 1, 1]}


def draw(evaluation, estimators, *, nbins, key, mask, out_path):
    from superbit_lensing.plotter import pub_rc

    import matplotlib.pyplot as plt

    # superbit_lensing's own rcParams, used as-is; usetex dropped only when
    # TeX cannot render here.
    rc = pub_rc(fontsize=14)
    if rc.get("text.usetex") and not tex_available():
        warn_once()
        rc = {**rc, "text.usetex": False}

    panels = (
        ("gamma", "Rgamma_{est}_metacal", r"Metacal $R^{\gamma}$",
         r"Metacalibration $R^\gamma$", r"R^{\gamma}"),
        ("psf", "Rpsf_{est}_metacal", r"Metacal $R^{\rm PSF}$",
         r"Metacalibration $R^{\rm PSF}$", r"R^{\rm PSF}"),
    )
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.75),
                                 constrained_layout=True)
        for ax, (panel, template, ylabel, title, symbol) in zip(axes, panels):
            target = TARGETS[panel]
            # The notebook's y-range is set by the data; the target line has to
            # be inside it or the one horizontal reference in the figure is
            # cropped away.
            ymin, ymax = target, target
            all_x = []
            for estimator in estimators:
                arrays = response_arrays(evaluation, estimator, template, key,
                                         mask)
                if arrays is None:
                    continue
                s2n, components = arrays
                for entry in ("11", "22"):
                    color = COLORS.get(estimator, "0.4")
                    linestyle, marker = RESPONSE_STYLES[(estimator, entry)]
                    x, y, e = _as_sorted(*_bin_response(components[entry], s2n,
                                                        nbins))
                    if not len(x):
                        continue
                    all_x.append(x)
                    ax.fill_between(x, y - e, y + e, color=color, alpha=0.18,
                                    linewidth=0)
                    ax.plot(x, y, c=color, ls=linestyle, lw=2.2, alpha=0.95,
                            label=fr"{estimator} ${symbol}_{{{entry}}}$")
                    ax.errorbar(
                        x, y, yerr=e, c=color, fmt=marker, ms=4.0, mfc="white",
                        mew=1.1, capsize=2.5, elinewidth=1.0, lw=0, alpha=0.95,
                        label="_nolegend_",
                    )
                    ymin = min(ymin, np.nanmin(y - e))
                    ymax = max(ymax, np.nanmax(y + e))

            if not all_x:
                ax.text(0.5, 0.5, "no response columns", ha="center",
                        va="center", transform=ax.transAxes)
                continue

            ax.axhline(target, color="k", ls=":", lw=1.4,
                       label=("unit response" if panel == "gamma"
                              else "zero response"))
            ypad = 0.06 * (ymax - ymin) if np.isfinite(ymax - ymin) else 0.05
            ax.set_ylim(ymin - ypad, ymax + ypad)

            joined = np.concatenate(all_x)
            joined = joined[np.isfinite(joined) & (joined > 0)]
            log_pad = 0.15 * (np.log10(joined.max()) - np.log10(joined.min()))
            xlo = 10 ** (np.log10(joined.min()) - log_pad)
            xhi = 10 ** (np.log10(joined.max()) + log_pad)
            style_log_x(ax, xlo, xhi)
            ax.set_xlabel("SNR")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=12)
            if panel == "gamma":
                ax.legend(frameon=False, fontsize=10, loc="best")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--nbins", type=int, default=25,
                        help="quantile bins in S/N (the notebook's value)")
    parser.add_argument("--s2n-column", default="s2n")
    parser.add_argument("--out", type=Path, default=Path("response_vs_snr.pdf"))
    parser.add_argument("--cut", choices=("none", "superbit", "truth", "both"),
                        default="none")
    parser.add_argument("--min-hlr", type=float, default=None)
    parser.add_argument("--min-resolution", type=float, default=None)
    parser.add_argument("--psf-fwhm", type=float, default=0.5)
    args = parser.parse_args(argv)

    try:
        import superbit_lensing.plotter  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "This figure is styled by superbit_lensing.plotter.pub_rc.\n"
            "Install superbit-lensing (or put it on PYTHONPATH) and run again.\n"
        )
        return 2

    evaluation = Evaluation(args.fits)
    estimators = args.estimators or [e for e in evaluation.estimators()
                                     if e in ("shearnet", "ngmix")]

    mask = None
    if args.cut != "none":
        from selection import paired_mask

        plus, minus = _pair_tables(evaluation, component=0)
        mask = np.ones(len(plus), dtype=bool)
        if args.cut in ("truth", "both"):
            mask &= paired_mask(plus, minus, "truth", min_hlr=args.min_hlr,
                                min_resolution=args.min_resolution,
                                psf_fwhm=args.psf_fwhm)
        if args.cut in ("superbit", "both"):
            mask &= paired_mask(plus, minus, "superbit")

    path = draw(evaluation, estimators, nbins=args.nbins, key=args.s2n_column,
                mask=mask, out_path=args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
