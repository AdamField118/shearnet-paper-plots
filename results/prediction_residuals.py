"""fig:prediction-residuals -- residual ellipticity predictions on the held-out set.

Per the repository house rule, the drawing and the fit are done by
``superbit_lensing``:

    superbit_lensing.plotter.plot_comparison

which plots ``measured - truth`` against truth, fits a quadratic, marks the
``y = 0`` line and shades the tolerance band, and returns the fit coefficients
with their errors. That is exactly the figure the caption describes, so nothing
about the science is re-implemented here. Manuscript labels are applied to the
returned artists before saving. This module pulls the right columns out
of the evaluation FITS and hands them over.

WHICH COLUMNS, AND WHY
----------------------
The ``+gamma`` population at the UNROTATED ring station, against ``g_th`` for
the same objects. ``g_th`` is the composed shape -- the intrinsic ellipticity
with the applied shear on it -- so it spans the whole shape distribution and the
residual ``measured - truth`` is the estimator's per-object error.

Do not ring-average either side here. The ring exists to cancel intrinsic
ellipticity, and ``g_th`` is dominated by it, so the ring mean of the truth
collapses to the applied shear: one value for every object. The regression then
has no abscissa left, and the quadratic fit inside ``plot_comparison`` is
singular. Ring averaging is right for ``m`` and for the leakage slope, and wrong
for this figure.

Only the ``+gamma`` population is drawn. Differencing the pair would show the
pair-matched estimator that ``tab:unit-test-bias`` already reports, not the
per-object residual this figure is about.

The point cloud is rasterized before the page is written -- see
:func:`_rasterize_scatter` -- because at 2x10^5 objects per panel the vector
version is a PDF that will not open.

    python prediction_residuals.py --fits ../evaluations/fourth.fits
    python prediction_residuals.py --fits ../evaluations/fourth.fits \\
        --cut both --min-resolution 1.0 --component 0
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
from paper_numbers import _pair_tables, _shape_column
from paper_labels import ellipticity_truth, label_prediction_residuals

_NO_SUPERBIT = (
    "This figure is drawn by superbit_lensing.plotter.plot_comparison.\n"
    "Install superbit-lensing (or put it on PYTHONPATH) and run again.\n"
    "It is deliberately not reimplemented here."
)


def panel_inputs(evaluation, estimators, component=0, mask=None):
    """``(cat, reference_key, compare_keys)`` for ``plot_comparison``."""
    plus, _ = _pair_tables(evaluation, component=component)

    truth = np.asarray(plus["g_th"], dtype=float)[:, component]
    cat = {}
    keep = np.isfinite(truth)
    for estimator in estimators:
        column = _shape_column(plus, estimator)
        values = np.asarray(plus[column], dtype=float)[:, component]
        cat[estimator] = values
        keep &= np.isfinite(values)
    if mask is not None:
        keep &= mask

    reference = ellipticity_truth(component)
    cat = {name: values[keep] for name, values in cat.items()}
    cat[reference] = truth[keep]
    return cat, reference, list(estimators)


def _rasterize_scatter(axes) -> int:
    """Draw the point cloud as pixels, leaving every other artist vector.

    ``plot_comparison`` calls ``scatter`` once per panel with no rasterization,
    which is right at the sizes it was written for and fatal at ours: one PDF
    path operator per point, two panels, 2x10^5 objects each, and the page
    becomes 18 MB of uncompressed drawing commands (6 MB on disk) that viewers
    will not open.

    Rasterizing only the ``PathCollection`` leaves the axes, the fit curve, the
    tolerance band and every piece of text as vectors, so the figure still
    scales and the text is still selectable and searchable. It is the same thing
    LITB-III-plots does with its large scatters (``sec1/fig1_footprint.ipynb``,
    ``sec2/sec2.3/fig2_target_footprints.py``, both ``rasterized=True``).

    Nothing about the figure is re-implemented: this sets one property on the
    artists ``plot_comparison`` returned.
    """
    from matplotlib.collections import PathCollection

    count = 0
    for ax in np.atleast_1d(axes).ravel():
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_rasterized(True)
                count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimators", nargs="*", default=None,
                        help="up to two; plot_comparison draws one panel each")
    parser.add_argument("--component", type=int, default=0, choices=(0, 1))
    parser.add_argument("--out", type=Path, default=Path("prediction_residuals.pdf"))
    parser.add_argument("--error-allowed", type=float, default=0.01,
                        help="half-width of the shaded tolerance band")
    parser.add_argument("--cut", choices=("none", "superbit", "truth", "both"),
                        default="none")
    parser.add_argument("--min-hlr", type=float, default=None)
    parser.add_argument("--min-resolution", type=float, default=None)
    parser.add_argument("--psf-fwhm", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300,
                        help="resolution of the rasterized point cloud "
                             "(plot_comparison's own value; 300 is print)")
    parser.add_argument("--no-rasterize", action="store_true",
                        help="keep every point a vector path -- only for a "
                             "small sample, see _rasterize_scatter")
    args = parser.parse_args(argv)

    try:
        from superbit_lensing.plotter import plot_comparison, pub_rc
    except ImportError:
        sys.stderr.write(_NO_SUPERBIT)
        return 2

    evaluation = Evaluation(args.fits)
    estimators = args.estimators or [e for e in evaluation.estimators()
                                     if e in ("ngmix", "shearnet")]
    estimators = estimators[:2]

    mask = None
    if args.cut != "none":
        from selection import paired_mask

        plus, minus = _pair_tables(evaluation, component=args.component)
        mask = np.ones(len(plus), dtype=bool)
        if args.cut in ("truth", "both"):
            mask &= paired_mask(plus, minus, "truth", min_hlr=args.min_hlr,
                                min_resolution=args.min_resolution,
                                psf_fwhm=args.psf_fwhm)
        if args.cut in ("superbit", "both"):
            mask &= paired_mask(plus, minus, "superbit")

    cat, reference, compare = panel_inputs(evaluation, estimators,
                                           args.component, mask)
    print(f"{len(cat[reference])} objects, estimators: {', '.join(compare)}")

    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # superbit_lensing's own rcParams, used as-is. The one key overridden is
    # usetex, and only when TeX cannot actually render here -- the env var alone
    # was not enough, because the cluster HAS latex and still fails on a missing
    # font package, inside tight_layout, after the figure is computed.
    rc = pub_rc(fontsize=14)
    if rc.get("text.usetex") and not tex_available():
        warn_once()
        rc = {**rc, "text.usetex": False}
    with plt.rc_context(rc):
        # save_path is deliberately NOT passed: plot_comparison would write the
        # file before we can rasterize, and an unrasterized scatter of this many
        # points is a PDF no viewer will open (see _rasterize_scatter). The
        # savefig below uses plot_comparison's own dpi and bbox_inches.
        fig, axes, coefficients = plot_comparison(
            cat, reference, compare,
            error_allowed=args.error_allowed,
        )
        label_prediction_residuals(fig, axes, compare, reference, coefficients, args.component)
        if not args.no_rasterize:
            _rasterize_scatter(axes)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    for key, fit in coefficients.items():
        quadratic, errors = fit["quadratic"], fit["errors"]
        print(f"  {key}:  slope {quadratic[1]:+.5f} +/- {errors[1]:.5f}   "
              f"offset {quadratic[2]:+.5f} +/- {errors[2]:.5f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
