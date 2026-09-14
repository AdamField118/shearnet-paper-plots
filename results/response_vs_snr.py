"""fig:response_snr -- the measured responses against signal-to-noise.

Left: the shear response :math:`R^{\\gamma}`, with :math:`R_{11}` and
:math:`R_{22}` drawn separately, because :math:`D_4` equivariance constrains
their difference not at all -- every group element acts on
:math:`(e_1, e_2)` as :math:`{\\rm diag}(\\pm 1, \\pm 1)`, so conjugating
:math:`R` by it zeroes the off-diagonals and says nothing about the diagonal.
Collapsing them to one curve would hide the one thing the architecture cannot
give you.

Right: the PSF response :math:`R^{\\rm PSF}`, whose correct value is zero at
every signal-to-noise, so the horizontal line is the target and not a reference.

The dashed line on the left is the analytic ensemble target. It is **1**, not
:math:`1 - \\sigma_e^2`: the observed shape transforms as
:math:`(\\varepsilon + \\gamma)/(1 + \\bar\\gamma\\varepsilon)`, giving
:math:`R_{11} = 1 - \\varepsilon_1^2 + \\varepsilon_2^2` per object, whose mean
over an isotropic population is exactly 1. The familiar :math:`1 - \\sigma_e^2`
belongs to the distortion convention and does not transfer.

Styling comes from ``superbit_lensing.plotter.pub_rc`` so this figure matches
the ones that package draws. There is no response-versus-S/N plotter in
``superbit_lensing`` to import, so the axes are built here; the binning and the
jackknife reuse :mod:`paper_numbers`.

    python response_vs_snr.py --fits ../evaluations/fourth.fits
    python response_vs_snr.py --fits ../evaluations/fourth.fits --nbins 8 --cut both
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
from paper_numbers import DEFAULT_NJACK, _jackknife_error, _pair_tables, _ring_mean
from response_diagnostics import _matrix_column

#: The analytic ensemble target for each panel, per the module docstring.
TARGETS = {"gamma": 1.0, "psf": 0.0}

STYLE = {
    ("shearnet", "11"): dict(color="#4C72B0", marker="o", ls="-"),
    ("shearnet", "22"): dict(color="#4C72B0", marker="s", ls="--"),
    ("ngmix", "11"): dict(color="#DD8452", marker="o", ls="-"),
    ("ngmix", "22"): dict(color="#DD8452", marker="s", ls="--"),
}


def binned_response(evaluation, estimator, template, key, nbins, njack, mask=None):
    """``[(s2n_centre, {entry: (value, error)}, n)]`` in equal-count bins."""
    plus, minus = _pair_tables(evaluation, component=0)
    up, down = (_matrix_column(plus, template, estimator),
                _matrix_column(minus, template, estimator))
    if up is None or down is None:
        return []
    matrix = 0.5 * (up + down)

    if key not in plus.colnames:
        return []
    s2n = np.asarray(plus[key], dtype=float)

    keep = np.isfinite(matrix).all(axis=(1, 2)) & np.isfinite(s2n)
    if mask is not None:
        keep &= mask
    index = np.flatnonzero(keep)
    index = index[np.argsort(s2n[index])]
    if len(index) < nbins * njack:
        nbins = max(1, len(index) // max(njack, 1))
    if nbins < 1:
        return []

    edges = np.linspace(0, len(index), nbins + 1).astype(int)
    out = []
    for i in range(nbins):
        selected = index[edges[i]:edges[i + 1]]
        if len(selected) < njack:
            continue
        entries = {}
        for a, b, name in ((0, 0, "11"), (1, 1, "22")):
            values = matrix[selected, a, b]
            entries[name] = (float(values.mean()), _jackknife_error(values, njack))
        out.append((float(np.median(s2n[selected])), entries, len(selected)))
    return out


def draw(evaluation, estimators, *, nbins, njack, key, mask, out_path):
    from superbit_lensing.plotter import pub_rc

    import matplotlib.pyplot as plt

    # superbit_lensing's own rcParams, used as-is; usetex dropped only when
    # TeX cannot render here.
    rc = pub_rc(fontsize=14)
    if rc.get("text.usetex") and not tex_available():
        warn_once()
        rc = {**rc, "text.usetex": False}

    panels = (
        ("gamma", "Rgamma_{est}_metacal", r"$R^{\gamma}$"),
        ("psf", "Rpsf_{est}_metacal", r"$R^{\rm PSF}$"),
    )
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (panel, template, ylabel) in zip(axes, panels):
            target = TARGETS[panel]
            drew = False
            for estimator in estimators:
                points = binned_response(evaluation, estimator, template, key,
                                         nbins, njack, mask)
                for entry in ("11", "22"):
                    if not points:
                        continue
                    x = [p[0] for p in points]
                    y = [p[1][entry][0] for p in points]
                    e = [p[1][entry][1] for p in points]
                    style = STYLE.get((estimator, entry), {})
                    ax.errorbar(x, y, yerr=e, capsize=3, lw=1.4, ms=5,
                                label=rf"{estimator} $R_{{{entry}}}$", **style)
                    drew = True
            ax.axhline(target, color="k", ls=":", lw=1.4,
                       label=("analytic target" if panel == "gamma"
                              else "zero (the target)"))
            ax.set_xscale("log")
            ax.set_xlabel("signal-to-noise ratio")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, ncol=2)
            if not drew:
                ax.text(0.5, 0.5, "no response columns", ha="center",
                        va="center", transform=ax.transAxes)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--nbins", type=int, default=8)
    parser.add_argument("--njack", type=int, default=DEFAULT_NJACK)
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

    path = draw(evaluation, estimators, nbins=args.nbins, njack=args.njack,
                key=args.s2n_column, mask=mask, out_path=args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
