"""tab:response-diag -- the measured response matrices of the fiducial model.

The caption asks for two things per estimator: the shear response
:math:`R^{\\gamma}`, whose ensemble target is the identity, and the PSF response
:math:`R^{\\rm PSF}`, whose target is zero in every entry. Both are read from the
per-object columns and averaged over the ring, with a delete-one-block jackknife
over :data:`paper_numbers.DEFAULT_NJACK` blocks -- the same estimator and the
same block count the rest of the paper's errors use.

WHY THE ENSEMBLE TARGET FOR R11 IS 1 AND NOT 1 - sigma_e^2
-----------------------------------------------------------
The observed shape transforms as ``(eps + gamma) / (1 + conj(gamma) eps)``, so
per object ``R11 = 1 - eps1^2 + eps2^2``, which exceeds 1 whenever the galaxy is
elongated along the second component. Averaging over an isotropic distribution
gives ``<eps1^2> = <eps2^2>`` and therefore ``<R11> = 1`` exactly. The familiar
``1 - sigma_e^2`` intuition belongs to the *distortion* convention and does not
transfer. ShearNet's own ``gamma_target: analytic`` trains against exactly this
derivative (``shearnet/core/inloop.py``), so the comparison below is against the
quantity the network was optimized to match, not a convention chosen here.

The translation and isotropy rows of the original table are not emitted: those
are training-time diagnostics and the evaluation FITS carries no column for
them.

    python response_diagnostics.py --fits ../evaluations/fourth.fits
    python response_diagnostics.py --fits ../evaluations/fourth.fits --cut both \\
        --min-resolution 1.0 --psf-fwhm 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation_fits import Evaluation
from paper_numbers import DEFAULT_NJACK, _jackknife_error, _pair_tables, _ring_mean

#: (column template, LaTeX row label, target matrix). The target is what the
#: entry should be if the estimator is right, and it is printed beside the
#: measurement so a reader does not have to remember which entries are zero.
RESPONSES = (
    ("Rgamma_{est}_metacal", r"$R^{\gamma}$", np.eye(2)),
    ("Rpsf_{est}_metacal", r"$R^{\rm PSF}$", np.zeros((2, 2))),
)

ENTRIES = ((0, 0, "11"), (1, 1, "22"), (0, 1, "12"), (1, 0, "21"))


def _matrix_column(table, template: str, estimator: str):
    """The per-object 2x2 response, ring-averaged, or None if absent."""
    from paper_numbers import _station_suffixes

    base = template.format(est=estimator)
    if not _station_suffixes(table.colnames, base):
        return None
    return _ring_mean(table, base)


def measure(evaluation, estimator: str, template: str, njack=DEFAULT_NJACK,
            mask=None):
    """``{entry: (value, error)}`` for one response matrix.

    Both populations are averaged, because the response is a property of the
    measurement rather than of the applied shear, and using one sign would throw
    away half the sample for no reason.
    """
    plus, minus = _pair_tables(evaluation, component=0)
    up, down = (_matrix_column(plus, template, estimator),
                _matrix_column(minus, template, estimator))
    if up is None or down is None:
        return None
    matrix = 0.5 * (up + down)

    keep = np.isfinite(matrix).all(axis=(1, 2))
    if mask is not None:
        keep &= mask
    matrix = matrix[keep]
    if len(matrix) < njack:
        return None

    out = {}
    for i, j, name in ENTRIES:
        values = matrix[:, i, j]
        out[name] = (float(values.mean()), _jackknife_error(values, njack))
    out["n_used"] = len(matrix)
    return out


def latex_rows(evaluation, estimators, njack=DEFAULT_NJACK, mask=None):
    lines = []
    for template, label, target in RESPONSES:
        for estimator in estimators:
            result = measure(evaluation, estimator, template, njack, mask)
            if result is None:
                lines.append(f"{label} & \\textsc{{{estimator}}} & "
                             + " & ".join([r"\pending"] * len(ENTRIES)) + r" \\")
                continue
            cells = []
            for i, j, name in ENTRIES:
                value, error = result[name]
                # Subtract the target so the column reads as a residual, which
                # is what the caption promises: the entries that should vanish
                # then all sit at zero and are directly comparable.
                cells.append(f"${value - target[i, j]:+.4f} \\pm {error:.4f}$")
            lines.append(f"{label} & \\textsc{{{estimator}}} & "
                         + " & ".join(cells) + r" \\")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--njack", type=int, default=DEFAULT_NJACK)
    parser.add_argument("--cut", choices=("none", "superbit", "truth", "both"),
                        default="none")
    parser.add_argument("--min-hlr", type=float, default=None)
    parser.add_argument("--min-resolution", type=float, default=None)
    parser.add_argument("--psf-fwhm", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

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

    lines = [
        f"% tab:response-diag from {args.fits.name}",
        f"% columns: response & estimator & R11 & R22 & R12 & R21",
        r"% entries are MEASURED MINUS TARGET: R^gamma against the identity,",
        r"% R^PSF against zero, so every column should read consistent with 0.",
        f"% jackknife blocks: {args.njack}   sample cut: {args.cut}",
        *latex_rows(evaluation, estimators, args.njack, mask),
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\n% wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
