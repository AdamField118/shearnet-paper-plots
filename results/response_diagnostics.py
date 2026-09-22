"""tab:response-diag -- the measured response matrices of the fiducial model.

Report the ensemble metacalibration matrices directly, without subtracting an
identity or target. A measured shape statistic need not have unit response.
Response fidelity is assessed through calibrated shear bias, not its amplitude.
Per-object columns are ring- and pair-averaged with delete-one-block errors.

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

RESPONSES = (
    ("Rgamma_{est}_metacal", r"$R^{\gamma}$"),
    ("Rpsf_{est}_metacal", r"$R^{\rm PSF}$"),
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
    for template, label in RESPONSES:
        for estimator in estimators:
            result = measure(evaluation, estimator, template, njack, mask)
            if result is None:
                lines.append(f"{label} & \\textsc{{{estimator}}} & "
                             + " & ".join([r"\pending"] * len(ENTRIES)) + r" \\")
                continue
            cells = []
            for i, j, name in ENTRIES:
                value, error = result[name]
                cells.append(f"${value:+.4f} \\pm {error:.4f}$")
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
        r"% entries are mean measured responses; no identity subtraction.",
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
