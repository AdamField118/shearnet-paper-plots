"""tab:response-diag -- the measured response matrices of the fiducial model.

The ensemble metacalibration matrices, reported directly without subtracting an
identity or target: a measured shape statistic need not have unit response,
and response fidelity is assessed through the calibrated shear bias.

Rows, per estimator:

* ``R^gamma`` -- the mean metacal shear response over the selected sample;
* ``R^S``     -- metacal's selection response of the cut (zero with ``--cut none``);
* ``R^PSF``   -- the mean metacal PSF response over the selected sample.

``R^gamma + R^S`` is the response m and c are calibrated by (printed as a
comment). Every entry is the mean of the +g1 and -g1 populations, over every
ring station, with a delete-one-block jackknife over objects. The cut and its
bookkeeping are :mod:`shear_stats`'s; the comment lines say which was used.

    python response_diagnostics.py --fits ../evaluations/fourth.fits
    python response_diagnostics.py --fits ../evaluations/fourth.fits --cut none
"""

from __future__ import annotations

import argparse
from pathlib import Path


from catalog import Catalog
from shear_stats import (DEFAULT_NJACK, ESTIMATORS, G1_PAIR, Sample, cut_from_name,
                         responses)

ROWS = (("Rg", r"$R^{\gamma}$"), ("RS", r"$R^{\rm S}$"), ("Rp", r"$R^{\rm PSF}$"))
ENTRIES = ((0, 0, "11"), (1, 1, "22"), (0, 1, "12"), (1, 0, "21"))


def measure(catalog, estimators, cut, njack=DEFAULT_NJACK) -> dict:
    sample = Sample(catalog, list(G1_PAIR), cut=cut, estimators=estimators)
    return {"sample": sample,
            "results": {est: responses(sample, est, njack=njack) for est in estimators}}


def latex_rows(results) -> list:
    lines = []
    for key, label in ROWS:
        for est, r in results.items():
            value, error = r[key]
            cells = [f"${value[i, j]:+.4f} \\pm {error[i, j]:.4f}$" for i, j, _ in ENTRIES]
            lines.append(f"{label} & \\textsc{{{est}}} & " + " & ".join(cells) + r" \\")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimators", nargs="*", default=list(ESTIMATORS))
    parser.add_argument("--njack", type=int, default=DEFAULT_NJACK)
    parser.add_argument("--cut", choices=("metacal", "none"), default="metacal",
                        help="metacal (default): ngmix T/Tpsf > 1 and s2n > 10 on each "
                             "metacal product's own fit, with R^S. none: every record.")
    parser.add_argument("--min-t-ratio", type=float, default=None)
    parser.add_argument("--min-s2n", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    catalog = Catalog(args.fits)
    measured = measure(catalog, args.estimators, cut, args.njack)
    sample, results = measured["sample"], measured["results"]

    counts = sample.counts()
    lines = [
        f"% tab:response-diag from {args.fits.name} ({catalog.run_name})",
        "% columns: response & estimator & 11 & 22 & 12 & 21",
        "% entries are means over the +g1 and -g1 populations and every ring station;",
        f"% errors: delete-one-block jackknife over objects, {args.njack} blocks",
        f"% cut: {cut.describe() if cut else 'none (every measurable record)'}",
    ]
    for scene in G1_PAIR:
        c = counts[scene]
        lines.append(f"%   {scene}: {c['selected_noshear']} of {c['records']} records "
                     f"selected ({c['selected_noshear'] / c['records']:.1%})")
    for est, r in results.items():
        total, error = r["R"]
        lines.append(f"% R = R^gamma + R^S, {est}: "
                     + ", ".join(f"R{name} = {total[i, j]:+.4f} +/- {error[i, j]:.4f}"
                                 for i, j, name in ENTRIES))
    lines += latex_rows(results)
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\n% wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
