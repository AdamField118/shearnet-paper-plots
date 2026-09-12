"""Localise the metacal numerator inflation, per object.

The ensemble numbers say the following, for every UT4-dataset run:

    num_plain / <R>  =  1.00 to 1.04      (the plain fit, divided by metacal's R)
    num_metacal / <R> = 1.11 to 1.17      (metacal's own noshear shape)

So metacal's RESPONSE is right -- it calibrates the plain measurement to a few
percent -- and the bias is entirely that metacal's noshear shape is larger than
the plain fit of the same galaxy. The inflation is 1.2% on UT1 and 11.8% on UT4,
and it switches on between UT2 and UT3, which is where the COSMOS size
distribution enters.

This script asks WHICH OBJECTS inflate, by splitting the same paired estimator
into bins of resolution and signal-to-noise:

    m_plain    = <(e_plain+  - e_plain-)/2>  / <R> / gamma - 1
    m_metacal  = <(e_mcal+   - e_mcal-)/2>   / <R> / gamma - 1

computed identically except for which shape column feeds the numerator, so any
difference between them is the inflation and nothing else.

    python diagnose_metacal.py --fits evaluations/fourth.fits
    python diagnose_metacal.py --fits evaluations/second.fits --estimator ngmix

If the inflation is flat across both binnings it is not a resolution effect and
the reconvolution is suspect globally. If it piles up in the poorly resolved or
faint bins, it is the deconvolve/reconvolve step failing where it is known to be
hardest, and the reconvolution PSF is the thing to change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation_fits import Evaluation
from paper_numbers import _pair_tables, _ring_mean, _station_suffixes


def _first_present(table, candidates):
    for base in candidates:
        if _station_suffixes(table.colnames, base):
            return base
    return None


def paired(plus, minus, base, column=0):
    """``(e_plus - e_minus)/2`` per object, ring-averaged, one component."""
    return 0.5 * (_ring_mean(plus, base)[:, column] - _ring_mean(minus, base)[:, column])


def response(plus, minus, base):
    """``(R_plus + R_minus)/2`` per object, ring-averaged, the 11 entry."""
    return 0.5 * (_ring_mean(plus, base)[:, 0, 0] + _ring_mean(minus, base)[:, 0, 0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimator", default="ngmix")
    parser.add_argument("--nbin", type=int, default=6)
    args = parser.parse_args(argv)

    ev = Evaluation(args.fits)
    est = args.estimator
    gamma = float(ev.header.get("SHEAR_TR", 0.01))
    plus, minus = _pair_tables(ev, component=0)

    plain_col = _first_present(plus, [f"e_{est}", f"e_{est}_uncorrected"])
    mcal_col = _first_present(plus, [f"e_{est}_metacal_raw",
                                     f"e_{est}_metacal_corrected"])
    resp_col = _first_present(plus, [f"Rgamma_{est}_metacal", f"R_{est}_metacal"])
    if not all((plain_col, mcal_col, resp_col)):
        raise SystemExit(
            f"missing a column for {est!r}: plain={plain_col} "
            f"metacal={mcal_col} response={resp_col}"
        )

    num_plain = paired(plus, minus, plain_col)
    num_mcal = paired(plus, minus, mcal_col)
    resp = response(plus, minus, resp_col)
    finite = np.isfinite(num_plain) & np.isfinite(num_mcal) & np.isfinite(resp)

    print(f"file       {args.fits.name}")
    print(f"estimator  {est}   gamma={gamma}   n={finite.sum()} of {len(finite)}")
    print(f"columns    plain={plain_col}  metacal={mcal_col}  R={resp_col}\n")

    r = resp[finite].mean()
    for name, num in (("plain", num_plain), ("metacal", num_mcal)):
        m = num[finite].mean() / r / gamma - 1.0
        print(f"  ensemble m ({name:7s}) = {m:+.5f}     "
              f"numerator/gamma = {num[finite].mean()/gamma:.4f}")
    print(f"  ensemble <R11>        = {r:.4f}")
    inflation = num_mcal[finite].mean() / num_plain[finite].mean()
    print(f"  metacal / plain       = {inflation:.4f}"
          f"   <- the whole bias is here\n")

    # Resolution is what separates UT2 from UT3, so it is the first split.
    keys = []
    tpsf = np.asarray(plus["Tpsf"], dtype=float) if "Tpsf" in plus.colnames else None
    tgal = np.asarray(plus[f"T_{est}"], dtype=float) if f"T_{est}" in plus.colnames else None
    if tgal is not None and tpsf is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            keys.append(("resolution T_gal/T_PSF", tgal / tpsf))
    if "hlr_th" in plus.colnames:
        keys.append(("half-light radius", np.asarray(plus["hlr_th"], dtype=float)))
    if "s2n" in plus.colnames:
        keys.append(("signal-to-noise", np.asarray(plus["s2n"], dtype=float)))

    for label, key in keys:
        ok = finite & np.isfinite(key)
        print(f"  --- by {label} ---")
        print(f"  {'range':>22s} {'n':>7s} {'<R11>':>7s} "
              f"{'m_plain':>9s} {'m_metacal':>10s} {'mcal/plain':>11s}")
        order = np.argsort(key[ok])
        idx = np.flatnonzero(ok)[order]
        edges = np.linspace(0, len(idx), args.nbin + 1).astype(int)
        for i in range(args.nbin):
            sel = idx[edges[i]:edges[i + 1]]
            if len(sel) < 50:
                continue
            rr = resp[sel].mean()
            mp = num_plain[sel].mean() / rr / gamma - 1.0
            mm = num_mcal[sel].mean() / rr / gamma - 1.0
            ratio = num_mcal[sel].mean() / num_plain[sel].mean()
            print(f"  {key[sel].min():9.3f}..{key[sel].max():<10.3f} {len(sel):7d} "
                  f"{rr:7.4f} {mp:+9.4f} {mm:+10.4f} {ratio:11.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
