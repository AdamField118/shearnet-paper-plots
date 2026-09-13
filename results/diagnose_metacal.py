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

It piles up: on UT4 the smallest hlr bin has m_metacal = +0.70 and the largest
+0.013. So the thing to change is the SAMPLE, and ``--cut`` applies one --
SuperBIT's own thresholds, or the shear-independent truth cut. See
:mod:`selection` for why those two are not interchangeable.

    python diagnose_metacal.py --fits evaluations/fourth.fits --cut superbit
    python diagnose_metacal.py --fits evaluations/fourth.fits --cut truth \\
        --min-resolution 1.0 --psf-fwhm 0.5
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


def paired_sum(plus, minus, base, column=0):
    """``(e_plus + e_minus)/2`` per object -- a shear-independent outlier score.

    The paired estimator differences the two populations, so the applied shear
    enters ``e_plus`` and ``e_minus`` with opposite signs and CANCELS in this
    sum, while the ring average has already cancelled the intrinsic
    ellipticity. What is left is measurement noise and fit pathology, with an
    expectation of zero.

    That makes |this| the right thing to rank objects by when hunting for the
    handful that own the variance: trimming on it cannot preferentially delete
    high-response galaxies, which is exactly what trimming on the numerator
    itself would do, and that would bias m low by construction.
    """
    return 0.5 * (_ring_mean(plus, base)[:, column] + _ring_mean(minus, base)[:, column])


def response(plus, minus, base):
    """``(R_plus + R_minus)/2`` per object, ring-averaged, the 11 entry."""
    return 0.5 * (_ring_mean(plus, base)[:, 0, 0] + _ring_mean(minus, base)[:, 0, 0])


def naive_error(num, resp, gamma):
    """Delta-method error on ``<num>/<R>/gamma``, assuming objects are i.i.d.

    This is the error the sample would have if every object were an
    independent draw. Compared against the jackknife it is a test of the
    BLOCKING rather than of the data: the two agree when jackknife blocks are
    interchangeable, and the jackknife runs away when they are not -- which is
    what happens if objects arrive in an order correlated with the thing m
    depends on, because then each deleted block is a different galaxy
    population rather than a different sample of one.
    """
    n = len(num)
    if n < 4:
        return float("nan")
    mean_num, mean_resp = num.mean(), resp.mean()
    if mean_resp == 0:
        return float("nan")
    ratio = mean_num / mean_resp
    covariance = np.cov(num, resp)
    variance = (covariance[0, 0]
                - 2.0 * ratio * covariance[0, 1]
                + ratio**2 * covariance[1, 1]) / (n * mean_resp**2)
    return float(np.sqrt(max(variance, 0.0)) / gamma)


def m_with_error(num, resp, gamma, njack=20, shuffle=True):
    """``(m, sigma_m)`` for the ratio-of-means estimator.

    m is ``<num>/<R>/gamma - 1``, so its error is not the error on a mean and
    cannot be read off the scatter of ``num``: numerator and denominator are
    correlated per object and the ratio has to be re-formed inside each
    jackknife sample. Printing m without this invites reading a 2-sigma
    fluctuation in a subsample as a result, which is exactly the decision about
    to be made from these tables.
    """
    n = len(num)
    if n < 4:
        return float("nan"), float("nan")
    if shuffle:
        # Blocks must be interchangeable. Objects arrive in catalog order, and
        # m depends steeply on size, so contiguous blocks are different galaxy
        # populations rather than repeat samples of one -- which inflates the
        # jackknife without anything being wrong with the data. A fixed
        # permutation makes the blocks exchangeable and keeps the number
        # reproducible.
        index = np.random.default_rng(12345).permutation(n)
        num, resp = num[index], resp[index]
    njack = max(2, min(int(njack), n))
    edges = np.linspace(0, n, njack + 1).astype(int)
    num_total, resp_total = num.sum(), resp.sum()
    samples = []
    for i in range(njack):
        lo, hi = edges[i], edges[i + 1]
        keep = n - (hi - lo)
        if keep < 1:
            continue
        samples.append(((num_total - num[lo:hi].sum()) / keep)
                       / ((resp_total - resp[lo:hi].sum()) / keep) / gamma - 1.0)
    samples = np.asarray(samples, dtype=float)
    m = num.mean() / resp.mean() / gamma - 1.0
    error = np.sqrt((len(samples) - 1) / len(samples) * np.sum((samples - samples.mean()) ** 2))
    return float(m), float(error)


def _outlier_census(plus, minus, mcal_col, num_plain, num_mcal, resp,
                    finite, gamma, njack):
    """How few objects own the mean, and what m looks like without them.

    An m error of 0.018 on 42000 objects is ten to thirty times coarser than
    the paper needs, and it grows when the sample is cut harder rather than
    shrinking. That is the signature of a heavy tail: a handful of catastrophic
    fits carrying both the mean and its variance, so every threshold chosen
    from these tables is being fitted to noise.

    Objects are ranked by |(e_plus + e_minus)/2|, which carries no shear (see
    :func:`paired_sum`), so the trim cannot manufacture the answer it is
    testing for.
    """
    score = np.abs(paired_sum(plus, minus, mcal_col))
    usable = finite & np.isfinite(score)
    order = np.flatnonzero(usable)[np.argsort(score[usable])]
    n = len(order)
    if n < 100:
        return

    total = num_mcal[order].sum()
    print("  --- how concentrated is the metacal numerator? ---")
    for k in sorted({1, 10, 100, max(1, n // 1000), max(1, n // 100)}):
        worst = order[-k:]
        share = num_mcal[worst].sum() / total if total else float("nan")
        print(f"  the {k:6d} most extreme objects ({100.0 * k / n:6.3f}% of the "
              f"sample) carry {100.0 * share:+7.2f}% of the sum")

    print("\n  --- m after trimming that tail (shear-independent trim) ---")
    print(f"  {'trimmed':>9s} {'n':>7s} {'m_plain':>18s} {'m_metacal':>18s} "
          f"{'mcal/plain':>11s}")
    for fraction in (0.0, 1e-4, 1e-3, 1e-2, 3e-2):
        keep = order[: n - int(round(fraction * n))] if fraction else order
        if len(keep) < 100:
            continue
        mp, mp_err = m_with_error(num_plain[keep], resp[keep], gamma, njack)
        mm, mm_err = m_with_error(num_mcal[keep], resp[keep], gamma, njack)
        ratio = num_mcal[keep].mean() / num_plain[keep].mean()
        print(f"  {100 * fraction:8.2f}% {len(keep):7d} "
              f"{mp:+9.4f}+/-{mp_err:<6.4f} {mm:+9.4f}+/-{mm_err:<6.4f} "
              f"{ratio:11.4f}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--estimator", default="ngmix")
    parser.add_argument("--nbin", type=int, default=6)
    parser.add_argument("--cut", choices=("none", "superbit", "truth", "both"),
                        default="none",
                        help="restrict the sample before measuring (see selection.py)")
    parser.add_argument("--njack", type=int, default=20,
                        help="jackknife blocks for the error on every m")
    parser.add_argument("--min-sn", type=float, default=None,
                        help="override SuperBIT's min_sn (default 15, far below "
                             "this sample's floor of ~51, so it cuts nothing)")
    parser.add_argument("--max-sn", type=float, default=None,
                        help="override SuperBIT's max_sn (default 1000)")
    parser.add_argument("--min-hlr", type=float, default=None,
                        help="--cut truth: keep hlr_th >= this, in arcsec")
    parser.add_argument("--min-resolution", type=float, default=None,
                        help="--cut truth: keep hlr_th >= this many PSF half-widths")
    parser.add_argument("--psf-fwhm", type=float, default=0.5,
                        help="--cut truth: PSF FWHM in arcsec, for --min-resolution")
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

    if args.cut != "none":
        from selection import paired_mask

        if args.cut in ("truth", "both"):
            finite &= paired_mask(plus, minus, "truth", min_hlr=args.min_hlr,
                                  min_resolution=args.min_resolution,
                                  psf_fwhm=args.psf_fwhm)
        if args.cut in ("superbit", "both"):
            cuts = None
            if args.min_sn is not None or args.max_sn is not None:
                from selection import superbit_cuts

                cuts = superbit_cuts()
                if args.min_sn is not None:
                    cuts["min_sn"] = args.min_sn
                if args.max_sn is not None:
                    cuts["max_sn"] = args.max_sn
            finite &= paired_mask(plus, minus, "superbit", cuts=cuts)

    print(f"file       {args.fits.name}")
    print(f"estimator  {est}   gamma={gamma}   n={finite.sum()} of {len(finite)}")
    print(f"columns    plain={plain_col}  metacal={mcal_col}  R={resp_col}")
    print(f"cut        {args.cut}\n")

    if finite.sum() < 100:
        # A threshold outside the data's range is a typo, not a result, and
        # every number below it would be NaN. Print what the sample actually
        # spans so the next attempt lands inside it.
        print(f"  only {finite.sum()} objects survive the cut -- nothing to measure.\n")
        for name in ("s2n_ngmix", "s2n", "T_ngmix", "Tpsf", "hlr_th"):
            if name in plus.colnames:
                values = np.asarray(plus[name], dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    print(f"  {name:>10s} spans {values.min():.4g} to "
                          f"{values.max():.4g}   (median {np.median(values):.4g})")
        print("\n  Pick a threshold inside those ranges and run again.")
        return 1

    r = resp[finite].mean()
    for name, num in (("plain", num_plain), ("metacal", num_mcal)):
        m, err = m_with_error(num[finite], resp[finite], gamma, args.njack)
        _, block_err = m_with_error(num[finite], resp[finite], gamma, args.njack,
                                    shuffle=False)
        naive = naive_error(num[finite], resp[finite], gamma)
        print(f"  ensemble m ({name:7s}) = {m:+.5f} +/- {err:.5f}"
              f"   ({abs(m) / err if err else float('nan'):.1f} sigma)")
        print(f"      sigma: jackknife {err:.5f}   in catalog order "
              f"{block_err:.5f}   i.i.d. {naive:.5f}")
        print(f"      per-object scatter: numerator {num[finite].std():.4f}")
    print(f"  ensemble <R11>        = {r:.4f}")
    inflation = num_mcal[finite].mean() / num_plain[finite].mean()
    print(f"  metacal / plain       = {inflation:.4f}"
          f"   <- the whole bias is here\n")

    _outlier_census(plus, minus, mcal_col, num_plain, num_mcal, resp,
                    finite, gamma, args.njack)

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
              f"{'m_plain':>18s} {'m_metacal':>18s} {'mcal/plain':>11s}")
        order = np.argsort(key[ok])
        idx = np.flatnonzero(ok)[order]
        edges = np.linspace(0, len(idx), args.nbin + 1).astype(int)
        for i in range(args.nbin):
            sel = idx[edges[i]:edges[i + 1]]
            if len(sel) < 50:
                continue
            rr = resp[sel].mean()
            mp, mp_err = m_with_error(num_plain[sel], resp[sel], gamma, args.njack)
            mm, mm_err = m_with_error(num_mcal[sel], resp[sel], gamma, args.njack)
            ratio = num_mcal[sel].mean() / num_plain[sel].mean()
            print(f"  {key[sel].min():9.3f}..{key[sel].max():<10.3f} {len(sel):7d} "
                  f"{rr:7.4f} {mp:+9.4f}+/-{mp_err:<6.4f} "
                  f"{mm:+9.4f}+/-{mm_err:<6.4f} {ratio:11.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
