"""ngmix's fitted flux, magnitude and S/N on the fiducial (UT4) catalog.

A sanity check of the simulation's depth, not a paper figure. One record per
galaxy (scene ``zero``, first ring station), every measurable fit, no cut.
Magnitudes use ``mag = zp - 2.5 log10(flux)`` with ``--zp`` (default 30, the
``magzp`` superbit-lensing's MEDS files are written with). They are shown for
the fitted flux as simulated (the 36-exposure sum) and divided by 36 (one
300 s exposure), since which of the two the noise level corresponds to is the
open question.

With ``--real-cat`` (e.g. LITB III's ``mega_annular_catalog_*.fits``) its
``s2n_noshear`` and ``MAG_AUTO`` are drawn on top for comparison.

    python tools/flux_snr_check.py --runs ~/weak_lensing/ShearNet/runs/unit_tests
    python tools/flux_snr_check.py --runs ... --real-cat mega_annular_catalog_20251121_110457.fits
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "results"))
from catalog import Catalog, find_run  # noqa: E402

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
NEXP = 36


def summary(name, x):
    p = np.percentile(x, PERCENTILES)
    print(f"{name:28s} " + "  ".join(f"{v:9.2f}" for v in p))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, help="ShearNet runs/unit_tests directory")
    p.add_argument("--zp", type=float, default=30.0, help="magnitude zero point")
    p.add_argument("--real-cat", default=None, help="optional real-data catalog to overlay")
    p.add_argument("--out", default="flux_snr_check.png")
    args = p.parse_args(argv)

    path = find_run(args.runs, "fourth")
    if path is None:
        raise SystemExit(f"no finished fourth unit test under {args.runs}")
    cat = Catalog(path)
    print(f"{path}: {cat.run_name}, {cat.n_objects} galaxies, noise from its config")
    ng = cat.columns("NGMIX", ["flux_original", "s2n_original", "flags_original",
                               "s2n_noshear", "flags_noshear"], ["zero"])
    truth = cat.columns("TRUTH", ["flux_model"], ["zero"])
    first = lambda a: a[0, 0]  # noqa: E731  scene zero, ring station 0
    ok = (first(ng["flags_original"]) == 0) & (first(ng["flags_noshear"]) == 0)
    flux = first(ng["flux_original"])[ok]
    s2n = first(ng["s2n_original"])[ok]
    s2n_mcal = first(ng["s2n_noshear"])[ok]
    flux_true = first(truth["flux_model"])[ok]
    pos = flux > 0
    mag = args.zp - 2.5 * np.log10(flux[pos])
    mag_1exp = args.zp - 2.5 * np.log10(flux[pos] / NEXP)

    print(f"{ok.sum()} measurable of {cat.n_objects}; {(~pos).sum()} with fitted flux <= 0\n")
    print(f"{'percentile':28s} " + "  ".join(f"{q:8d}%" for q in PERCENTILES))
    summary("true flux (catalog)", flux_true)
    summary("ngmix flux", flux)
    summary(f"ngmix mag (zp {args.zp:g})", mag)
    summary(f"ngmix mag of flux/{NEXP}", mag_1exp)
    summary("ngmix s2n (original)", s2n)
    summary("ngmix s2n (metacal noshear)", s2n_mcal)
    print(f"\nfraction with s2n_noshear < 10: {np.mean(s2n_mcal < 10):.3f}")

    real = None
    if args.real_cat:
        from astropy.table import Table
        t = Table.read(args.real_cat)
        good = np.isfinite(t["s2n_noshear"]) & (t["s2n_noshear"] > 0)
        real = {"s2n": np.asarray(t["s2n_noshear"])[good],
                "mag": np.asarray(t["MAG_AUTO"])[good]}
        print(f"\nreal catalog {Path(args.real_cat).name}: {good.sum()} objects")
        summary("real s2n_noshear", real["s2n"])
        summary("real MAG_AUTO", real["mag"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    bins = np.logspace(np.log10(max(flux[pos].min(), 1)), np.log10(flux.max()), 60)
    axes[0].hist(flux_true, bins=bins, histtype="step", lw=1.5, label="true (catalog)")
    axes[0].hist(flux[pos], bins=bins, histtype="step", lw=1.5, label="ngmix fit")
    axes[0].set(xscale="log", xlabel="flux [ADU, 36-exposure sum]", ylabel="galaxies")
    mbins = np.linspace(15, 30, 61)
    axes[1].hist(mag, bins=mbins, histtype="step", lw=1.5, density=True, label="ngmix, flux as simulated")
    axes[1].hist(mag_1exp, bins=mbins, histtype="step", lw=1.5, density=True,
                 label=f"ngmix, flux / {NEXP}")
    sbins = np.logspace(0, np.log10(max(s2n.max(), 10)), 60)
    axes[2].hist(s2n_mcal, bins=sbins, histtype="step", lw=1.5, density=True,
                 label="ngmix s2n_noshear (sim)")
    axes[2].hist(s2n_mcal / np.sqrt(NEXP), bins=sbins, histtype="step", lw=1.5, density=True,
                 ls="--", label=f"sim s2n / {np.sqrt(NEXP):g}")
    if real is not None:
        axes[1].hist(real["mag"], bins=mbins, histtype="step", lw=1.5, density=True,
                     color="k", label="real MAG_AUTO")
        axes[2].hist(real["s2n"], bins=sbins, histtype="step", lw=1.5, density=True,
                     color="k", label="real s2n_noshear")
    axes[1].set(xlabel=f"magnitude (zp {args.zp:g})", ylabel="density")
    axes[2].axvline(10, color="0.5", ls=":", label="s2n = 10 cut")
    axes[2].set(xscale="log", xlabel="S/N", ylabel="density")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cat.run_name}: ngmix fits, scene 'zero', one record per galaxy, no cut")
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
