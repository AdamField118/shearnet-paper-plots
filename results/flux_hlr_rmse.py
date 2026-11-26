"""RMSE of the flux and HLR predictions of ShearNet and ngmix, with error bars.

The stamps are those of flux_hlr_residuals.py (the paper's residual figure):
scene "zero", the original (un-metacalibrated) stamps, both estimators' flags
zero, and by default the paper cut on ngmix's noshear fit. ngmix's HLR is the
circularized half-light radius of its Gaussian fit, as in that figure.

Reported for each estimator, RMSE = sqrt(mean (pred - true)^2) of
  HLR   [arcsec]
  flux  [counts]  (FLUX_AUTO units)
  flux  [nJy]     F_nu = counts x 2.943
  flux  [mag]     m = -2.5 log10(counts) + 30.228, so the residual is
                  -2.5 log10(pred / true) and the zero point cancels; only
                  stamps where both estimators' flux is positive enter it
and the ShearNet - ngmix difference of each. Errors are the standard deviation
over --nboot bootstrap resamples of galaxies: every ring station of a galaxy is
kept together, since they share its noise field and PSF. The difference is
computed on the same resample, so its error accounts for the pairing.

Run from shearnet-paper-plots/results:

    python flux_hlr_rmse.py --runs ~/weak_lensing/ShearNet/runs/unit_tests
    python flux_hlr_rmse.py --runs ~/weak_lensing/ShearNet/runs/unit_tests --cut none
"""
import argparse
from pathlib import Path

import numpy as np

from catalog import Catalog, find_run
from flux_hlr_residuals import gaussian_hlr
from shear_stats import cut_from_name, ngmix_passes

NJY_PER_COUNT = 2.943
MAG_ZEROPOINT = 30.228  # cancels in a magnitude residual


def residuals_by_object(cat, cut, chunk=4096):
    """``(object, sn - true, ng - true, true)``; the selection of flux_hlr_residuals.residuals."""
    parts = []
    for start in range(0, cat.n_objects, chunk):
        rows = slice(start, min(start + chunk, cat.n_objects))

        def read(table, names):
            return cat.columns(table, names, ["zero"], objects=rows)

        truth = read("TRUTH", ["flux_model", "hlr"])
        sn = read("SHEARNET", ["flags_original", "flux_original", "hlr_original"])
        columns = ["flags_original", "flux_original", "T_original", "g_original"]
        if cut is not None:
            columns += ["flags_noshear", "T_noshear", "s2n_noshear"]
        ng = read("NGMIX", columns)
        ok = (sn["flags_original"] == 0) & (ng["flags_original"] == 0)
        if cut is not None:
            ok &= ngmix_passes(ng, "noshear", cut, read("STAMP", ["psf_T_admom"])["psf_T_admom"])
        target = np.stack((truth["flux_model"], truth["hlr"]), axis=-1)
        pred_sn = np.stack((sn["flux_original"], sn["hlr_original"]), axis=-1)
        pred_ng = np.stack((ng["flux_original"], gaussian_hlr(ng["T_original"], ng["g_original"])),
                           axis=-1)
        ok &= np.all((target > 0) & np.isfinite(target) & np.isfinite(pred_sn)
                     & np.isfinite(pred_ng), axis=-1)
        obj = np.broadcast_to(np.arange(rows.start, rows.stop), ok.shape)
        if ok.any():
            t = target[ok]
            parts.append((obj[ok], pred_sn[ok] - t, pred_ng[ok] - t, t))
    if not parts:
        raise SystemExit("no matched zero-shear stamps survive the selection")
    return tuple(np.concatenate(group) for group in zip(*parts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fits", type=Path, help="merged fiducial UT4 catalog")
    source.add_argument("--runs", type=Path, help="find the completed fourth run here")
    p.add_argument("--run", default="fourth", help="unit test under --runs (default fourth)")
    p.add_argument("--cut", choices=("metacal", "none"), default="metacal")
    p.add_argument("--min-t-ratio", type=float)
    p.add_argument("--min-s2n", type=float)
    p.add_argument("--nboot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    path = args.fits or find_run(args.runs, args.run)
    if path is None:
        p.error(f"no completed {args.run} catalog under {args.runs}")
    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    with Catalog(path) as cat:
        obj, r_sn, r_ng, truth = residuals_by_object(cat, cut)

    galaxies, inverse = np.unique(obj, return_inverse=True)
    flux_true = truth[:, 0]
    positive = (r_sn[:, 0] + flux_true > 0) & (r_ng[:, 0] + flux_true > 0)

    def mag(true, pred):
        return (-2.5 * np.log10(pred) + MAG_ZEROPOINT) - (-2.5 * np.log10(true) + MAG_ZEROPOINT)

    # (label, format, stamps used, residual per estimator on those stamps)
    metrics = [("HLR   [arcsec]", ".4g", slice(None), {e: r[:, 1] for e, r in
                                                       (("ShearNet", r_sn), ("ngmix", r_ng))}),
               ("flux  [counts]", ".4g", slice(None), {e: r[:, 0] for e, r in
                                                       (("ShearNet", r_sn), ("ngmix", r_ng))}),
               ("flux  [nJy]", ".4g", slice(None), {e: NJY_PER_COUNT * r[:, 0] for e, r in
                                                    (("ShearNet", r_sn), ("ngmix", r_ng))}),
               ("flux  [mag]", ".4f", positive, {e: mag(flux_true[positive],
                                                        (r[:, 0] + flux_true)[positive])
                                                 for e, r in (("ShearNet", r_sn), ("ngmix", r_ng))})]
    # per-galaxy sums, so a bootstrap of galaxies is a weighted sum over them
    sums = []
    for _, _, used, residual in metrics:
        idx = inverse[used]
        sums.append((np.bincount(idx, minlength=len(galaxies)).astype(float),
                     {e: np.bincount(idx, weights=x ** 2, minlength=len(galaxies))
                      for e, x in residual.items()}))

    def rmse(weights):
        return [{e: np.sqrt(weights @ s / (weights @ n)) for e, s in sq.items()}
                for n, sq in sums]

    point = rmse(np.ones(len(galaxies)))
    rng = np.random.default_rng(args.seed)
    boot = [rmse(rng.multinomial(len(galaxies), np.full(len(galaxies), 1 / len(galaxies)))
                 .astype(float)) for _ in range(args.nboot)]

    print(f"{path}")
    print(f"cut: {cut.describe() if cut else 'none'}")
    print(f"{len(obj)} stamps (scene zero, all ring stations) of {len(galaxies)} galaxies; "
          f"errors: 1 sigma, {args.nboot} bootstrap resamples of galaxies")
    print(f"magnitudes: m = -2.5 log10(counts) + {MAG_ZEROPOINT}; {int((~positive).sum())} "
          f"stamps with a non-positive predicted flux (either estimator) left out of them\n")
    print(f"{'RMSE':16s} {'ShearNet':>22s} {'ngmix':>22s} {'ShearNet - ngmix':>24s}")
    for i, (label, fmt, _, _) in enumerate(metrics):
        sn, ng = point[i]["ShearNet"], point[i]["ngmix"]
        esn, eng = (np.std([b[i][e] for b in boot], ddof=1) for e in ("ShearNet", "ngmix"))
        ediff = np.std([b[i]["ShearNet"] - b[i]["ngmix"] for b in boot], ddof=1)
        print(f"{label:16s} {f'{sn:{fmt}} +/- {esn:.2g}':>22s} {f'{ng:{fmt}} +/- {eng:.2g}':>22s} "
              f"{f'{sn - ng:+{fmt}} +/- {ediff:.2g}':>24s}")


if __name__ == "__main__":
    main()
