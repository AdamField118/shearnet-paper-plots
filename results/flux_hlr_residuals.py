"""Flux and circularized half-light-radius residuals on the original UT4 stamps.

The signed histograms follow ShearNet's notebooks/comparison.ipynb (1bea72b).
Compare the same selected zero-shear stamps in both panels, before metacalibration.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from catalog import Catalog, find_run
from paper_colors import COLORS
from shear_stats import PAPER_CUT, cut_from_name, ngmix_passes


def gaussian_hlr(T, g):
    """Area-equivalent Gaussian HLR from its trace and reduced ellipticity."""
    norm2 = np.sum(g * g, axis=-1)
    valid = np.isfinite(norm2) & (norm2 < 1) & np.isfinite(T) & (T > 0)
    radius = np.full(T.shape, np.nan)
    # T = a^2 + b^2; (1-|g|^2)/(1+|g|^2) = 2ab/T.
    radius[valid] = np.sqrt(np.log(2) * T[valid] * (1 - norm2[valid]) / (1 + norm2[valid]))
    return radius


def residuals(cat, cut=PAPER_CUT, chunk=4096):
    """Return matched (stamp, quantity) residuals and truth, keeping scalar columns only."""
    cfg = yaml.safe_load(cat.key_values("CONFIG")["evaluation"])
    model = cfg["evaluation"]["ngmix"]["gal_model"]
    if model != "gauss":
        raise ValueError(f"expected a Gaussian ngmix galaxy fit for the radius conversion; got {model!r}")
    parts = []
    for start in range(0, cat.n_objects, chunk):
        rows = slice(start, start + chunk)

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
            psf_T = read("STAMP", ["psf_T_admom"])["psf_T_admom"]
            ok &= ngmix_passes(ng, "noshear", cut, psf_T)
        target = np.stack((truth["flux_model"], truth["hlr"]), axis=-1)
        pred_sn = np.stack((sn["flux_original"], sn["hlr_original"]), axis=-1)
        pred_ng = np.stack((ng["flux_original"], gaussian_hlr(ng["T_original"], ng["g_original"])),
                           axis=-1)
        ok &= np.all((target > 0) & np.isfinite(target) & np.isfinite(pred_sn)
                     & np.isfinite(pred_ng), axis=-1)
        if ok.any():
            target = target[ok]
            parts.append((pred_sn[ok] - target, pred_ng[ok] - target, target))
    if not parts:
        raise ValueError("no matched zero-shear stamps survive the requested selection")
    return tuple(np.concatenate(group) for group in zip(*parts))


def plot(sn, ng, truth):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    fig.subplots_adjust(left=.105, right=.99, top=.80, bottom=.34, wspace=.31)
    axes[0].set_ylabel("Fraction of stamps per bin", fontsize=10)
    for i, (ax, title, xlabel) in enumerate(zip(
        axes, ("Total flux", "Circularized half-light radius"),
        (r"$\widehat{F}-F$ [counts]", r"$\widehat{r}_{1/2}-r_{1/2}$ [arcsec]"),
    )):
        lo, hi = np.percentile(np.r_[sn[:, i], ng[:, i]], [1, 99])
        lo, hi = min(lo, 0.), max(hi, 0.)
        padding = .025 * (hi - lo) if hi > lo else np.finfo(float).eps
        edges = np.linspace(lo - padding, hi + padding, 51)
        ax.axvline(0, color=".5", ls=":", lw=1, zorder=0)
        for values, est, label, ls in ((sn, "shearnet", "ShearNet", "-"),
                                       (ng, "ngmix", "ngmix", "--")):
            counts = np.histogram(values[:, i], bins=edges)[0]
            ax.stairs(counts / len(values), edges, label=label, color=COLORS[est],
                      linestyle=ls, linewidth=1.65)
        ax.set(xlabel=xlabel, title=title, xlim=(edges[0], edges[-1]), ylim=(0, None))
        med = [100 * np.median(np.abs(r[:, i]) / truth[:, i]) for r in (sn, ng)]
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, .045,
                 f"Median absolute fractional error\nShearNet {med[0]:.3g}%    ngmix {med[1]:.3g}%",
                 ha="center", va="bottom", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.5, 1.01))
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fits", type=Path, help="merged fiducial UT4 catalog")
    source.add_argument("--runs", type=Path, help="find the completed fourth run here")
    p.add_argument("--cut", choices=("metacal", "none"), default="metacal")
    p.add_argument("--min-t-ratio", type=float)
    p.add_argument("--min-s2n", type=float)
    p.add_argument("--out", type=Path, default=Path("output/paper/flux_hlr_residuals.pdf"))
    args = p.parse_args(argv)
    path = args.fits or find_run(args.runs, "fourth")
    if path is None:
        p.error(f"no completed fourth catalog under {args.runs}")
    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    with Catalog(path) as cat:
        sn, ng, truth = residuals(cat, cut)
    fig = plot(sn, ng, truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"{path}: {len(truth)} paired zero-shear stamps")
    for name, r in (("ShearNet", sn), ("ngmix", ng)):
        flux, hlr = 100 * np.median(np.abs(r) / truth, axis=0)
        print(f"{name}: median absolute fractional error: flux {flux:.4g}%, HLR {hlr:.4g}%")
    # Both journal versions quote this same measurement; changing the sample
    # changes the sentence as well as the figure.
    fs, fn = (100 * np.median(np.abs(r[:, 0]) / truth[:, 0]) for r in (sn, ng))
    sentence = (rf"The median absolute fractional flux error is {fs:.3g}\% for "
                rf"\textsc{{ShearNet}} and {fn:.3g}\% for \textsc{{ngmix}}.")
    args.out.with_suffix(".tex").write_text(sentence + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
