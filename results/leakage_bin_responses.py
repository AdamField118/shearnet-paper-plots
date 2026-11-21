"""R^gamma and R^PSF in the PSF-leakage figure's bins.

The same objects as Figure 5 (scene ``zero``, ring-averaged, kept only if every
ring station passes the cut) binned the same way (``--nbin`` percentile bins of
e^PSF_i), and in each bin the median of each estimator's diagonal R^gamma_ii and
R^PSF_ii with its standard error -- the median is also what upstream's binwise
R^PSF correction uses. Component 1 is binned in e1^PSF, component 2 in e2^PSF.
The responses are as measured (not divided by R).

It also prints, per estimator, mean R^PSF_ii / R_ii over these objects: the
leakage slope alpha_i that R^PSF predicts for the shape *without* the R^PSF
correction (compare ``psf_leakage.py --rpsf off``). For an estimator whose
R^PSF describes its leakage the two agree.

    python leakage_bin_responses.py --fits ../evaluations/fourth.fits --out ../output/leakage_bin_responses
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from catalog import Catalog
from paper_colors import COLORS
from shear_stats import (LEAKAGE_VARIANT, Sample, calibrated_leakage_inputs,
                         cut_from_name, leakage_inputs)

ESTIMATORS = ("shearnet", "ngmix")
NAMES = {"shearnet": "ShearNet", "ngmix": "ngmix"}
STYLE = {"shearnet": dict(marker="o", ls="-"), "ngmix": dict(marker="^", ls="--")}
ROWS = (("gamma", r"$R^{\gamma}_{%d%d}$"), ("psf", r"$R^{\rm PSF}_{%d%d}$"))


def binned(x, y, nbin, min_count):
    """Percentile bins of x: (median x, median y, standard error of y, count)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.percentile(x, np.linspace(0, 100, nbin + 1))
    rows = []
    for i in range(nbin):
        upper = x <= edges[i + 1] if i == nbin - 1 else x < edges[i + 1]
        m = (x >= edges[i]) & upper
        n = int(m.sum())
        if n >= min_count:
            rows.append((np.median(x[m]), np.median(y[m]), np.std(y[m], ddof=1) / np.sqrt(n), n))
    return np.array(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fits", required=True, type=Path)
    p.add_argument("--cut", choices=("metacal", "none"), default="metacal")
    p.add_argument("--min-t-ratio", type=float, default=None)
    p.add_argument("--min-s2n", type=float, default=None)
    p.add_argument("--nbin", type=int, default=10)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    args = p.parse_args(argv)

    catalog = Catalog(args.fits)
    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    sample = Sample(catalog, ["zero"], cut=cut, estimators=ESTIMATORS)
    lines = [f"{catalog.run_name}; cut: {cut.describe() if cut else 'none'}",
             "per e^PSF_i bin: median e^PSF_i, median R_ii (+/- standard error), count"]

    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 10,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex="col", layout="constrained")
        for est in ESTIMATORS:
            data = leakage_inputs(sample, est, scene="zero", variant=LEAKAGE_VARIANT[est])
            R = calibrated_leakage_inputs(sample, est, scene="zero")["R"]
            for i in (1, 2):
                rpsf = data[f"r{i}{i}_psf"]
                lines.append(f"\n{NAMES[est]} component {i}: mean R^PSF/R = "
                             f"{np.mean(rpsf) / R[i - 1]:+.4f} (the alpha_{i} R^PSF predicts "
                             f"without its correction; R_{i}{i} = {R[i - 1]:.4f}), "
                             f"{data['n_objects']} objects")
                for row, (kind, label) in enumerate(ROWS):
                    b = binned(data[f"e{i}_psf"], data[f"r{i}{i}_{kind}"], args.nbin, args.min_count)
                    ax = axes[row, i - 1]
                    ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], color=COLORS[est], mfc="white",
                                mec=COLORS[est], ms=6, lw=1.5, capsize=3, elinewidth=1.2,
                                label=NAMES[est], **STYLE[est])
                    ax.set_ylabel(label % (i, i))
                    lines.append(f"  R^{kind}_{i}{i}: " + "; ".join(
                        f"{x:+.3f}: {y:+.4f}+/-{e:.4f} ({int(n)})" for x, y, e, n in b))
        for i in (1, 2):
            axes[0, i - 1].axhline(1, color="0.35", lw=0.7, ls=":", zorder=0)
            axes[1, i - 1].axhline(0, color="0.35", lw=0.7, zorder=0)
            axes[1, i - 1].set_xlabel(rf"$e_{i}^{{\rm PSF}}$ (bin median)")
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside upper center", ncol=2, frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{fmt}"), dpi=200,
                    metadata={"CreationDate": None, "ModDate": None} if fmt == "pdf" else None)
    plt.close(fig)
    text = "\n".join(lines)
    args.out.with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out}.pdf/.png/.txt")


if __name__ == "__main__":
    main()
