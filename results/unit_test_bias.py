"""Figure 4: |m| and |c| across UT1--UT4, on the cut sample, calibrated with R^S.

No categorical UT bias plot exists in LITB-III-plots (sec5/fig19 is a radial
cluster-shear test), superbit_lensing.plotter, or ShearNet's
research/shear_bias/plots_from_fits.ipynb (checked 2026-09-21). Only the bar layout
is new. Every number comes from :func:`shear_stats.shear_bias`.

m is component 1 along the applied shear; c is the orthogonal component 2 on
that same +/-g1 pair (Equation 3). With the default ``--cut metacal`` both
estimators are measured on the sample that passes ngmix's T/Tpsf > 1 and
s2n > 10 on the noshear fit, each population selected on its own, and calibrated
by R = <R^gamma> + R^S (see shear_stats). The JSON written next to the figure
carries m and c without R^S too, and the selected counts.

Missing values are NEVER drawn as zero-height bars. --draft explicitly creates
an empty layout when no results have been supplied.

    python unit_test_bias.py --runs ~/ShearNet/runs/unit_tests --out ../output/unit_test_bias.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from catalog import RUNS
ESTIMATORS = ("shearnet", "ngmix")
from paper_colors import COLORS, HATCHES
NAMES = {"shearnet": "ShearNet", "ngmix": "ngmix"}


def _num(x):
    x = float(np.asarray(x))
    return x if np.isfinite(x) else None


def collect(root, args):
    from catalog import Catalog, find_run
    from shear_stats import G1_PAIR, Sample, cut_from_name, shear_bias

    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    data = {"components": {"m": 1, "c": 2, "applied_shear": 1},
            "cut": cut.describe() if cut else "none", "njack": args.njack, "runs": {}}
    for name in RUNS:
        path = find_run(root, name)
        row = {"estimators": {}}
        if path is not None:
            catalog = Catalog(path)
            header = catalog.header
            row.update(source=str(path.resolve()), bytes=path.stat().st_size,
                       run_name=catalog.run_name, checkpoint_sha256=header.get("CKPTSHA"),
                       evaluated=header.get("DATE"))
            sample = Sample(catalog, list(G1_PAIR), cut=cut, estimators=ESTIMATORS)
            row["counts"] = {s: sample.counts()[s] for s in G1_PAIR}
            for est in ESTIMATORS:
                b = shear_bias(sample, est, njack=args.njack)
                row["estimators"][est] = {
                    "m1": _num(b["m"][0]), "m1_err": _num(b["m"][1]),
                    "c2": _num(b["c"][0]), "c2_err": _num(b["c"][1]),
                    "m1_without_RS": _num(b["m_without_RS"][0]),
                    "m1_without_RS_err": _num(b["m_without_RS"][1]),
                    "R11": _num(b["R_aa"][0]), "RS11": _num(b["RS_aa"][0]),
                    "RS11_err": _num(b["RS_aa"][1]), "n_selected": b["n_selected"]}
            del sample
            catalog.close()
        data["runs"][name] = row
    return data


def report(data) -> str:
    """The numbers as the text quotes them."""
    lines = [f"cut: {data.get('cut')}"]
    for name, label in zip(RUNS, ("UT1", "UT2", "UT3", "UT4")):
        for est in ESTIMATORS:
            v = data["runs"].get(name, {}).get("estimators", {}).get(est)
            if not v or v.get("m1") is None:
                lines.append(f"  {label} {est:8s} pending")
                continue
            lines.append(
                f"  {label} {est:8s} m1 = ({v['m1'] * 1e3:+.2f} +/- {v['m1_err'] * 1e3:.2f})e-3"
                f"   c2 = ({v['c2'] * 1e5:+.2f} +/- {v['c2_err'] * 1e5:.2f})e-5"
                f"   [without R^S: m1 = {v['m1_without_RS'] * 1e3:+.2f}e-3;"
                f" R^S_11 = {v['RS11']:+.4f}; n = {v['n_selected']}]")
    return "\n".join(lines)


def draw(data):
    """Return a figure; no fitting or calibration happens in this function."""
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 11,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.25), layout="constrained")
        for ax, key, symbol, scale in zip(axes, ("m1", "c2"), ("m", "c"), (1e3, 1e5)):
            any_value = False
            for j, est in enumerate(ESTIMATORS):
                for i, run in enumerate(RUNS):
                    value = data.get("runs", {}).get(run, {}).get("estimators", {}).get(est, {})
                    y, err = value.get(key), value.get(key + "_err")
                    x = i + (j - .5) * .36
                    if y is None or not np.isfinite(y):
                        ax.text(x, .12, "…", color=COLORS[est], ha="center",
                                transform=ax.get_xaxis_transform(), fontsize=16)
                        continue
                    any_value = True
                    error = float(err)*scale if err is not None and np.isfinite(err) and err >= 0 else None
                    height = abs(float(y))*scale
                    if error is not None:
                        lo, hi = (float(y)-float(err))*scale, (float(y)+float(err))*scale
                        lower = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
                        upper = max(abs(lo), abs(hi))
                        error = np.array([[height-lower], [upper-height]])
                    ax.bar(x, height, width=.32, color=COLORS[est],
                           alpha=1, hatch=HATCHES[est], yerr=error, capsize=3, linewidth=.6,
                           edgecolor="black", error_kw={"elinewidth": 1})
            ax.set_xticks(range(4), ["UT1", "UT2", "UT3", "UT4"])
            ax.set_xlim(-.6, 3.6)
            ax.set_xlabel("Unit test")
            exponent = -3 if symbol == "m" else -5
            label = rf"|{symbol}|"
            ax.set_ylabel(rf"${label}$ ($10^{{{exponent}}}$)")
            ax.spines[["top", "right"]].set_visible(False)
            if any_value:
                ax.axhline(0, color="0.35", lw=.7, zorder=0)
                ax.margins(y=.12)
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.text(.5, .5, "Measurements pending", ha="center", va="center",
                        color="0.4", transform=ax.transAxes)
        fig.legend(handles=[Patch(facecolor=COLORS[e], edgecolor="black", hatch=HATCHES[e], label=NAMES[e])
                                for e in ESTIMATORS], loc="outside upper center", ncol=2, frameon=False)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", type=Path,
                     help="a directory of <name>.fits, or ShearNet's runs/unit_tests")
    src.add_argument("--values", type=Path, help="replay an exported numeric JSON")
    src.add_argument("--draft", action="store_true", help="explicit empty, pending layout")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--njack", type=int, default=20)
    p.add_argument("--cut", choices=("metacal", "none"), default="metacal")
    p.add_argument("--min-t-ratio", type=float, default=None)
    p.add_argument("--min-s2n", type=float, default=None)
    args = p.parse_args(argv)
    if args.runs:
        if not args.runs.is_dir():
            p.error(f"--runs does not exist: {args.runs}")
        data = collect(args.runs, args)
    elif args.values:
        data = json.loads(args.values.read_text())
    else:
        data = {"components": {"m": 1, "c": 2, "applied_shear": 1}, "runs": {}}
    if data.get("components") != {"m": 1, "c": 2, "applied_shear": 1}:
        p.error("numeric JSON components do not match the paper convention")
    if data.get("runs"):
        print(report(data))
    fig = draw(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(f"wrote {args.out}; m uses component 1, c component 2 on the g1 pair")


if __name__ == "__main__":
    main()
