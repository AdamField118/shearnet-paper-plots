"""Figure 4: |m| and |c| across UT1--UT4, using the paper table's exact statistics.

No categorical UT bias plot exists in LITB-III-plots (sec5/fig19 is a radial
cluster-shear test), superbit_lensing.plotter, or ShearNet's
research/shear_bias/plots_from_fits.ipynb (checked 2026-09-21). Only the bar layout
is new. Numerical estimates remain in paper_numbers.run_numbers.

m is component 1 along the applied shear; c is the orthogonal component 2 on
that same +/-g1 pair. These definitions belong in the manuscript caption.
Missing values are NEVER drawn as zero-height bars. --draft explicitly creates
an empty layout when no selected-catalog results have been supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

RUNS = ("first", "second", "third", "fourth")
ESTIMATORS = ("shearnet", "ngmix")
from paper_colors import COLORS, HATCHES
NAMES = {"shearnet": "ShearNet", "ngmix": "ngmix"}


def collect(root, args):
    from evaluation_fits import Evaluation
    from paper_tables import find_run, _mask_for, _corrections_from
    from paper_numbers import run_numbers

    chosen = _corrections_from(args)
    data = {"components": {"m": 1, "c": 2, "applied_shear": 1},
            "cut": args.cut, "corrections": chosen, "runs": {}}
    for name in RUNS:
        path = find_run(root, name)
        row = {"estimators": {}}
        if path is not None:
            ev = Evaluation(path)
            row.update(source=str(path.resolve()),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            mask = _mask_for(ev, args.cut, args)
            for est in ESTIMATORS:
                numbers = run_numbers(ev, est, njack=args.njack, mask=mask,
                                      correction=chosen.get(est))
                row["estimators"][est] = {
                    key: (float(numbers[key]) if numbers.get(key) is not None
                          and np.isfinite(numbers[key]) else None)
                    for key in ("m1", "m1_err", "c2", "c2_err")}
                row["estimators"][est]["problems"] = numbers["problems"]
        data["runs"][name] = row
    return data


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
    src.add_argument("--runs", type=Path)
    src.add_argument("--values", type=Path, help="replay an exported numeric JSON")
    src.add_argument("--draft", action="store_true", help="explicit empty, pending layout")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--njack", type=int, default=20)
    p.add_argument("--cut", choices=("none", "superbit", "truth", "both"), default="none")
    p.add_argument("--min-hlr", type=float, default=None)
    p.add_argument("--min-resolution", type=float, default=None)
    p.add_argument("--psf-fwhm", type=float, default=.5)
    p.add_argument("--correction", action="append", metavar="EST=CORR")
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
    fig = draw(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(f"wrote {args.out}; m uses component 1, c component 2 on the g1 pair")


if __name__ == "__main__":
    main()
