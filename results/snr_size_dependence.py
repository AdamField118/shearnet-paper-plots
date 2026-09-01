"""Multiplicative bias as a function of galaxy S/N and half-light radius.

Each panel bins the held-out population along one axis and recomputes ``m`` inside
each bin, so the points measure a population dependence of the estimator rather
than a mismatch against the global response. The shaded band is the Stage IV
requirement, |m| <= 1e-3.

Where the numbers come from
---------------------------
The ``BINNED`` HDU of the evaluation FITS is binned by **flux only** (``bin_by`` is
fixed to ``"flux"`` in ShearNet's ``research/shear_bias/run.py`` at the pinned
commit), so it cannot supply the S/N and size panels this figure needs. The
per-object ``TAB_P`` / ``TAB_M`` tables do carry ``s2n`` and ``hlr_th``, so the bins
are formed here and the bias in each bin is computed by **ShearNet's own estimator**,
``shearnet.methods.anacal.paired_bias``, called with ``bin_values``.

That is deliberate: ``m`` is a pair-matched ratio of means with a jackknife error and
a specific ``c`` convention, and a second implementation of it here would be a second
thing that can disagree with the SUMMARY table in the same file. Nothing about the
bias math lives in this module.

Usage
-----
    python snr_size_dependence.py --fits .../evaluation.fits
    python snr_size_dependence.py --fits .../evaluation.fits --nbins 10
    python snr_size_dependence.py --fits .../evaluation.fits --bin-by s2n
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_fits import DISPLAY_NAME, Evaluation  # noqa: E402

#: Stage IV multiplicative-bias requirement.
STAGE_IV_M = 1e-3

#: Per-object columns we can bin along, with axis labels.
BIN_AXES = {
    "s2n": {"label": r"galaxy S/N", "log": True},
    "hlr_th": {"label": r"half-light radius [arcsec]", "log": False},
    "flux_th": {"label": r"flux [counts]", "log": True},
}

COLORS = {"shearnet": "#B40426", "ngmix": "#3B4CC0", "anacal": "#2E7D74"}
MARKERS = {"shearnet": "o", "ngmix": "^", "anacal": "s"}


def _import_shearnet():
    """Import ShearNet's bias estimator, with an actionable error."""
    extra = Path(os.environ.get("SHEARNET_DIR", "")).expanduser()
    if str(extra) and extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
    try:
        from shearnet.methods.anacal import ShapeMeasurement, paired_bias  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import ShearNet, which computes the binned bias.\n"
            "Install it at the commit pinned in the README, or set SHEARNET_DIR.\n"
            f"Underlying error: {exc}"
        ) from exc
    return ShapeMeasurement, paired_bias


def _shape_columns(table: Table, estimator: str):
    """Resolve the (e, response) column pair for one estimator.

    Prefers the metacal-corrected prediction, which is what the corrected SUMMARY
    rows use; falls back to the uncorrected one so a run without metacal still plots.
    """
    cols = set(table.colnames)
    for e_col in (f"e_{estimator}_metacal_corrected",
                  f"e_{estimator}_metacal_raw",
                  f"e_{estimator}_uncorrected"):
        if e_col in cols:
            break
    else:
        raise KeyError(f"no shape column for {estimator!r} in {sorted(cols)[:8]}...")

    r_col = f"Rgamma_{estimator}_metacal"
    if r_col not in cols:
        raise KeyError(f"no response column {r_col}")
    return e_col, r_col


def binned_bias(ev, estimator, bin_by, *, nbins=8, component=0, njac=20):
    """m in bins of ``bin_by``, via ShearNet's own paired_bias."""
    ShapeMeasurement, paired_bias = _import_shearnet()

    plus, minus = ev.table("TAB_P"), ev.table("TAB_M")
    if bin_by not in plus.colnames:
        raise KeyError(
            f"TAB_P has no {bin_by!r} column; available binnable columns: "
            f"{[c for c in BIN_AXES if c in plus.colnames]}"
        )

    e_col, r_col = _shape_columns(plus, estimator)
    flag_col = f"flag_{estimator}"

    def _measure(tab):
        flags = (np.asarray(tab[flag_col], dtype=bool)
                 if flag_col in tab.colnames else None)
        return ShapeMeasurement(
            e=np.asarray(tab[e_col], dtype=float),
            dedg=np.asarray(tab[r_col], dtype=float),
            flags=flags,
        )

    # Bin on the pair mean so a galaxy lands in the same bin in both populations.
    values = 0.5 * (np.asarray(plus[bin_by], dtype=float)
                    + np.asarray(minus[bin_by], dtype=float))

    shear_true = float(ev.header.get("SHEAR_T", 0.01))
    estimate = paired_bias(
        _measure(plus), _measure(minus), shear_true,
        component=component, njac=njac,
        c_convention=str(ev.header.get("C_CONVEN", "lin2026")),
        resample=str(ev.header.get("RESAMPLE", "jackknife")),
        bin_values=values, nbins=nbins,
    )
    if estimate.bins is None:
        raise RuntimeError("paired_bias returned no binned result")
    return estimate


def _panel(ax, ev, estimators, bin_by, *, nbins, component, njac):
    axis = BIN_AXES.get(bin_by, {"label": bin_by, "log": False})
    for estimator in estimators:
        est = binned_bias(ev, estimator, bin_by, nbins=nbins,
                          component=component, njac=njac)
        b = est.bins
        edges = np.asarray(b["edges"], dtype=float)
        centres = 0.5 * (edges[:-1] + edges[1:])
        m = np.asarray(b["m"], dtype=float)
        m_err = np.asarray(b["m_err"], dtype=float)
        n = min(len(centres), len(m))
        ax.errorbar(
            centres[:n], m[:n], yerr=m_err[:n],
            fmt=MARKERS.get(estimator, "o"), ms=7, capsize=3, elinewidth=1.6,
            color=COLORS.get(estimator, "k"), mfc="white",
            mec=COLORS.get(estimator, "k"),
            label=DISPLAY_NAME.get(estimator, estimator),
        )
        print(f"  {estimator:9s} {bin_by:8s} m range "
              f"[{np.nanmin(m[:n]):+.2e}, {np.nanmax(m[:n]):+.2e}]")

    ax.axhspan(-STAGE_IV_M, STAGE_IV_M, color="0.75", alpha=0.45, zorder=0,
               label=r"Stage IV  $|m|\leq10^{-3}$")
    ax.axhline(0.0, color="0.4", lw=0.9, ls=":", zorder=1)
    if axis["log"]:
        ax.set_xscale("log")
    ax.set_xlabel(axis["label"])
    ax.set_ylabel(r"$m$")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fits", required=True, help="ShearNet evaluation FITS")
    p.add_argument("--bin-by", nargs="+", default=["s2n", "hlr_th"],
                   help="per-object columns to bin along, one panel each")
    p.add_argument("--estimators", nargs="+", default=None)
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--njac", type=int, default=20)
    p.add_argument("-o", "--out", default=None,
                   help="output stem. Default ../figures/snr_size_dependence")
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    ev = Evaluation(args.fits)
    print(ev)

    estimators = args.estimators or ev.estimators()
    if not estimators:
        raise SystemExit("no estimators found in SUMMARY")
    print(f"estimators: {estimators}")

    n = len(args.bin_by)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.0), squeeze=False)
    for ax, bin_by in zip(axes[0], args.bin_by):
        _panel(ax, ev, estimators, bin_by,
               nbins=args.nbins, component=args.component, njac=args.njac)
    axes[0][0].legend(frameon=False, fontsize=10)
    fig.tight_layout()

    stem = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "figures" / "snr_size_dependence"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.format:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
