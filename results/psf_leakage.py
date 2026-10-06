"""PSF leakage figure: median centered shape against PSF ellipticity.

Each panel plots the median centered shape against the corresponding PSF ellipticity,
binned in e^PSF. Labels distinguish ellipticity from response-corrected shear units.
The global mean is subtracted before binning. The fitted
slope is the leakage coefficient alpha; an ideal estimator is flat.

Per the repository house rule, the binning, the jackknife alpha/beta fit and the
drawing are all done by ``superbit_lensing``:

    superbit_lensing.plotter.PSFLeakagePanelMaker   binning + jackknife fit
    superbit_lensing.plotter.save_all_panels_to_fits  panel-data FITS
    superbit_lensing.plotter.plot_psf_leakage_comparison  the figure

This module only pulls the right columns out of the ShearNet evaluation FITS and
hands them over. Every shape is ring-averaged, because the ring average cancels
intrinsic ellipticity while leaving the leakage signal.

THE SHAPE AND THE SAMPLE
------------------------
By default (``--shape calibrated``) each estimator is shown on the shape its m
and c are calibrated from, ring-averaged over the four stations of the
**unsheared** population (scene ``zero``):

* ngmix: ``g_noshear`` (metacal), **R^PSF-corrected** and divided by its
  ``R = <R^gamma> + R^S``. The R^PSF correction is ``superbit_lensing``'s own
  (``PSFLeakagePanelMaker(correct_psf_leakage=True)``: ``R^PSF`` per
  ``e^PSF`` percentile bin, LITB III Appendix D);
* ShearNet: ``g_original`` divided by its ``R = <R^gamma> + R^S``, **never**
  R^PSF-corrected.

``--shape raw`` restores the previous figure: both on ``g_original``, no
response division, no R^PSF.

With the default ``--cut metacal`` the sample is the paper's cut (ngmix
T/Tpsf > 1 and s2n > 10 on the noshear fit, see :mod:`shear_stats`), and an
object is kept only if **every** ring station passes, so the ring average still
cancels intrinsic shape. The same objects are used for both estimators.
``--cut none`` uses every object.

Usage
-----
    python psf_leakage.py --fits ../evaluations/fourth.fits
    python psf_leakage.py --fits ../evaluations/fourth.fits --shape raw
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import DISPLAY_NAME, Catalog  # noqa: E402
from paper_labels import label_psf_leakage  # noqa: E402
from shear_stats import (Sample, calibrated_leakage_inputs, cut_from_name,  # noqa: E402
                         leakage_inputs)

#: --shape choices. calibrated: each estimator's own m/c pipeline (see above).
SHAPES = ("calibrated", "raw")

#: The shape names paper_labels uses to label the y axis.
LABEL_SHAPE = {"calibrated": "rgamma", "raw": "raw"}


def _import_superbit():
    """Import the SuperBIT leakage helpers, with an actionable error."""
    extra = Path(os.environ.get("SUPERBIT_LENSING_DIR", "")).expanduser()
    if str(extra) and extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
    try:
        from superbit_lensing.plotter import (  # noqa: WPS433
            PSFLeakagePanelMaker,
            plot_psf_leakage_comparison,
            save_all_panels_to_fits,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import superbit_lensing, which draws this figure.\n"
            "Install it at the commit pinned in the README, or set "
            "SUPERBIT_LENSING_DIR to a checkout.\n"
            f"Underlying error: {exc}"
        ) from exc
    return PSFLeakagePanelMaker, save_all_panels_to_fits, plot_psf_leakage_comparison


def panel_fits_for(data, out_fits, *, nbin=10, min_count=20, njac=30,
                   correct_psf_leakage=False):
    """Build superbit panel data from :func:`shear_stats.leakage_inputs` and
    write its panel FITS.

    Returns the fitted ``(alpha1, alpha1_err, alpha2, alpha2_err)``, the slopes
    the paper quotes.
    """
    PSFLeakagePanelMaker, save_all_panels_to_fits, _ = _import_superbit()

    maker = PSFLeakagePanelMaker(
        e1_gal=data["e1_gal"],
        e2_gal=data["e2_gal"],
        e1_psf=data["e1_psf"],
        e2_psf=data["e2_psf"],
        r11_psf=data["r11_psf"],
        r22_psf=data["r22_psf"],
        NBIN=nbin,
        MIN_COUNT=min_count,
        njac=njac,
        # upstream's R^PSF correction, for ngmix only (calibrated_leakage_inputs
        # says which); ShearNet is never R^PSF-corrected
        correct_psf_leakage=correct_psf_leakage,
    )

    panels = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, x, xlab) in zip(
        axes,
        (("E1PSF", data["e1_psf"], r"$e_1^{\rm PSF}$"),
         ("E2PSF", data["e2_psf"], r"$e_2^{\rm PSF}$")),
    ):
        panel = maker.make_panel(ax, x_psf=x, xlab=xlab, return_data=True)
        panels.append((label, panel))
    plt.close(fig)          # the scratch axes exist only to drive make_panel

    Path(out_fits).parent.mkdir(parents=True, exist_ok=True)
    save_all_panels_to_fits(str(out_fits), panels, overwrite=True)

    # Each panel fits BOTH shape components against its own PSF component, so a
    # panel carries one diagonal term and one cross term. The leakage
    # coefficients are the diagonals: alpha_1 from (e1 vs e1_PSF) in the E1PSF
    # panel, alpha_2 from (e2 vs e2_PSF) in the E2PSF panel. Reading both alphas
    # off a single panel would report a cross term as a leakage coefficient.
    # This is also what plot_psf_leakage_comparison draws (it pairs alpha_idx 1
    # with E1PSF and 2 with E2PSF).
    by_label = dict(panels)
    return (by_label["E1PSF"]["alpha_full_1"], by_label["E1PSF"]["alpha_err_1"],
            by_label["E2PSF"]["alpha_full_2"], by_label["E2PSF"]["alpha_err_2"])


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fits", required=True, help="shearnet-eval catalog")
    p.add_argument(
        "--estimators", nargs=2, default=["shearnet", "ngmix"], metavar=("A", "B"),
        help="the two estimators to compare",
    )
    p.add_argument("--nbin", type=int, default=10, help="e^PSF bins")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--njac", type=int, default=30, help="jackknife resamples")
    p.add_argument("--cut", choices=("metacal", "none"), default="metacal")
    p.add_argument("--min-t-ratio", type=float, default=None)
    p.add_argument("--min-s2n", type=float, default=None)
    p.add_argument("--shape", choices=SHAPES, default="calibrated",
                   help="calibrated (default): ngmix noshear - R^PSF e^PSF over R, "
                        "ShearNet original over R. raw: both g_original, uncorrected.")
    p.add_argument(
        "--panel-dir", default=None,
        help="keep the intermediate per-estimator panel FITS here",
    )
    p.add_argument("-o", "--out", default=None,
                   help="output stem. Default ../figures/psf_leakage")
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    catalog = Catalog(args.fits)
    print(catalog)
    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    print(f"cut: {cut.describe() if cut else 'none'}")
    sample = Sample(catalog, ["zero"], cut=cut, estimators=args.estimators)
    chosen = list(args.estimators)

    _, _, plot_psf_leakage_comparison = _import_superbit()

    tmp = None
    panel_dir = Path(args.panel_dir) if args.panel_dir else None
    if panel_dir is None:
        tmp = tempfile.TemporaryDirectory()
        panel_dir = Path(tmp.name)
    panel_dir.mkdir(parents=True, exist_ok=True)

    try:
        panel_files = []
        for estimator in chosen:
            if args.shape == "calibrated":
                data = calibrated_leakage_inputs(sample, estimator, scene="zero")
            else:
                data = leakage_inputs(sample, estimator, scene="zero", variant="original")
                data.update(correct_psf_leakage=False, shape="original, uncorrected")
            out_fits = panel_dir / f"panels_{estimator}.fits"
            a1, a1e, a2, a2e = panel_fits_for(
                data, out_fits, nbin=args.nbin, min_count=args.min_count, njac=args.njac,
                correct_psf_leakage=data["correct_psf_leakage"])
            r_note = (f", R = ({data['R'][0]:.4f}, {data['R'][1]:.4f})"
                      if "R" in data else "")
            print(f"  {DISPLAY_NAME.get(estimator, estimator)} [{data['shape']}"
                  f"{', R^PSF-corrected' if data['correct_psf_leakage'] else ''}{r_note}, "
                  f"{data['n_objects']} of {data['n_total']} objects]: "
                  f"alpha1 = {a1:+.4f} +/- {a1e:.4f}, "
                  f"alpha2 = {a2:+.4f} +/- {a2e:.4f}")
            panel_files.append(out_fits)

        stem = Path(args.out) if args.out else (
            Path(__file__).resolve().parent.parent / "figures" / "psf_leakage"
        )
        stem.parent.mkdir(parents=True, exist_ok=True)

        plot_psf_leakage_comparison(
            str(panel_files[0]), str(panel_files[1]),
            label_nfw=DISPLAY_NAME.get(chosen[0], chosen[0]),
            label_nonfw=DISPLAY_NAME.get(chosen[1], chosen[1]),
            components=(1, 2),
            save_path=None,
        )
        fig = plt.gcf()
        from paper_colors import COLORS, recolor_artists, style_leakage_lines
        recolor_artists(fig, {"magenta": COLORS[chosen[0]], "teal": COLORS[chosen[1]]})
        style_leakage_lines(fig)
        label_psf_leakage(fig, shapes=[LABEL_SHAPE[args.shape]] * len(chosen))
        for fmt in args.format:
            path = stem.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
