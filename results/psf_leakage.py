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

WHICH SHAPE, AND WHY IT IS NOT ONE SHAPE
----------------------------------------
By default each estimator is fitted on the shape its alpha is *quoted* on --
:func:`_reported_shape`, read from ``paper_numbers`` so this figure cannot drift
from ``tab:unit-test-bias``. Those differ by estimator: ShearNet is reported under
``rgamma`` (the raw image, metacal never touching it, divided by its ensemble shear
response) and ngmix under ``noshear_rgamma`` (the reconvolved image divided by
its ensemble shear response, without the PSF-response subtraction). There is no single shape that
makes the figure and the text agree, and taking one for both would silently put a
different number in the caption than in the table.

``--shape`` overrides that with one shape for both, which is the right thing when
the question is "what does this correction do", and the wrong thing when the figure
is going in the paper. ``--compare-shapes`` prints alpha under every shape.

``plot_psf_leakage_comparison`` compares exactly two datasets, so with more than two
estimators present, pass ``--estimators`` to choose the pair.

Usage
-----
    python psf_leakage.py --fits .../evaluation.fits
    python psf_leakage.py --fits .../evaluation.fits --estimators shearnet ngmix
    python psf_leakage.py --fits .../evaluation.fits --compare-shapes
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
from evaluation_fits import DISPLAY_NAME, Evaluation  # noqa: E402
from paper_labels import label_psf_leakage


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


def _reported_shape(estimator: str) -> str:
    """The shape this estimator's alpha is quoted on, from paper_numbers.

    Each estimator is reported on its own shape -- ShearNet on the raw image
    divided by its shear response, ngmix on metacal's noshear image divided by
    its own -- so there is no single shape that makes this figure agree with the
    alpha in the text. Resolving through ``paper_numbers`` rather than picking
    one shape for both is what keeps ``fig:psf_leakage`` and
    ``tab:unit-test-bias`` describing the same measurement.
    """
    from paper_numbers import reported_leakage_shape

    return reported_leakage_shape(estimator)


def panel_fits_for(evaluation, estimator, out_fits, *, nbin=10, min_count=20,
                   njac=30, shape="raw"):
    """Build superbit panel data for one estimator and write its panel FITS.

    Returns the fitted ``(alpha1, alpha1_err, alpha2, alpha2_err)``, which is the
    number Table 6 of the paper reports.
    """
    PSFLeakagePanelMaker, save_all_panels_to_fits, _ = _import_superbit()

    data = evaluation.leakage_inputs(estimator, shape=shape)
    if data["n_dropped"]:
        print(f"  {estimator}: dropped {data['n_dropped']} non-finite rows")

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
        # The response correction is what this figure is measuring, so it must
        # not be applied to the shapes before the slope is fitted.
        correct_psf_leakage=False,
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
    p.add_argument("--fits", required=True, help="ShearNet evaluation FITS")
    p.add_argument(
        "--estimators", nargs=2, default=None, metavar=("A", "B"),
        help="the two estimators to compare (default: the first two present)",
    )
    p.add_argument("--nbin", type=int, default=10, help="e^PSF bins")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--njac", type=int, default=30, help="jackknife resamples")
    p.add_argument(
        "--panel-dir", default=None,
        help="keep the intermediate per-estimator panel FITS here",
    )
    p.add_argument("-o", "--out", default=None,
                   help="output stem. Default ../figures/psf_leakage")
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--shape", default=None,
                   choices=("raw", "raw_rgamma", "noshear", "noshear_rgamma",
                            "noshear_rpsf", "noshear_rgamma_rpsf"),
                   help="ONE shape for both estimators, overriding the "
                        "per-estimator default. Named by the corrections it "
                        "carries: raw = the original image, metacal never "
                        "touched it. noshear = metacal's reconvolved image. "
                        "_rgamma = divided by <R^gamma>. _rpsf = Rbar^PSF "
                        "subtracted. Default is each estimator's own reported "
                        "shape, which is what the tables quote -- use this only "
                        "to put both pipelines on one footing deliberately.")
    p.add_argument("--compare-shapes", action="store_true",
                   help="print alpha for every shape and stop, so the effect of "
                        "each correction is visible rather than argued")
    args = p.parse_args(argv)

    ev = Evaluation(args.fits)

    if args.compare_shapes:
        # tempfile is imported at module scope. Importing it again HERE makes
        # the name local to the whole of main(), so the use on the NORMAL path
        # (the TemporaryDirectory further down) raises UnboundLocalError
        # whenever this branch is not taken -- which is every ordinary run.
        with tempfile.TemporaryDirectory() as scratch:
            for estimator in (args.estimators or ev.leakage_estimators()):
                print(f"\n{estimator}")
                for shape in Evaluation.LEAKAGE_SHAPES:
                    try:
                        data = ev.leakage_inputs(estimator, shape=shape)
                        a1, a1e, a2, a2e = panel_fits_for(
                            ev, estimator,
                            os.path.join(scratch, f"{estimator}_{shape}.fits"),
                            nbin=args.nbin, min_count=args.min_count,
                            njac=args.njac, shape=shape)
                        print(f"  {shape:<20} alpha1 = {a1:+.5f} +/- {a1e:.5f}   "
                              f"alpha2 = {a2:+.5f} +/- {a2e:.5f}\n"
                              f"  {'':<20} [{data['shape_column']}]")
                    except Exception as exc:
                        print(f"  {shape:<20} unavailable: "
                              f"{type(exc).__name__}: {exc}")
        return
    print(ev)

    available = ev.leakage_estimators()
    if not available:
        raise SystemExit(
            "No estimator in the LEAKAGE table has both a shape column and "
            "Rpsf_<est>_metacal; nothing to plot."
        )

    if args.estimators:
        chosen = list(args.estimators)
        missing = [e for e in chosen if e not in available]
        if missing:
            raise SystemExit(
                f"requested {missing} but LEAKAGE only supports {available}"
            )
    else:
        if len(available) < 2:
            raise SystemExit(
                f"the comparison figure needs two estimators, found {available}"
            )
        chosen = available[:2]
    print(f"comparing: {chosen}")

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
            shape = args.shape or _reported_shape(estimator)
            out_fits = panel_dir / f"panels_{estimator}.fits"
            a1, a1e, a2, a2e = panel_fits_for(
                ev, estimator, out_fits,
                nbin=args.nbin, min_count=args.min_count, njac=args.njac,
                shape=shape,
            )
            print(f"  {DISPLAY_NAME.get(estimator, estimator)} [{shape}]: "
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
        label_psf_leakage(fig, shapes=[args.shape or _reported_shape(e) for e in chosen])
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
