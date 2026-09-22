"""PSF illustration: observed PSFEx moment maps across the detector.

The paper default keeps only the observed row of the upstream plot. Use
--rows diagnostic for the former observed/model/residual comparison.

Reproduces Figure C4 of the SuperBIT weak-lensing paper (Saha et al. 2026) for the
PSF that ShearNet is actually trained on: a 3x3 grid of (e1, e2, T) columns against
(observed PSFEx / EM5 model / residual) rows, laid out over CCD coordinates.

**All of the science and all of the drawing is done by ``superbit_lensing``**, not
here. This module only:

  1. resolves which PSFEx file to use (from a ShearNet config, or an explicit path),
  2. caches the expensive moment-map computation to an ``.npz``, and
  3. applies the manuscript PSF labels and writes the figure out.

The two calls that matter are
:func:`superbit_lensing.em5.compute_em5_psfex_maps` and
:func:`superbit_lensing.em5.plot_em5_psfex_maps`. Deliberately *not* reimplemented:
the grid sampling, the adaptive-moment and EM5 fits, the colour maps, the shared
colour limits, and the panel layout all come from the SuperBIT pipeline, so this
figure and the published one cannot drift apart.

One consequence worth knowing: ``plot_em5_psfex_maps`` hard-codes its colour limits
(|e| <= 0.06, residual |de| <= 0.015, |dT| <= 0.005). Those are tuned to the SuperBIT
survey PSF. A PSF with stronger ellipticity will saturate them. That is upstream's
call, and this module does not override it -- if it needs to change, change it in
``superbit_lensing`` so both papers move together.

Usage
-----
    # from a ShearNet config (reads paths.psfex_model_file)
    python psf_properties.py --config ~/ShearNet/research/unit_tests/fourth/config.yaml

    # or point straight at a PSFEx model
    python psf_properties.py --psf /path/to/model.psf

    # a DIRECTORY of models is fine too -- the unit-test configs use one, because
    # training draws a different one of the 50 SuperBIT models per object. One is
    # chosen (sorted order, --psf-index to change it) and its name is printed,
    # because the figure shows a representative and the caption must say which.
    python psf_properties.py --psf ../psf_data/emp_psfs_best/psfex-output

    # the moment maps are slow; cache and reuse them
    python psf_properties.py --psf model.psf --cache maps/psf_maps.npz
    python psf_properties.py --cache maps/psf_maps.npz          # replot, no recompute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from paper_labels import label_psf_properties  # noqa: E402


def _import_superbit():
    """Import the SuperBIT helpers, with an actionable message when missing.

    ``superbit-lensing`` is not on PyPI, so it is either pip-installed from a
    checkout or made importable via ``SUPERBIT_LENSING_DIR``. See the README for
    the pinned commit.
    """
    extra = Path(
        __import__("os").environ.get("SUPERBIT_LENSING_DIR", "")
    ).expanduser()
    if str(extra) and extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

    try:
        from superbit_lensing.em5 import (  # noqa: WPS433
            EM5PsfexMaps,
            compute_em5_psfex_maps,
            plot_em5_psfex_maps,
        )
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "Could not import superbit_lensing, which produces this figure.\n"
            "Install it from a checkout at the commit pinned in the README:\n"
            "    git clone https://github.com/superbit-collaboration/superbit-lensing.git\n"
            "    cd superbit-lensing && git checkout <pinned commit> && pip install -e .\n"
            "or point SUPERBIT_LENSING_DIR at the checkout.\n"
            f"Underlying error: {exc}"
        ) from exc

    return EM5PsfexMaps, compute_em5_psfex_maps, plot_em5_psfex_maps


def psfex_from_config(config_path: Path) -> str:
    """Read ``paths.psfex_model_file`` out of a ShearNet config.

    Using the config rather than a hard-coded path keeps this figure tied to the
    same PSF the network was trained against.
    """
    import yaml

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    try:
        psf = cfg["paths"]["psfex_model_file"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"{config_path} has no paths.psfex_model_file entry; pass --psf instead."
        ) from exc

    if not psf:
        raise ValueError(f"{config_path}: paths.psfex_model_file is empty.")
    return psf


def resolve_psf_model(psf_file, index=0):
    """One PSFEx ``.psf`` file, given either a file or a directory of them.

    ``paths.psfex_model_file`` is a file OR a directory: ShearNet accepts both
    (``shearnet.core.dataset.generate_dataset``), and the unit-test configs use a
    directory, drawing a different one of the 50 SuperBIT models per object.
    ``compute_em5_psfex_maps`` maps a single model, so one has to be chosen, and
    the figure is a REPRESENTATIVE of the training set rather than "the" PSF --
    the caption has to say which file it is for that to mean anything, so the
    chosen path is printed and returned.

    The sort matters and is not cosmetic. It matches
    ``shearnet.core.dataset.search_psf_files``, which sorts because the list
    index is part of the seeded per-object draw; bare ``glob`` returns filesystem
    order (emp47, emp48, emp26, emp11, ... on this set), so an unsorted pick
    would name a different model on a different machine.
    """
    psf_path = Path(psf_file).expanduser()
    if psf_path.is_file():
        return psf_path

    if psf_path.is_dir():
        models = sorted(psf_path.glob("*.psf"))
        if not models:
            raise FileNotFoundError(
                f"{psf_path} is a directory with no .psf files in it"
            )
        if not 0 <= index < len(models):
            raise IndexError(
                f"--psf-index {index} is out of range: {psf_path} holds "
                f"{len(models)} models (0 to {len(models) - 1})"
            )
        chosen = models[index]
        print(f"{psf_path} holds {len(models)} PSFEx models; using index "
              f"{index} of {len(models) - 1}: {chosen.name}")
        print(f"  -> name this file in the figure caption. Other models are "
              f"reachable with --psf-index, or --psf for a specific path.")
        return chosen

    raise FileNotFoundError(
        f"PSFEx model not found: {psf_path}\n"
        "This takes a .psf file or a directory containing them."
    )


def build_maps(psf_file, cache=None, recompute=False, psf_index=0, **kwargs):
    """Return EM5/PSFEx moment maps, computing them only when necessary.

    The computation fits adaptive moments and an EM5 mixture at every grid point,
    which takes minutes at the default step; the cache makes replotting instant.
    """
    EM5PsfexMaps, compute_em5_psfex_maps, _ = _import_superbit()

    cache = Path(cache) if cache else None
    if cache and cache.is_file() and not recompute:
        print(f"loading cached maps from {cache}")
        return EM5PsfexMaps.from_npz(str(cache))

    if psf_file is None:
        raise ValueError(
            "No PSF model given and no usable cache. Pass --psf or --config "
            "(or --cache pointing at an existing .npz)."
        )

    psf_path = resolve_psf_model(psf_file, index=psf_index)

    print(f"computing moment maps from {psf_path} (this is the slow step)")
    maps = compute_em5_psfex_maps(str(psf_path), **kwargs)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        maps.to_npz(str(cache), psfex_file=str(psf_path))
        print(f"cached maps to {cache}")

    return maps


def retain_observed_row(fig, axes):
    """Remove diagnostic artists from the upstream figure; retain its maps/colors.

    The upstream function always creates a residual row. This is a presentation
    edit to its returned artists, not a second moment measurement or plotter.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FormatStrFormatter
    fig.set_layout_engine(None)
    keep = set(axes[0])
    for ax in list(fig.axes):
        if ax not in keep:
            ax.remove()
    for label in list(fig.texts):
        label.remove()
    fig.set_size_inches(10.8, 2.65)
    for col, ax in enumerate(axes[0]):
        ax.set_position([.065 + col*.315, .22, .255, .64])
        ax.set_axes_locator(None)
        ax.set_xticks([2000, 4000, 6000, 8000])
        ax.set_yticks([2000, 4000, 6000])
        ax.set_xlabel("X [pixels]", fontsize=11)
        ax.set_ylabel("Y [pixels]" if col == 0 else "", fontsize=11)
        ax.tick_params(labelsize=10, labelleft=col == 0, labelbottom=True)
        ax.set_title(ax.get_title(), fontsize=12)
        im = ax.images[0]
        im.colorbar = None
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=.10)
        bar = fig.colorbar(im, cax=cax)
        lo, hi = im.get_clim()
        bar.set_ticks([lo, 0, hi] if col < 2 else [lo, (lo+hi)/2, hi])
        bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        bar.ax.tick_params(labelsize=10, pad=2)
    return axes[:1]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("PSF source (one of)")
    src.add_argument("--psf", default=None,
                     help="a PSFEx .psf model, or a directory of them")
    src.add_argument(
        "--config", default=None,
        help="ShearNet config yaml; reads paths.psfex_model_file from it",
    )
    src.add_argument(
        "--psf-index", type=int, default=0,
        help="which model to use when the source is a directory, in sorted "
             "order (the same order ShearNet draws from). Default 0. The figure "
             "shows ONE model out of the training set, so the caption has to "
             "name it -- the chosen filename is printed.",
    )

    p.add_argument(
        "--cache", default=None,
        help="npz path for the computed moment maps (reused unless --recompute)",
    )
    p.add_argument("--recompute", action="store_true", help="ignore an existing cache")

    grid = p.add_argument_group("sampling grid (passed to superbit_lensing)")
    grid.add_argument("--step", type=int, default=200, help="grid step in pixels")
    grid.add_argument("--image-xsize", type=int, default=9600)
    grid.add_argument("--image-ysize", type=int, default=6400)
    grid.add_argument("--margin", type=int, default=0)
    grid.add_argument("--scale", type=float, default=0.141, help="arcsec/pixel")

    rows = p.add_argument_group("panel selection")
    rows.add_argument("--rows", choices=("observed", "diagnostic"), default="observed",
                      help="paper default: observed PSF only; diagnostic restores all rows")
    rows.add_argument("--no-observed-row", action="store_true")
    rows.add_argument("--no-model-row", action="store_true")

    p.add_argument(
        "-o", "--out", default=None,
        help="output stem (no extension). Default ../figures/psf_properties",
    )
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    if args.rows == "observed" and args.no_observed_row:
        p.error("--no-observed-row requires --rows diagnostic")

    psf_file = args.psf
    if psf_file is None and args.config:
        psf_file = psfex_from_config(Path(args.config).expanduser())
        print(f"PSF model from config: {psf_file}")

    maps = build_maps(
        psf_file,
        cache=args.cache,
        recompute=args.recompute,
        psf_index=args.psf_index,
        step=args.step,
        image_xsize=args.image_xsize,
        image_ysize=args.image_ysize,
        margin=args.margin,
        scale=args.scale,
    )

    _, _, plot_em5_psfex_maps = _import_superbit()
    fig, _axes = plot_em5_psfex_maps(
        maps,
        show=False,
        SHOW_OBSERV_ROW=not args.no_observed_row,
        SHOW_MODEL_ROW=not args.no_model_row,
    )

    label_psf_properties(
        _axes, residual_only=args.no_observed_row and args.no_model_row,
    )

    from paper_colors import color_psf_maps
    color_psf_maps(_axes)

    if args.rows == "observed":
        _axes = retain_observed_row(fig, _axes)

    stem = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "figures" / "psf_properties"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.format:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
