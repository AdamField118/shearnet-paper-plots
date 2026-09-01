"""PSF properties figure: PSFEx vs. ngmix EM5 moment maps across the detector.

Reproduces Figure C4 of the SuperBIT weak-lensing paper (Saha et al. 2026) for the
PSF that ShearNet is actually trained on: a 3x3 grid of (e1, e2, T) columns against
(observed PSFEx / EM5 model / residual) rows, laid out over CCD coordinates.

**All of the science and all of the drawing is done by ``superbit_lensing``**, not
here. This module only:

  1. resolves which PSFEx file to use (from a ShearNet config, or an explicit path),
  2. caches the expensive moment-map computation to an ``.npz``, and
  3. writes the figure out.

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


def build_maps(psf_file, cache=None, recompute=False, **kwargs):
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

    psf_path = Path(psf_file).expanduser()
    if not psf_path.is_file():
        raise FileNotFoundError(f"PSFEx model not found: {psf_path}")

    print(f"computing moment maps from {psf_path} (this is the slow step)")
    maps = compute_em5_psfex_maps(str(psf_path), **kwargs)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        maps.to_npz(str(cache), psfex_file=str(psf_path))
        print(f"cached maps to {cache}")

    return maps


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("PSF source (one of)")
    src.add_argument("--psf", default=None, help="path to a PSFEx .psf model")
    src.add_argument(
        "--config", default=None,
        help="ShearNet config yaml; reads paths.psfex_model_file from it",
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
    rows.add_argument("--no-observed-row", action="store_true")
    rows.add_argument("--no-model-row", action="store_true")

    p.add_argument(
        "-o", "--out", default=None,
        help="output stem (no extension). Default ../figures/psf_properties",
    )
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    psf_file = args.psf
    if psf_file is None and args.config:
        psf_file = psfex_from_config(Path(args.config).expanduser())
        print(f"PSF model from config: {psf_file}")

    maps = build_maps(
        psf_file,
        cache=args.cache,
        recompute=args.recompute,
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
