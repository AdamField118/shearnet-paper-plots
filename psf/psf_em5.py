"""Compare the empirical PSF used by a ShearNet run with its five-Gaussian fit.

The fit and shared log stretch follow LITB III, published Fig. C1:
s-Sayan/LITB-III-plots@4712964, Appendix/C/figC2_star_psfex_ngmix.ipynb.
A PSFEx file supplies the model, not the measured stellar vignette. The third
panel therefore shows the signed residual on the detector grid.
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from paper_colors import signed_cmap

# At the tangent point ShearNet's unrotated TAN WCS is a local pixel scale.
# shearnet/core/dataset.py: WCS_PARAMS, FITS (1-based) detector coordinates.
CCD_CENTER = (4800.5, 3211.5)
CCD_SCALE = 0.1408  # arcsec / detector pixel; the output sampling comes from the config


def render_model(path, size, scale, fft_size):
    import galsim
    from galsim.des import DES_PSFEx

    model = DES_PSFEx(str(path), wcs=galsim.PixelScale(CCD_SCALE))
    obj = model.getPSF(galsim.PositionD(*CCD_CENTER),
                       gsparams=galsim.GSParams(minimum_fft_size=fft_size,
                                                 maximum_fft_size=fft_size))
    # The empirical PSF already includes the detector pixel response.
    image = obj.drawImage(nx=size, ny=size, scale=scale, method="no_pixel", dtype=np.float64).array
    flux = image.sum()
    if not np.isfinite(image).all() or flux <= 0:
        raise ValueError("The PSFEx stamp must be finite with positive total flux")
    return image / flux


def em5_image(image, scale):
    import ngmix

    cy, cx = (np.asarray(image.shape) - 1) / 2
    obs = ngmix.Observation(image, jacobian=ngmix.DiagonalJacobian(row=cy, col=cx, scale=scale))
    guess = ngmix.guessers.GMixPSFGuesser(ngauss=5, rng=np.random.RandomState(1457))(obs)
    fit = ngmix.em.EMFitter(tol=1.e-6, maxiter=50000).go(obs=obs, guess=guess)
    if fit["flags"]:
        raise RuntimeError(f"EM5 failed: {fit.get('message', fit['flags'])}")
    model = fit.make_image()
    if not np.isfinite(model).all():
        raise ValueError("The fitted EM5 image contains nonfinite pixels")
    return model


def comparison(model, fit):
    # Fit to the original linear pixel values. Apply clipping only for display.
    # Show the central 33 pixels (the LITB III vignette size), after fitting
    # the whole simulation stamp. Both panels use limits from the visible pixels.
    side = min(33, *model.shape)
    y, x = (np.asarray(model.shape) - side) // 2
    model, fit = (a[y:y+side, x:x+side] for a in (model, fit))
    residual = model - fit
    values = np.concatenate((model.ravel(), fit.ravel()))
    lo, hi = np.percentile(values[values > 0], (1, 99.7))
    norm = LogNorm(lo, hi, clip=True)
    limit = np.abs(residual).max()
    cmap = signed_cmap()
    fig = plt.figure(figsize=(7.1, 3.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1, .055),
                            left=.04, right=.955, top=.88, bottom=.21, hspace=.12, wspace=.13)
    axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    for ax, values, title in zip(axes, (model, fit, residual),
                                 ("PSFEx model", "Five-Gaussian fit", "PSFEx minus fit")):
        ax.set_title(title, fontsize=11)
        ax.set(xticks=[], yticks=[])
        ax.imshow(values, origin="lower", interpolation="nearest", cmap=cmap,
                  norm=Normalize(-limit, limit) if ax is axes[-1] else norm)
    cb = fig.colorbar(axes[0].images[0], cax=fig.add_subplot(grid[1, :2]), orientation="horizontal")
    cb.set_label("Fraction of stamp flux per pixel")
    cb = fig.colorbar(axes[-1].images[0], cax=fig.add_subplot(grid[1, 2]), orientation="horizontal",
                      ticks=[-limit, 0, limit] if limit else [0], format="%.2g")
    cb.set_label("Flux difference per pixel")
    for ax in fig.axes[3:]:
        ax.tick_params(labelsize=8)
    return fig


def main(argv=None):
    import yaml

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="fourth_worstpsf.yaml")
    parser.add_argument("--out", type=Path, default=Path("output/paper/psf_em5.pdf"))
    args = parser.parse_args(argv)
    config = args.config.resolve()
    sim = yaml.safe_load(config.read_text())["simulation"]
    path = config.parent / Path(sim["psf"]["psfex_file"]).expanduser()
    if not path.is_file():
        parser.error(f"No PSFEx model file at {path} (from {config})")
    scale = float(sim["pixel_scale"])
    stamp = render_model(path, sim["stamp_size"], scale, sim["jax_fft_size"])
    fit = em5_image(stamp, scale)
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False,
                         "font.size": 9, "pdf.fonttype": 42}):
        fig = comparison(stamp, fit)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=.07, metadata={"Title": "PSFEx model and five-Gaussian fit",
                    "Subject": f"{path.name} at detector (x, y) = {CCD_CENTER}"})
        plt.close(fig)
    print(f"wrote {args.out} from {path} at {CCD_CENTER}")


if __name__ == "__main__":
    main()
