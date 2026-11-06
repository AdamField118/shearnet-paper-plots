"""Render the architecture PSF thumbnail from the saved simulation pixels."""
from pathlib import Path
import sys

import numpy as np
from matplotlib.colors import AsinhNorm
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "results"))
from paper_colors import signed_cmap

PSF_CMAP = signed_cmap()
# Linear near the background, logarithmic in the bright core.
LINEAR_FRACTION_OF_PEAK = 1e-3


def psf_norm(psf):
    if not np.isfinite(psf).all() or psf.max() <= 0:
        raise ValueError("Expected a finite PSF stamp with a positive peak")
    peak = float(psf.max())
    return AsinhNorm(
        linear_width=LINEAR_FRACTION_OF_PEAK * peak,
        vmin=0.0,
        vmax=peak,
        clip=True,
    )


if __name__ == "__main__":
    stamps = HERE / "stamps"
    with np.load(stamps / "single_stamp.npz") as data:
        psf = data["psf"]
    rgb = PSF_CMAP(psf_norm(psf)(psf), bytes=True)[..., :3]
    Image.fromarray(rgb).save(stamps / "psf.png")
