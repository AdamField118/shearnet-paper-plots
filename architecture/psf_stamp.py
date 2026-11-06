"""Render both architecture input thumbnails from the saved simulation arrays."""
from pathlib import Path
import sys

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "results"))
from paper_colors import signed_cmap

STAMP_CMAP = signed_cmap()
DISPLAY_PERCENTILES = (1.0, 99.7)
# The same value labels the crop in the Typst figure.
PREVIEW_SIDE = int((HERE / "stamps/psf_preview_side.txt").read_text())


def psf_preview(psf):
    if psf.ndim != 2 or not 0 < PREVIEW_SIDE <= min(psf.shape):
        raise ValueError("Expected a PSF image at least as large as its preview")
    row, col = ((size - PREVIEW_SIDE) // 2 for size in psf.shape)
    return psf[row:row + PREVIEW_SIDE, col:col + PREVIEW_SIDE]


def galaxy_norm(galaxy):
    low, high = np.percentile(galaxy, DISPLAY_PERCENTILES)
    return Normalize(vmin=low, vmax=high, clip=True)


def psf_norm(psf):
    positive = psf[psf > 0]
    if not np.isfinite(psf).all() or not positive.size:
        raise ValueError("Expected a finite PSF stamp with a positive peak")
    low, high = np.percentile(positive, DISPLAY_PERCENTILES)
    return LogNorm(vmin=low, vmax=high, clip=True)


if __name__ == "__main__":
    stamps = HERE / "stamps"
    with np.load(stamps / "single_stamp.npz") as data:
        for name, normalization in (("galaxy", galaxy_norm), ("psf", psf_norm)):
            pixels = psf_preview(data[name]) if name == "psf" else data[name]
            rgb = STAMP_CMAP(normalization(pixels)(pixels), bytes=True)[..., :3]
            # PNG row zero is at the top. Figure C1 uses origin="lower".
            Image.fromarray(np.flipud(rgb)).save(stamps / f"{name}.png")
