"""LITB III Figure C3 (star, PSFEx model, em5 fit) in the paper's navy-gray-gold map.

House-rule search: level 1, s-Sayan/LITB-III-plots @4712964, draws this figure in
`Appendix/C/figC2_star_psfex_ngmix.ipynb`. Its input, `star_to_psfs_images.fits`,
is not in that repository (`files/` is git-ignored); it lives on `hen` under
`/data/analysis/superbit_2023/sayan/litb3_files`. So instead of re-running the
notebook, this script recolors the published figure, as `recolor_paper5.py` does.

The notebook draws every panel with `imshow(cmap="inferno", norm=LogNorm(...))`,
so each pixel in the PDF is an index into Matplotlib's 256-entry inferno table.
Each image's palette must match that table exactly; each entry is then replaced
by the same index of `signed_cmap()`. Pixel indices, the shared log
normalization, labels, ticks and layout are untouched, so the result is what the
notebook would draw with `cmap=signed_cmap()`. Anything unexpected is rejected.

    python tools/recolor_litb3_c3.py --pdf star_to_psfs_to_ngmix_images.pdf \\
        --out figures/litb3_c3_star_psfex_em5.pdf
"""
from pathlib import Path
import argparse, hashlib, re, sys

import fitz
import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from paper_colors import signed_cmap

# star_to_psfs_to_ngmix_images.pdf as published with LITB III (arXiv:2603.18376v2).
SOURCE_SHA256 = "e9073bb6eca1eae559856b7a2864dbda2f88b87ae927c989af669c0bfadba341"
IMAGES = 4  # three panels and the colorbar


def recolor(doc):
    """Swap each indexed image's inferno palette for the paper map, in place."""
    inferno = matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255
    paper = (signed_cmap()(np.linspace(0, 1, 256))[:, :3] * 255).astype("uint8")
    images = doc[0].get_images()
    if len(images) != IMAGES:
        raise ValueError(f"expected {IMAGES} images, found {len(images)}")
    for xref, *_ in images:
        cs = doc.xref_get_key(xref, "ColorSpace")[1]
        match = re.search(r"<([A-Fa-f0-9]+)>", cs)
        if not cs.startswith("[/Indexed/DeviceRGB") or not match:
            raise ValueError("expected an indexed RGB image")
        colors = np.frombuffer(bytes.fromhex(match[1]), dtype=np.uint8).reshape(-1, 3)
        # Identify inferno entries by exact match, under either PDF rounding rule.
        best = None
        for lut in (inferno.astype("uint8"), np.round(inferno).astype("uint8")):
            dist = np.sum((colors[:, None, :].astype(float) - lut[None, :, :]) ** 2, axis=-1)
            if best is None or dist.min(axis=1).max() < best[0]:
                best = (dist.min(axis=1).max(), dist.argmin(axis=1))
        if best[0] != 0:
            raise ValueError(f"palette is not inferno (worst squared error {best[0]})")
        replacement = paper[best[1]].tobytes().hex()
        doc.xref_set_key(xref, "ColorSpace", cs[:match.start(1)] + replacement + cs[match.end(1):])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", required=True, type=Path, help="published star_to_psfs_to_ngmix_images.pdf")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    raw = args.pdf.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"unexpected source {args.pdf}")
    doc = fitz.open(stream=raw, filetype="pdf")
    recolor(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.out, garbage=4, deflate=True, no_new_id=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
