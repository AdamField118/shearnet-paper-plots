#!/usr/bin/env python3
"""Replay the five figures in the 2026-09-21 revision from the supplied ZIP.

Archive path only: never apply these coordinates to a new run. The original
PSF moment arrays/evaluation FITS were not supplied. This copies the three
unchanged PDFs, retains only the observed PSF row (and its original x labels),
and calls unit_test_bias.draw for the explicit pending-results layout.
Use native psf/ and results/ generators for new data, not this archive command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import fitz

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "results"))
from unit_test_bias import draw
import matplotlib.pyplot as plt


def crop_psf(data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != "8f062bab93593255ef76eb8929d67b5e3c3716c13e49d9e8fd02d7fb2d307601":
        raise ValueError("PSF PDF differs from ShearNet_Paper(4).zip; use the native generator")
    source = fitz.open(stream=data, filetype="pdf")
    width = source[0].rect.width
    out = fitz.open()
    page = out.new_page(width=width, height=284)
    # Keep the observed map and color bars unchanged, with a small top margin.
    page.show_pdf_page(fitz.Rect(0, 4, width, 249), source, 0,
                       clip=fitz.Rect(0, 0, width, 245))
    # Reuse the original detector-coordinate labels from the bottom row.
    for x0, x1 in ((53, 373), (446, 766), (839, 1159)):
        page.show_pdf_page(fitz.Rect(x0, 245, x1, 280), source, 0,
                           clip=fitz.Rect(x0, 773, x1, 808))
    # Enlarge the detector/color-bar labels for a two-column manuscript width.
    # Reuse the exact displayed numbers, not inferred map limits or refitted data.
    labels = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                box = fitz.Rect(span["bbox"])
                if (span["text"] in {"2000", "4000", "6000", "8000", "0.06", "0.00", "−0.06", "-0.06", "0.09", "0.14", "0.18", "X [pixels]", "Y [pixels]"}
                    and abs(span["size"] - 10) < .1 and box.y0 >= 0 and box.y1 < 285):
                    labels.append((span["text"], box, line["dir"]))
    enlarged = fitz.open()
    final = enlarged.new_page(width=width + 50, height=310)
    final.show_pdf_page(fitz.Rect(30, 4, width + 30, 288), out, 0)
    for text, box, direction in labels:
        shifted = box + (30, 4, 30, 4)
        final.draw_rect(shifted, color=None, fill=(1,1,1), overlay=True)
    for text, box, direction in labels:
        cx, cy = (box.x0+box.x1)/2 + 30, (box.y0+box.y1)/2 + 4
        size = 18
        if text == "Y [pixels]":
            final.insert_text((18, 185), text, fontsize=size, fontname="helv", rotate=90)
        else:
            if box.x1 < 53: cx -= 4
            if text == "X [pixels]": cy = 294
            elif box.y0 > 246: cy = 264
            length = fitz.get_text_length(text.replace("−", "-"), fontname="helv", fontsize=size)
            final.insert_text((cx-length/2, cy+size*.35), text.replace("−", "-"),
                              fontsize=size, fontname="helv")
    result = enlarged.tobytes(garbage=4, deflate=True)
    enlarged.close()
    out.close()
    source.close()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-zip", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    manifest = json.loads(Path(__file__).with_name("advisor_figures_20260921.json").read_text())
    with zipfile.ZipFile(a.paper_zip) as z:
        originals = {name: z.read("figures/"+name) for name in manifest["sources"]}
    for name, expected in manifest["sources"].items():
        if hashlib.sha256(originals[name]).hexdigest() != expected:
            raise ValueError(f"{name}: source does not match the supplied September 21 ZIP")
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in originals.items():
        if name == "psf_properties.pdf":
            data = crop_psf(data)
        (a.out/name).write_bytes(data)
    data = {"components": {"m": 1, "c": 2, "applied_shear": 1}, "runs": {}}
    fig = draw(data)
    fig.savefig(a.out/"unit_test_bias.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    (a.out/"unit_test_bias.json").write_text(json.dumps(data, indent=2)+"\n")
    # Hash the visible rendering, since PDF object IDs need not be identical.
    for name, expected in manifest.get("rendered_rgb_sha256", {}).items():
        with fitz.open(a.out/name) as d:
            pix = d[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            actual = hashlib.sha256(pix.samples).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name}: rendering differs; check the pinned replay dependencies")
    print(f"Reproduced five manuscript figures in {a.out}")


if __name__ == "__main__":
    main()
