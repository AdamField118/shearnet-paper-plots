"""Draw the configured ShearNet-D4 architecture as a paper figure.

The forward path matches ``fourth_inloop_shearnet_d4/config.yaml`` in the
ShearNet repository: joint D4 orbit evaluation, separate ``shearnet-d4``
galaxy/PSF backbones, transformer fusion, inverse alignment and Reynolds
averaging, four-head learned spatial pooling, odd shear heads, and invariant
size/flux heads.

Usage
-----
    python shearnet_d4_architecture.py
    python shearnet_d4_architecture.py --format pdf png --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


FIGSIZE = (11.0, 5.5)
XLIM = (0.0, 180.0)
YLIM = (0.0, 92.0)

# Standalone exports use the same data-coordinate scale as the composed
# figure, with just enough outer padding to keep strokes and labels intact.
PANEL_SPECS = {
    "forward": ((11.0, 3.06), (0.0, 180.0), (42.0, 92.0)),
    "detail3": ((7.6, 3.31), (29.8, 123.2), (0.8, 41.5)),
    "detail4": ((7.2, 2.65), (0.5, 79.5), (0.5, 29.5)),
    "details56": ((7.2, 5.27), (123.7, 179.3), (0.8, 41.5)),
}

C = {
    "ink": "#171717",
    "muted": "#5A6168",
    "line": "#2D3135",
    "galaxy": "#C45A15",
    "galaxy_fill": "#F7E3D5",
    "psf": "#087C78",
    "psf_fill": "#DCEFED",
    "orbit": "#B47B13",
    "orbit_fill": "#FBF1D8",
    "feature": "#4C6FA8",
    "feature_fill": "#DFE8F5",
    "fusion": "#34764B",
    "fusion_fill": "#E1F0E5",
    "pool": "#6652A3",
    "pool_fill": "#EAE6F6",
    "head": "#A43B46",
    "head_fill": "#F5E1E4",
    "training_fill": "#F3F3F1",
}

FS = {
    # The 11-inch source is normally inserted at a 7.1-7.3 inch two-column
    # width. These sizes therefore land at approximately 6-9 pt in print.
    "group": 14.0,
    "label": 12.0,
    "body": 10.8,
    "small": 9.8,
    "tiny": 9.0,
    "formula": 11.2,
}

ORBIT_LABELS = [
    r"$I$",
    r"$R_{90}$",
    r"$R_{180}$",
    r"$R_{270}$",
    r"$P$",
    r"$R_{90}P$",
    r"$R_{180}P$",
    r"$R_{270}P$",
]
W1 = [(-1) ** (i % 4) for i in range(8)]
W2 = [(-1) ** ((i % 4) + (i // 4)) for i in range(8)]


def set_style(usetex: bool = False) -> None:
    """Use a restrained serif style that survives full-page-width scaling."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "text.usetex": usetex,
            "mathtext.fontset": "cm" if usetex else "dejavuserif",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def load_stamps() -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        Path(__file__).resolve().parent / "stamps" / "single_stamp.npz",
        Path(__file__).resolve().parent / "single_stamp.npz",
        Path("/mnt/data/single_stamp.npz"),
    ]
    stamp_path = next((path for path in candidates if path.is_file()), None)
    if stamp_path is None:
        tried = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError("stamp file not found. Tried:\n  - " + tried)
    with np.load(stamp_path) as data:
        missing = {"galaxy", "psf"} - set(data.files)
        if missing:
            raise KeyError(f"{stamp_path} is missing array(s): {sorted(missing)}")
        galaxy = np.asarray(data["galaxy"])
        psf = np.asarray(data["psf"])
    if galaxy.shape != (53, 53) or psf.shape != (53, 53):
        raise ValueError(f"expected configured 53x53 stamps, got {galaxy.shape} and {psf.shape}")
    return galaxy, psf


def d4_apply(image: np.ndarray, i: int) -> np.ndarray:
    """Apply ``R90**r . P**m``, matching ``shearnet.core.models._d4_apply``."""
    r, m = i % 4, i // 4
    result = np.flip(image, axis=0) if m else image
    return np.rot90(result, r) if r else result


def rounded_box(ax, x, y, w, h, *, edge=C["line"], fill="white", lw=0.9,
                radius=1.0, linestyle="-", zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text_box(ax, x, y, w, h, text, *, edge=C["line"], fill="white", fontsize=None,
             weight="normal", radius=0.8, lw=0.85, color=C["ink"], linespacing=1.15):
    rounded_box(ax, x, y, w, h, edge=edge, fill=fill, lw=lw, radius=radius)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize or FS["body"],
        fontweight=weight,
        color=color,
        multialignment="center",
        linespacing=linespacing,
        zorder=7,
    )


def arrow(ax, p0, p1, *, color=C["line"], lw=1.2, rad=0.0, style="-|>", zorder=8):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=10.0,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=zorder,
        )
    )


def stamp(ax, cx, cy, size, image, *, norm, edge):
    ax.imshow(
        image,
        extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2),
        cmap="viridis",
        norm=norm,
        interpolation="bilinear",
        aspect="auto",
        zorder=4,
    )
    ax.add_patch(
        plt.Rectangle(
            (cx - size / 2, cy - size / 2),
            size,
            size,
            facecolor="none",
            edgecolor=edge,
            linewidth=0.8,
            zorder=5,
        )
    )


def slab_stack(ax, x, y, *, color, n=4, w=5.0, h=8.0, dx=0.75, dy=0.65,
               label=None, zorder=3):
    """Draw a compact stack of spatial feature maps."""
    for k in reversed(range(n)):
        xk, yk = x + k * dx, y + k * dy
        poly = Polygon(
            [(xk, yk), (xk + w, yk), (xk + w + 0.8, yk + h), (xk + 0.8, yk + h)],
            closed=True,
            facecolor=to_rgba(color, 0.15 + 0.05 * (n - k)),
            edgecolor=color,
            linewidth=0.65,
            zorder=zorder + (n - k),
        )
        ax.add_patch(poly)
    if label:
        ax.text(x + w / 2 + n * dx / 2, y - 1.6, label, ha="center", va="top",
                fontsize=FS["small"], color=C["ink"], zorder=8)


def stage_label(ax, x, y, text, *, sub=None, weight="bold", style="normal"):
    ax.text(x, y, text, ha="center", va="top", fontsize=FS["small"],
            color=C["ink"], fontweight=weight, style=style)
    if sub:
        ax.text(x, y - 3.1, sub, ha="center", va="top", fontsize=FS["tiny"],
                color=C["muted"], linespacing=1.15)


def attention_icon(ax, cx, cy, size, center, color):
    """Schematic learned spatial probability map; not a measured diagnostic."""
    yy, xx = np.mgrid[-1:1:31j, -1:1:31j]
    sx, sy = center
    image = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 0.22)
    ax.imshow(
        image,
        extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2),
        cmap="Purples",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        aspect="auto",
        zorder=4,
    )
    ax.add_patch(plt.Rectangle((cx - size / 2, cy - size / 2), size, size,
                               facecolor="none", edgecolor=color, linewidth=0.55, zorder=5))


def draw_main_pipeline(ax, galaxy, psf, gal_norm, psf_norm):
    """Draw the sparse, left-to-right forward path."""
    # Inputs
    stamp(ax, 8.5, 74.0, 10.5, galaxy, norm=gal_norm, edge=C["galaxy"])
    stamp(ax, 8.5, 58.5, 10.5, psf, norm=psf_norm, edge=C["psf"])
    ax.text(8.5, 80.8, "galaxy", ha="center", fontsize=FS["body"],
            color=C["galaxy"], fontweight="bold")
    ax.text(8.5, 65.3, "PSF", ha="center", fontsize=FS["body"],
            color=C["psf"], fontweight="bold")
    stage_label(ax, 8.5, 48.0, "1. Inputs", sub=r"$53\!\times\!53$ pixels")

    # Joint orbit, following the compact stacked presentation common in D4 papers.
    rounded_box(ax, 18.5, 52.5, 23.0, 37.0, edge=C["orbit"], fill=C["orbit_fill"],
                lw=1.0, radius=2.1)
    ax.text(30.0, 86.6, r"Joint $D_4$ orbit", ha="center", va="center",
            fontsize=FS["label"], color=C["ink"], fontweight="bold")
    for i, label in enumerate(ORBIT_LABELS):
        col, row = i % 2, i // 2
        x, y = 20.5 + col * 9.6, 77.8 - row * 6.2
        text_box(ax, x, y, 8.5, 4.7, label, edge=C["orbit"], fill="white",
                 fontsize=FS["small"], radius=0.7, lw=0.7)
    ax.text(30.0, 55.6, "galaxy and PSF\ntransformed jointly", ha="center", va="center",
            fontsize=FS["tiny"], color=C["muted"], style="italic", linespacing=1.05)
    stage_label(ax, 30.0, 48.0, r"2. Joint orbit", sub="8 joint variants")
    arrow(ax, (14.2, 66.0), (18.4, 66.0), color=C["line"])

    # One weight-shared per-variant network.
    rounded_box(ax, 45.0, 55.0, 24.5, 31.0, edge=C["fusion"], fill="white", lw=1.0, radius=1.7)
    ax.text(57.25, 84.0, r"$F_{\theta}$", ha="center", va="center",
            fontsize=FS["group"], fontweight="bold", color=C["fusion"])
    ax.text(57.25, 79.3, "one parameter set\nfor all 8 variants", ha="center", va="center",
            fontsize=FS["tiny"], color=C["muted"], style="italic", linespacing=1.0)
    text_box(ax, 47.0, 70.2, 8.0, 6.5, "galaxy\nCNN", edge=C["galaxy"],
             fill=C["galaxy_fill"], fontsize=FS["small"], weight="bold")
    text_box(ax, 47.0, 59.4, 8.0, 6.5, "PSF\nCNN", edge=C["psf"],
             fill=C["psf_fill"], fontsize=FS["small"], weight="bold")
    text_box(ax, 59.0, 63.0, 8.5, 10.0, "fusion", edge=C["fusion"],
             fill=C["fusion_fill"], fontsize=FS["body"], weight="bold")
    arrow(ax, (55.0, 73.5), (58.9, 69.5), color=C["galaxy"], rad=-0.08)
    arrow(ax, (55.0, 62.7), (58.9, 66.3), color=C["psf"], rad=0.08)
    stage_label(ax, 57.25, 48.0, r"3. Shared $F_{\theta}$", sub="shared weights")
    arrow(ax, (41.5, 66.0), (44.9, 66.0))

    # Eight fused spatial maps.
    slab_stack(ax, 73.0, 62.0, color=C["feature"], n=5, w=4.6, h=12.5, dx=0.75, dy=0.65)
    ax.text(77.0, 78.4, r"$F_0,\ldots,F_7$", ha="center", fontsize=FS["body"],
            color=C["feature"], fontweight="bold")
    stage_label(ax, 77.0, 48.0, r"Tensor $F_i$", sub=r"$8\!\times\!13^2\!\times\!64$",
                weight="normal", style="italic")
    arrow(ax, (69.5, 66.0), (72.8, 66.0))

    # Alignment and the signed/sign-free group averages.
    rounded_box(ax, 84.0, 55.0, 23.5, 31.0, edge=C["orbit"], fill="white", lw=1.0, radius=1.7)
    text_box(ax, 87.2, 75.6, 17.1, 6.5,
             "inverse-align" + "\n" + r"$\widetilde F_i=g_i^{-1}F_i$",
             edge=C["orbit"], fill=C["orbit_fill"], fontsize=FS["tiny"],
             linespacing=1.0)
    ax.text(95.75, 68.9,
            r"$\Psi_c=\frac{1}{8}\sum_i w_c(g_i)\widetilde F_i$",
            ha="center", va="center", fontsize=FS["formula"], color=C["ink"])
    ax.text(95.75, 62.7,
            r"$\Psi_{\rm inv}=\frac{1}{8}\sum_i\widetilde F_i$",
            ha="center", va="center", fontsize=FS["formula"], color=C["ink"])
    ax.text(95.75, 58.2, "signed / sign-free\naverages", ha="center", va="center",
            fontsize=FS["tiny"], color=C["muted"], linespacing=1.05)
    stage_label(ax, 95.75, 48.0, "4. Reynolds", sub="signed / invariant")
    arrow(ax, (81.5, 66.0), (83.9, 66.0))

    # Three spatial representation maps.
    for j, (label, y, color) in enumerate([
        (r"$\Psi_1$", 74.0, C["head"]),
        (r"$\Psi_2$", 66.0, C["head"]),
        (r"$\Psi_{\rm inv}$", 58.0, C["feature"]),
    ]):
        slab_stack(ax, 111.0, y, color=color, n=2, w=4.0, h=5.5, dx=0.55, dy=0.45)
        ax.text(116.8, y + 2.8, label, ha="left", va="center", fontsize=FS["body"], color=C["ink"])
    stage_label(ax, 116.0, 48.0, r"$\Psi$ maps", sub=r"$3\!\times\!13^2\!\times\!64$",
                weight="normal", style="italic")
    arrow(ax, (107.5, 66.0), (110.8, 66.0))

    # Learned spatial pooling, shown as schematic probability maps.
    rounded_box(ax, 123.0, 55.0, 21.5, 31.0, edge=C["pool"], fill="white", lw=1.0, radius=1.7)
    ax.text(133.75, 82.2, "Attention maps", ha="center", fontsize=FS["small"],
            color=C["pool"], fontweight="bold")
    centers = [(-0.35, -0.25), (0.35, -0.25), (-0.35, 0.35), (0.35, 0.35)]
    ax.text(133.75, 78.8, r"$A_1,\ldots,A_4$", ha="center", va="center",
            fontsize=FS["tiny"], color=C["muted"])
    for k, center in enumerate(centers):
        cx = 129.9 + (k % 2) * 7.7
        cy = 73.4 - (k // 2) * 7.8
        attention_icon(ax, cx, cy, 6.1, center, C["pool"])
    ax.text(133.75, 57.7, r"$A_k\leftarrow\Psi_{\rm inv}$", ha="center",
            fontsize=FS["small"], color=C["ink"])
    stage_label(ax, 133.75, 48.0, "5. Pooling", sub="4 spatial maps")
    arrow(ax, (120.8, 66.0), (122.9, 66.0))

    # Separate equivariant and invariant heads.
    rounded_box(ax, 147.0, 55.0, 20.5, 31.0, edge=C["head"], fill="white", lw=1.0, radius=1.7)
    text_box(ax, 149.3, 72.8, 15.9, 8.0, "bias-free\nodd MLP", edge=C["head"],
             fill=C["head_fill"], fontsize=FS["small"], weight="bold")
    text_box(ax, 149.3, 59.5, 15.9, 8.0, "invariant\nMLP", edge=C["feature"],
             fill=C["feature_fill"], fontsize=FS["small"], weight="bold")
    stage_label(ax, 157.25, 48.0, "6. Output heads", sub="shear + scalars")
    arrow(ax, (144.5, 66.0), (146.9, 66.0))
    arrow(ax, (167.5, 76.8), (170.5, 76.8), color=C["head"])
    arrow(ax, (167.5, 63.5), (170.5, 63.5), color=C["feature"])
    ax.text(171.2, 76.8, r"$\widehat g_1,\widehat g_2$", ha="left", va="center",
            fontsize=FS["group"], color=C["head"], fontweight="bold")
    ax.text(171.2, 63.5, "hlr, flux", ha="left", va="center",
            fontsize=FS["label"], color=C["feature"], fontweight="bold")


def draw_architecture_detail(ax):
    """Compact inset containing the exact configured module depths and widths."""
    rounded_box(ax, 31.0, 2.0, 91.0, 38.5, edge=C["fusion"], fill="white", lw=0.9,
                radius=1.2)
    ax.text(33.2, 37.5, r"Detail 3: shared per-variant $F_{\theta}$", ha="left", va="center",
            fontsize=FS["label"], color=C["ink"], fontweight="bold")
    ax.text(119.8, 37.5, "distinct galaxy / PSF weights", ha="right", va="center",
            fontsize=FS["tiny"], color=C["muted"], style="italic")

    ax.text(33.0, 28.6, "Galaxy", ha="left", va="center", fontsize=FS["small"],
            color=C["galaxy"], fontweight="bold")
    ax.text(33.0, 16.8, "PSF", ha="left", va="center", fontsize=FS["small"],
            color=C["psf"], fontweight="bold")

    # Deliberately generous gaps keep the connector shafts visible after the
    # full figure is reduced to a two-column paper width.
    xs = [42.0, 51.7, 61.4, 72.6]
    widths = [7.0, 6.8, 8.5, 8.5]
    galaxy_text = [
        "stem\n53²×32",
        r"Res $\times2$" + "\n32 ch",
        r"$\downarrow2$" + "\n" + r"Res $\times2$" + "\n48 ch",
        r"$\downarrow2$" + "\n" + r"Res $\times1$" + "\n64 ch",
    ]
    psf_text = [
        "stem\n53²×32",
        r"Res $\times1$" + "\n32 ch",
        r"$\downarrow2$" + "\n" + r"Res $\times1$" + "\n48 ch",
        r"$\downarrow2$" + "\n" + r"Res $\times1$" + "\n64 ch",
    ]
    for row_y, texts, edge, fill in [
        (24.9, galaxy_text, C["galaxy"], C["galaxy_fill"]),
        (13.1, psf_text, C["psf"], C["psf_fill"]),
    ]:
        previous_end = None
        for x, w, text in zip(xs, widths, texts):
            text_box(ax, x, row_y, w, 7.4, text, edge=edge, fill=fill,
                     fontsize=7.8 if text.startswith("stem") else 8.2,
                     radius=0.55, lw=0.7, linespacing=1.0)
            if previous_end is not None:
                arrow(ax, (previous_end + 0.15, row_y + 3.7), (x - 0.15, row_y + 3.7),
                      color=edge, lw=1.0)
            previous_end = x + w

    text_box(ax, 84.3, 24.6, 9.5, 8.0,
             "dilated" + "\n" + r"$3\!\times\!3$" + "\n" + r"$d=1,2,4$",
             edge=C["galaxy"], fill=C["galaxy_fill"], fontsize=8.1, radius=0.55, lw=0.7)
    arrow(ax, (81.25, 28.6), (84.15, 28.6), color=C["galaxy"], lw=1.0)

    rounded_box(ax, 98.1, 7.3, 21.9, 27.0, edge=C["fusion"], fill=C["fusion_fill"],
                lw=0.8, radius=0.8)
    ax.text(109.05, 32.0, "Transformer\nfusion", ha="center", va="center",
            fontsize=FS["tiny"], color=C["fusion"], fontweight="bold", linespacing=1.0)
    text_box(ax, 99.5, 25.5, 19.1, 3.6,
             r"$1\!\times\!1$ projection" + "\n+ positions",
             edge=C["fusion"], fill="white", fontsize=7.8, radius=0.4, lw=0.6,
             linespacing=0.95)
    text_box(ax, 99.5, 19.6, 19.1, 4.0, "cross-attn\nG:Q; PSF:K,V",
             edge=C["fusion"], fill="white", fontsize=7.8, radius=0.4,
             lw=0.6, linespacing=1.0)
    text_box(ax, 99.5, 14.0, 19.1, 3.4, "self-attn + FFN",
             edge=C["fusion"], fill="white", fontsize=7.8, radius=0.4, lw=0.6)
    arrow(ax, (109.05, 25.35), (109.05, 23.75), color=C["fusion"], lw=0.8)
    arrow(ax, (109.05, 19.45), (109.05, 17.55), color=C["fusion"], lw=0.8)
    ax.text(109.05, 10.2, r"$d=64$; $h=4$" + "\nFFN width 256", ha="center", va="center",
            fontsize=8.1, color=C["muted"], linespacing=1.0)
    arrow(ax, (93.95, 28.6), (97.95, 27.2), color=C["galaxy"], rad=-0.07, lw=1.0)
    arrow(ax, (81.25, 16.8), (97.95, 19.2), color=C["psf"], rad=0.05, lw=1.0)

    ax.text(76.5, 4.8,
            r"LayerNorm + GELU; anti-aliased downsampling; outputs $13^2\!\times\!64$.",
            ha="center", va="center", fontsize=FS["tiny"], color=C["muted"])


def draw_symmetry_note(ax):
    rounded_box(ax, 2.0, 2.0, 26.0, 38.5, edge=C["orbit"], fill=C["orbit_fill"],
                lw=0.8, radius=1.2)
    ax.text(4.0, 37.7, "DETAIL 4", ha="left", va="center",
            fontsize=FS["tiny"], color=C["orbit"], fontweight="bold")
    ax.text(4.0, 34.7, "Reynolds weights", ha="left", va="center",
            fontsize=FS["small"], color=C["ink"], fontweight="bold")
    ax.text(5.0, 31.2, r"$w_1$", ha="left", va="center",
            fontsize=FS["tiny"], color=C["ink"])
    for x, pair in zip([7.1, 11.7, 16.3, 20.9], ["+  -", "+  -", "+  -", "+  -"]):
        text_box(ax, x, 27.2, 4.1, 3.2, pair, edge=to_rgba(C["orbit"], 0.7),
                 fill=to_rgba("white", 0.62), fontsize=8.2, radius=0.38, lw=0.55)
    ax.text(5.0, 26.0, r"$w_2$", ha="left", va="center",
            fontsize=FS["tiny"], color=C["ink"])
    for x, pair in zip([7.1, 11.7, 16.3, 20.9], ["+  -", "+  -", "-  +", "-  +"]):
        text_box(ax, x, 22.0, 4.1, 3.2, pair, edge=to_rgba(C["orbit"], 0.7),
                 fill=to_rgba("white", 0.62), fontsize=8.2, radius=0.38, lw=0.55)
    ax.text(15.0, 17.8, r"signed mean $\rightarrow \Psi_1,\Psi_2$" + "\n" +
            r"ordinary mean $\rightarrow \Psi_{\rm inv}$",
            ha="center", va="center", fontsize=FS["small"], color=C["muted"],
            linespacing=1.2)
    ax.plot([5.0, 25.0], [12.1, 12.1], color=to_rgba(C["orbit"], 0.55), linewidth=0.7)
    ax.text(15.0, 7.0, "Response term\n" + r"$K=4$ PSF-only orbits",
            ha="center", va="center", fontsize=FS["small"], color=C["ink"],
            linespacing=1.15)


def draw_symmetry_note_standalone(ax):
    """Landscape version of Detail 4 for an independent paper panel."""
    rounded_box(ax, 2.0, 2.0, 76.0, 26.0, edge=C["orbit"], fill=C["orbit_fill"],
                lw=0.8, radius=1.2)
    ax.text(4.0, 25.4, "Detail 4: Reynolds weights", ha="left", va="center",
            fontsize=FS["label"], color=C["ink"], fontweight="bold")

    pair_xs = [9.0, 14.6, 20.2, 25.8]
    rows = [
        (r"$w_1$", 18.1, ["+  -", "+  -", "+  -", "+  -"]),
        (r"$w_2$", 12.1, ["+  -", "+  -", "-  +", "-  +"]),
    ]
    for label, y, pairs in rows:
        ax.text(5.0, y + 1.65, label, ha="left", va="center",
                fontsize=FS["small"], color=C["ink"])
        for x, pair in zip(pair_xs, pairs):
            text_box(ax, x, y, 4.7, 3.3, pair, edge=to_rgba(C["orbit"], 0.7),
                     fill=to_rgba("white", 0.62), fontsize=FS["tiny"],
                     radius=0.38, lw=0.55)

    ax.text(43.0, 16.8,
            r"signed mean $\rightarrow \Psi_1,\Psi_2$" + "\n" +
            r"ordinary mean $\rightarrow \Psi_{\rm inv}$",
            ha="center", va="center", fontsize=FS["small"], color=C["muted"],
            linespacing=1.25)
    ax.plot([56.2, 56.2], [8.0, 22.0], color=to_rgba(C["orbit"], 0.55), linewidth=0.7)
    ax.text(67.0, 16.8, "Response term\n" + r"$K=4$ PSF-only orbits",
            ha="center", va="center", fontsize=FS["small"], color=C["ink"],
            linespacing=1.2)


def draw_pooling_detail(ax):
    rounded_box(ax, 125.0, 2.0, 53.0, 38.5, edge=C["pool"], fill="white", lw=0.8,
                radius=1.2)
    ax.text(127.0, 37.5, "Details 5-6: pooling + heads", ha="left", va="center",
            fontsize=FS["label"], color=C["ink"], fontweight="bold")
    text_box(ax, 128.0, 29.1, 5.5, 4.4, r"$\Psi_{\rm inv}$",
             edge=C["feature"], fill=C["feature_fill"], fontsize=FS["small"], radius=0.5)
    text_box(ax, 137.0, 28.8, 9.2, 5.0, "Dense 64\nGELU",
             edge=C["pool"], fill=C["pool_fill"], fontsize=8.2, radius=0.5,
             linespacing=1.0)
    text_box(ax, 149.7, 28.8, 9.6, 5.0, "Dense 4\n" + r"softmax$_p$",
             edge=C["pool"], fill=C["pool_fill"], fontsize=8.0, radius=0.5,
             linespacing=1.0)
    text_box(ax, 162.8, 29.1, 12.2, 4.4, r"$A_1,\ldots,A_4$",
             edge=C["pool"], fill="white", fontsize=8.2, radius=0.5)
    arrow(ax, (133.65, 31.3), (136.85, 31.3), color=C["pool"], lw=0.9)
    arrow(ax, (146.35, 31.3), (149.55, 31.3), color=C["pool"], lw=0.9)
    arrow(ax, (159.45, 31.3), (162.65, 31.3), color=C["pool"], lw=0.9)

    ax.text(151.5, 24.6, r"$s_{c,k}=\sum_p A_k(p)\Psi_c(p)$",
            ha="center", va="center", fontsize=FS["formula"], color=C["ink"])
    ax.text(151.5, 21.2,
            r"$s_c=[s_{c,1};\ldots;s_{c,4}]\in\mathbb{R}^{256}$",
            ha="center", va="center", fontsize=FS["small"], color=C["ink"])

    rounded_box(ax, 128.0, 7.0, 22.7, 11.0, edge=C["head"], fill=C["head_fill"],
                lw=0.75, radius=0.65)
    ax.text(139.35, 16.0, r"odd heads: $s_1,s_2$", ha="center", va="center",
            fontsize=FS["tiny"], color=C["head"], fontweight="bold")
    for x, w, label in [(129.4, 3.2, "256"), (135.5, 3.2, "128"),
                        (141.6, 3.2, "128"), (147.7, 2.3, "1")]:
        text_box(ax, x, 11.3, w, 2.8, label, edge=C["head"], fill="white",
                 fontsize=8.0, radius=0.35, lw=0.55)
    for x0, x1 in [(132.75, 135.35), (138.85, 141.45), (144.95, 147.55)]:
        arrow(ax, (x0, 12.7), (x1, 12.7), color=C["head"], lw=0.7)
    ax.text(139.35, 8.8, r"tanh $\rightarrow \widehat g_1,\widehat g_2$",
            ha="center", va="center", fontsize=FS["tiny"], color=C["ink"])

    rounded_box(ax, 152.3, 7.0, 22.7, 11.0, edge=C["feature"], fill=C["feature_fill"],
                lw=0.75, radius=0.65)
    ax.text(163.65, 16.0, r"invariant: $s_{\rm inv}$", ha="center", va="center",
            fontsize=FS["tiny"], color=C["feature"], fontweight="bold")
    for x, label in [(154.0, "256"), (161.3, "128"), (168.6, "1")]:
        text_box(ax, x, 11.3, 3.4, 2.8, label, edge=C["feature"], fill="white",
                 fontsize=8.0, radius=0.35, lw=0.55)
    for x0, x1 in [(157.55, 161.15), (164.85, 168.45)]:
        arrow(ax, (x0, 12.7), (x1, 12.7), color=C["feature"], lw=0.7)
    ax.text(163.65, 8.8, r"GELU $\rightarrow$ hlr, flux", ha="center", va="center",
            fontsize=FS["tiny"], color=C["ink"])

    arrow(ax, (151.5, 20.1), (139.35, 18.2), color=C["head"], rad=0.05, lw=1.0)
    arrow(ax, (151.5, 20.1), (163.65, 18.2), color=C["feature"], rad=-0.05, lw=1.0)
    ax.text(151.5, 4.2, r"Shared $A_k$ preserves $D_4$ symmetry.",
            ha="center", va="center", fontsize=FS["tiny"], color=C["muted"])


def build_figure(figsize=FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")

    galaxy, psf = load_stamps()
    gal_lo, gal_hi = np.percentile(galaxy, [1.0, 99.7])
    gal_norm = Normalize(vmin=float(gal_lo), vmax=float(gal_hi))
    positive_psf = psf[psf > 0]
    if not positive_psf.size:
        raise ValueError("PSF stamp must contain a positive pixel")
    psf_norm = LogNorm(vmin=float(np.percentile(positive_psf, 5.0)), vmax=float(psf.max()))

    draw_main_pipeline(ax, galaxy, psf, gal_norm, psf_norm)
    draw_symmetry_note(ax)
    draw_architecture_detail(ax)
    draw_pooling_detail(ax)
    return fig, ax


def build_panel(panel):
    """Build one standalone forward-path or detail figure."""
    if panel not in PANEL_SPECS:
        raise ValueError(f"unknown panel {panel!r}; expected one of {sorted(PANEL_SPECS)}")

    figsize, xlim, ylim = PANEL_SPECS[panel]
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")

    if panel == "forward":
        galaxy, psf = load_stamps()
        gal_lo, gal_hi = np.percentile(galaxy, [1.0, 99.7])
        gal_norm = Normalize(vmin=float(gal_lo), vmax=float(gal_hi))
        positive_psf = psf[psf > 0]
        if not positive_psf.size:
            raise ValueError("PSF stamp must contain a positive pixel")
        psf_norm = LogNorm(
            vmin=float(np.percentile(positive_psf, 5.0)), vmax=float(psf.max())
        )
        draw_main_pipeline(ax, galaxy, psf, gal_norm, psf_norm)
    elif panel == "detail3":
        draw_architecture_detail(ax)
    elif panel == "detail4":
        draw_symmetry_note_standalone(ax)
    else:
        draw_pooling_detail(ax)
    return fig, ax


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default=None,
                        help="output stem without extension (default: ../figures/shearnet_arch_d4)")
    parser.add_argument("--format", nargs="+", choices=("pdf", "png"), default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--usetex", action="store_true", help="use a local LaTeX installation")
    parser.add_argument(
        "--panel",
        choices=("split", "full", "forward", "detail3", "detail4", "details56"),
        default="split",
        help="figure to render; 'split' writes the four standalone paper figures",
    )
    args = parser.parse_args(argv)

    set_style(args.usetex)
    stem = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "figures" / "shearnet_arch_d4"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)

    panels = ("forward", "detail3", "detail4", "details56") if args.panel == "split" else (args.panel,)
    suffixes = {
        "forward": "forward",
        "detail3": "detail3_network",
        "detail4": "detail4_reynolds",
        "details56": "details56_pooling_heads",
    }

    for panel in panels:
        try:
            fig, _ = build_figure() if panel == "full" else build_panel(panel)
        except RuntimeError as exc:
            if not args.usetex:
                raise
            print(f"LaTeX rendering failed ({exc}); falling back to mathtext.")
            set_style(False)
            fig, _ = build_figure() if panel == "full" else build_panel(panel)

        panel_stem = stem
        if args.panel == "split":
            panel_stem = stem.with_name(f"{stem.name}_{suffixes[panel]}")
        for fmt in args.format:
            path = panel_stem.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=args.dpi)
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
