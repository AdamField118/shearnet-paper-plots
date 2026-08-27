"""Publication figure: the D4-equivariant ShearNet architecture.

Draws the flagship ``D4ForkLike`` model (shearnet/core/models.py) as a
left-to-right dataflow diagram:

    (galaxy, PSF) stamps
        -> D4 orbit (8 joint transforms)
        -> shared two-branch smooth backbones
        -> cross-attention transformer fusion
        -> inverse transform back to the reference frame
        -> sign-weighted Reynolds average  ->  Psi_1, Psi_2   (spin-2 equivariant)
                                           ->  Psi_inv        (D4 invariant)
        -> Gaussian-window pooling + heads  ->  (g1, g2) and (hlr, flux)

Pure matplotlib: no LaTeX toolchain required, so the figure builds anywhere the
conda environment builds.

Usage
-----
    python shearnet_d4_architecture.py                 # -> ../figures/shearnet_arch_d4.{pdf,png}
    python shearnet_d4_architecture.py -o out/arch     # custom stem
    python shearnet_d4_architecture.py --format png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------
# Style. One place to retune the whole figure.
# --------------------------------------------------------------------------

PALETTE = {
    "input": "#2D5AA0",      # stamps
    "orbit": "#6E378C",      # D4 orbit machinery
    "galaxy": "#D26419",     # galaxy branch
    "psf": "#197873",        # PSF branch
    "fusion": "#266C37",     # transformer fusion
    "equivar": "#B48C14",    # Reynolds / equivariant features
    "head": "#AF2D2D",       # output heads
    "edge": "#191932",
    "text": "#191932",
    "muted": "#828282",
}

FONT = {
    "title": 10.5,
    "block": 8.0,
    "sub": 6.6,
    "annot": 7.0,
    "tag": 6.2,
}

BOX_STYLE = "round,pad=0.03,rounding_size=0.10"


def _fig_style() -> None:
    """Rc settings shared by every panel of the figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,   # embed TrueType, not Type-3: journal-safe
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def block(
    ax,
    xy,
    w,
    h,
    label,
    color,
    sublabel=None,
    alpha=0.16,
    lw=1.3,
    fontsize=None,
    ls="-",
):
    """Draw a rounded block with a bold label and an optional detail line.

    ``xy`` is the lower-left corner in data coordinates. Returns the patch so
    callers can anchor arrows to it.
    """
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=BOX_STYLE,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle=BOX_STYLE,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            linestyle=ls,
            zorder=3,
        )
    )

    cx, cy = xy[0] + w / 2.0, xy[1] + h / 2.0
    if sublabel:
        ax.text(
            cx,
            cy + h * 0.16,
            label,
            ha="center",
            va="center",
            fontsize=fontsize or FONT["block"],
            color=PALETTE["text"],
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            cx,
            cy - h * 0.22,
            sublabel,
            ha="center",
            va="center",
            fontsize=FONT["sub"],
            color=PALETTE["muted"],
            zorder=4,
        )
    else:
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fontsize or FONT["block"],
            color=PALETTE["text"],
            fontweight="bold",
            zorder=4,
        )
    return patch


def arrow(ax, start, end, color=None, lw=1.3, ls="-", shrink=2.0, zorder=5):
    """Directed connector between two points in data coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            linestyle=ls,
            color=color or PALETTE["edge"],
            shrinkA=shrink,
            shrinkB=shrink,
            zorder=zorder,
        )
    )


def stamp(ax, cx, cy, size, kind="galaxy", n=13):
    """Render a small synthetic postage stamp (galaxy or PSF) as an inset image.

    Purely illustrative -- an elliptical Gaussian for the galaxy, a rounder,
    sharper one for the PSF -- so the figure shows what the network ingests.
    """
    g = np.linspace(-2.4, 2.4, n)
    xx, yy = np.meshgrid(g, g)
    if kind == "galaxy":
        # sheared, off-centre exponential-ish blob
        a = np.cos(0.6) * xx + np.sin(0.6) * yy
        b = -np.sin(0.6) * xx + np.cos(0.6) * yy
        img = np.exp(-np.sqrt((a / 1.35) ** 2 + (b / 0.72) ** 2 + 0.04) * 1.7)
        cmap = "magma"
    else:
        img = np.exp(-(xx**2 + yy**2) / 0.62)
        cmap = "cividis"

    ax.imshow(
        img,
        extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2),
        cmap=cmap,
        zorder=3,
        interpolation="bilinear",
    )
    ax.add_patch(
        plt.Rectangle(
            (cx - size / 2, cy - size / 2),
            size,
            size,
            facecolor="none",
            edgecolor=PALETTE["edge"],
            linewidth=0.9,
            zorder=4,
        )
    )


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def build_figure(figsize=(15.0, 6.4)):
    """Assemble the full D4 architecture diagram. Returns (fig, ax)."""
    _fig_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.5, 99.5)
    # tight around the drawn content (lowest annotation ~7.5, highest ~38)
    ax.set_ylim(6.5, 39.5)
    ax.axis("off")

    # ---------------- 1. inputs ----------------
    stamp(ax, 5.0, 27.0, 6.4, kind="galaxy")
    ax.text(5.0, 31.4, "galaxy", ha="center", fontsize=FONT["annot"],
            color=PALETTE["input"], fontweight="bold")
    stamp(ax, 5.0, 15.0, 6.4, kind="psf")
    ax.text(5.0, 19.4, "PSF", ha="center", fontsize=FONT["annot"],
            color=PALETTE["input"], fontweight="bold")
    ax.text(5.0, 10.4, r"53 $\times$ 53 px", ha="center",
            fontsize=FONT["tag"], color=PALETTE["muted"])

    # ---------------- 2. D4 orbit ----------------
    block(ax, (12.0, 13.0), 9.0, 16.0, r"$D_4$ orbit", PALETTE["orbit"],
          sublabel="8 joint\ntransforms")
    arrow(ax, (8.4, 27.0), (12.0, 24.0), color=PALETTE["input"])
    arrow(ax, (8.4, 15.0), (12.0, 18.0), color=PALETTE["input"])
    ax.text(16.5, 30.4, r"$g_i\cdot(x_{\rm gal},x_{\rm PSF})$", ha="center",
            fontsize=FONT["tag"], color=PALETTE["muted"])

    # ---------------- 3. shared two-branch backbones ----------------
    block(ax, (25.0, 24.0), 15.0, 9.5, "galaxy backbone", PALETTE["galaxy"],
          sublabel="Conv-LN-GeLU-AvgPool\n(16, 32)")
    block(ax, (25.0, 9.0), 15.0, 9.5, "PSF backbone", PALETTE["psf"],
          sublabel="Conv-LN-GeLU-AvgPool\n(16, 32)")
    arrow(ax, (21.0, 24.0), (25.0, 28.7), color=PALETTE["orbit"])
    arrow(ax, (21.0, 18.0), (25.0, 13.7), color=PALETTE["orbit"])
    ax.text(32.5, 35.0, "shared weights across all 8 orbit members",
            ha="center", fontsize=FONT["tag"], color=PALETTE["muted"], style="italic")

    # ---------------- 4. transformer fusion ----------------
    block(ax, (44.0, 15.0), 14.0, 12.5, "cross-attention\nfusion", PALETTE["fusion"],
          sublabel=r"galaxy $Q$ ; PSF $K,V$" + "\n" + r"$d=64$, 4 heads")
    arrow(ax, (40.0, 28.7), (44.0, 23.5), color=PALETTE["galaxy"])
    arrow(ax, (40.0, 13.7), (44.0, 19.0), color=PALETTE["psf"])

    # ---------------- 5. inverse transform ----------------
    block(ax, (61.5, 15.0), 9.5, 12.5, r"$g_i^{-1}$", PALETTE["orbit"],
          sublabel="align to\nreference frame")
    arrow(ax, (58.0, 21.25), (61.5, 21.25), color=PALETTE["fusion"])

    # ---------------- 6. Reynolds average ----------------
    ax.text(80.0, 37.6, "sign-weighted Reynolds average", ha="center",
            fontsize=FONT["annot"], color=PALETTE["equivar"], fontweight="bold")
    ax.text(
        80.0, 34.6,
        r"$\Psi_c=\frac{1}{8}\sum_i w_c(g_i)\,g_i^{-1}F(g_i\cdot x)$",
        ha="center", fontsize=FONT["annot"], color=PALETTE["text"],
    )

    block(ax, (74.0, 26.0), 12.0, 5.6, r"$\Psi_1,\ \Psi_2$", PALETTE["equivar"],
          sublabel="spin-2 equivariant")
    block(ax, (74.0, 12.2), 12.0, 5.6, r"$\Psi_{\rm inv}$", PALETTE["equivar"],
          sublabel=r"$D_4$ invariant", ls="--")
    arrow(ax, (71.0, 22.6), (74.0, 28.8), color=PALETTE["orbit"])
    arrow(ax, (71.0, 19.9), (74.0, 15.0), color=PALETTE["orbit"], ls="--")

    # ---------------- 7. pooling + heads ----------------
    ax.text(80.0, 22.9, r"$D_4$ Gaussian window $+$ GAP", ha="center",
            fontsize=FONT["tag"], color=PALETTE["muted"])

    block(ax, (89.5, 26.0), 9.0, 5.6, r"$(g_1, g_2)$", PALETTE["head"],
          sublabel="bias-free odd MLP")
    block(ax, (89.5, 12.2), 9.0, 5.6, r"hlr, flux", PALETTE["head"],
          sublabel="MLP", ls="--")
    arrow(ax, (86.0, 28.8), (89.5, 28.8), color=PALETTE["equivar"])
    arrow(ax, (86.0, 15.0), (89.5, 15.0), color=PALETTE["equivar"], ls="--")

    ax.text(
        94.0, 8.6,
        "auxiliary targets",
        ha="center", fontsize=FONT["tag"], color=PALETTE["muted"], style="italic",
    )
    ax.text(
        94.0, 34.0,
        "exactly spin-2\nby construction",
        ha="center", fontsize=FONT["tag"], color=PALETTE["head"], fontweight="bold",
    )

    fig.tight_layout()
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o", "--out",
        default=None,
        help="output path stem (no extension). Default: ../figures/shearnet_arch_d4",
    )
    p.add_argument(
        "--format", nargs="+", default=["pdf", "png"],
        help="one or more output formats (default: pdf png)",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    stem = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "figures" / "shearnet_arch_d4"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = build_figure()
    for fmt in args.format:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=args.dpi)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
