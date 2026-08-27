"""Publication figure: the D4-equivariant ShearNet architecture.

Draws the flagship ``D4ForkLike`` model (shearnet/core/models.py). The figure is
organised as tinted stages, left to right:

    Inputs -> D4 orbit -> shared two-branch backbone -> cross-attention fusion
           -> realign + sign-weighted Reynolds average -> heads

The orbit stage renders the eight *actual* D4 transforms of the galaxy stamp,
labelled and annotated with the spin-2 signs (w1, w2) that the Reynolds average
applies -- the construction that makes the shear output exactly equivariant.

Group convention matches ``_d4_apply`` in shearnet/core/models.py:
element ``i`` is ``R90**r . P**m`` with ``r = i % 4`` and ``m = i // 4``, and
the spin-2 signs are ``w1 = (-1)**r``, ``w2 = (-1)**(r + m)``.

Usage
-----
    python shearnet_d4_architecture.py                 # -> ../figures/shearnet_arch_d4.{pdf,png}
    python shearnet_d4_architecture.py --usetex        # real LaTeX text (needs a TeX install)
    python shearnet_d4_architecture.py --format png --dpi 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

C = {
    "stage_in": "#7E57A6",     # inputs stage tint
    "stage_orbit": "#C77B2E",  # orbit stage tint
    "stage_back": "#2E7D74",   # backbone stage tint
    "stage_fuse": "#2F7D45",   # fusion stage tint
    "stage_reyn": "#B8912A",   # Reynolds stage tint
    "stage_head": "#B03636",   # heads stage tint
    "galaxy": "#D26419",
    "psf": "#197873",
    "flow": "#2D5AA0",         # data-flow arrows
    "edge": "#1A1A2E",
    "text": "#1A1A2E",
    "muted": "#7A7A88",
    "pos": "#2F7D45",          # + sign
    "neg": "#B03636",          # - sign
}

F = {"stage": 11.0, "block": 9.0, "sub": 7.2, "tag": 6.6, "tiny": 5.8, "math": 8.5}

BOX = "round,pad=0.02,rounding_size=0.8"

# The eight D4 elements, in the model's own indexing order.
ORBIT_LABELS = [
    r"Rot $0^\circ$", r"Rot $90^\circ$", r"Rot $180^\circ$", r"Rot $270^\circ$",
    "Mirror", r"Mir$+$rot $90^\circ$", r"Mir$+$rot $180^\circ$", r"Mir$+$rot $270^\circ$",
]
W1 = [(-1.0) ** (i % 4) for i in range(8)]
W2 = [(-1.0) ** ((i % 4) + (i // 4)) for i in range(8)]


def set_style(usetex: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif" if usetex else "DejaVu Sans",
            "text.usetex": usetex,
            "mathtext.fontset": "dejavuserif" if not usetex else "cm",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------
# Synthetic stamps (illustrative renders of what the network ingests)
# --------------------------------------------------------------------------


def galaxy_stamp(n: int = 48, seed: int = 7) -> np.ndarray:
    """A sheared, off-axis exponential galaxy with faint pixel noise.

    A deliberately *asymmetric* off-centre knot is added on top of the smooth
    profile. A pure ellipse is invariant under a 180-degree rotation, so without
    the knot the eight orbit thumbnails would collapse into four identical-looking
    pairs and the figure would read as a bug; the knot makes every element of the
    orbit visually distinct.
    """
    rng = np.random.default_rng(seed)
    g = np.linspace(-2.6, 2.6, n)
    xx, yy = np.meshgrid(g, g)
    th = np.deg2rad(33.0)
    a = np.cos(th) * xx + np.sin(th) * yy
    b = -np.sin(th) * xx + np.cos(th) * yy
    r = np.sqrt((a / 1.5) ** 2 + (b / 0.66) ** 2 + 0.02)
    img = np.exp(-1.9 * r)
    # asymmetric star-forming knot, offset along the major axis
    img += 0.55 * np.exp(-(((a - 0.95) / 0.30) ** 2 + ((b - 0.16) / 0.26) ** 2))
    img += 0.030 * rng.standard_normal((n, n))
    return np.clip(img, 0, None)


def psf_stamp(n: int = 48, seed: int = 3) -> np.ndarray:
    """A compact, slightly anisotropic PSF core with a faint halo."""
    g = np.linspace(-2.6, 2.6, n)
    xx, yy = np.meshgrid(g, g)
    core = np.exp(-((xx / 0.52) ** 2 + (yy / 0.46) ** 2))
    halo = 0.18 * np.exp(-((xx / 1.5) ** 2 + (yy / 1.4) ** 2))
    return core + halo


def d4_apply(img: np.ndarray, i: int) -> np.ndarray:
    """Apply the i-th D4 element, matching ``_d4_apply`` in models.py."""
    r, m = i % 4, i // 4
    if m:
        img = np.flip(img, axis=0)
    if r:
        img = np.rot90(img, r)
    return img


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------


def stage(ax, x0, x1, y0, y1, title, color, alpha=0.055):
    """Tinted background region with a stage title along the top."""
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=1.2",
            facecolor=color, edgecolor=color, alpha=alpha, linewidth=0, zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=1.2",
            facecolor="none", edgecolor=color, alpha=0.5,
            linewidth=1.0, linestyle=(0, (5, 3)), zorder=1,
        )
    )
    ax.text(
        (x0 + x1) / 2.0, y1 - 1.9, title, ha="center", va="center",
        fontsize=F["stage"], color=color, fontweight="bold", zorder=6,
    )


def box(ax, xy, w, h, label, color, sub=None, alpha=0.17, lw=1.4, ls="-",
        fs=None, zorder=3):
    """Rounded block with bold label and optional detail line."""
    for fc, ec, a, l in ((color, color, alpha, 0), ("none", color, 1.0, lw)):
        ax.add_patch(
            FancyBboxPatch(xy, w, h, boxstyle=BOX, facecolor=fc, edgecolor=ec,
                           alpha=a, linewidth=l, linestyle=ls, zorder=zorder)
        )
    cx, cy = xy[0] + w / 2.0, xy[1] + h / 2.0
    if sub:
        ax.text(cx, cy + h * 0.17, label, ha="center", va="center",
                fontsize=fs or F["block"], color=C["text"], fontweight="bold",
                zorder=zorder + 2)
        ax.text(cx, cy - h * 0.23, sub, ha="center", va="center",
                fontsize=F["sub"], color=C["muted"], zorder=zorder + 2)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs or F["block"],
                color=C["text"], fontweight="bold", zorder=zorder + 2)


def arrow(ax, p0, p1, color=None, lw=1.5, ls="-", rad=0.0, zorder=5, ms=10):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
            linestyle=ls, color=color or C["flow"],
            connectionstyle=f"arc3,rad={rad}", shrinkA=2.5, shrinkB=2.5, zorder=zorder,
        )
    )


def imstamp(ax, cx, cy, size, img, cmap, border=None, lw=1.0, zorder=4):
    """Draw an image centred at (cx, cy) with a crisp border."""
    ax.imshow(
        img, extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2),
        cmap=cmap, zorder=zorder, interpolation="bilinear", aspect="auto",
    )
    ax.add_patch(
        plt.Rectangle((cx - size / 2, cy - size / 2), size, size, facecolor="none",
                      edgecolor=border or C["edge"], linewidth=lw, zorder=zorder + 1)
    )


def conv_stack(ax, x, y, chans, color, h0=5.2, wslab=1.5, gap=1.5, skew=0.9):
    """ForkLens-style stack of conv slabs, one per channel width.

    Each slab is a slightly skewed parallelogram so the stack reads as depth;
    the channel count is printed inside. Returns the x of the stack's right edge.
    """
    cx = x
    for k, ch in enumerate(chans):
        h = h0 + 0.9 * k
        pts = [
            (cx, y - h / 2), (cx + wslab, y - h / 2 + skew),
            (cx + wslab, y + h / 2 + skew), (cx, y + h / 2),
        ]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.30,
                             edgecolor=color, linewidth=1.3, zorder=3))
        ax.text(cx + wslab / 2, y + skew / 2, str(ch), ha="center", va="center",
                fontsize=F["tag"], color=C["text"], fontweight="bold", zorder=5)
        cx += wslab + gap
    return cx - gap


def sign_chip(ax, cx, cy, s, r=0.62):
    """Small +/- chip used to show the spin-2 weights on each orbit member."""
    col = C["pos"] if s > 0 else C["neg"]
    ax.add_patch(Circle((cx, cy), r, facecolor=col, alpha=0.20, edgecolor=col,
                        linewidth=1.0, zorder=6))
    ax.text(cx, cy + 0.02, "$+$" if s > 0 else "$-$", ha="center", va="center",
            fontsize=F["tiny"], color=col, fontweight="bold", zorder=7)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def build_figure(figsize=(19.0, 8.9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 153)
    # tight around the stage panels themselves (Y_BOT .. Y_TOP below)
    ax.set_ylim(9.0, 73.5)
    ax.axis("off")

    gal = galaxy_stamp()
    psf = psf_stamp()

    Y_TOP, Y_BOT = 72.0, 10.5      # stage extents
    Y_G, Y_P = 52.0, 22.0          # galaxy / PSF branch centre lines
    Y_MID = 37.0

    # ================= stage 1: inputs =================
    stage(ax, 1, 20, Y_BOT, Y_TOP, "Inputs", C["stage_in"])
    imstamp(ax, 10.5, Y_G, 11.0, gal, "magma")
    ax.text(10.5, Y_G + 7.6, "galaxy", ha="center", fontsize=F["block"],
            color=C["galaxy"], fontweight="bold")
    imstamp(ax, 10.5, Y_P, 11.0, psf, "cividis")
    ax.text(10.5, Y_P + 7.6, "PSF", ha="center", fontsize=F["block"],
            color=C["psf"], fontweight="bold")
    ax.text(10.5, 14.6, r"$53\times53$ px", ha="center", fontsize=F["tag"],
            color=C["muted"])

    # ================= stage 2: D4 orbit =================
    stage(ax, 22, 62, Y_BOT, Y_TOP, r"$D_4$ Orbit  (8 joint transforms)",
          C["stage_orbit"])

    cols = [28.5, 37.0, 45.5, 54.0]
    grows = [58.0, 45.0]           # galaxy orbit: m = 0 (rotations), m = 1 (mirrored)
    tsz, psz = 6.2, 3.5
    for i in range(8):
        r_i, m_i = i % 4, i // 4
        cx, cy = cols[r_i], grows[m_i]
        imstamp(ax, cx, cy, tsz, d4_apply(gal, i), "magma", lw=0.8)
        ax.text(cx, cy - tsz / 2 - 1.35, ORBIT_LABELS[i], ha="center",
                fontsize=F["tiny"], color=C["text"])
        # spin-2 weights this orbit member carries into the Reynolds average
        sign_chip(ax, cx - 1.35, cy + tsz / 2 + 1.5, W1[i])
        sign_chip(ax, cx + 1.35, cy + tsz / 2 + 1.5, W2[i])

    # the SAME eight elements act on the PSF stamp, in lock-step with the galaxy
    for i in range(8):
        cx = 27.4 + i * 4.3
        imstamp(ax, cx, 25.5, psz, d4_apply(psf, i), "cividis", lw=0.6)
    ax.text(42.0, 30.4, r"the same $g_i$ acts on the PSF stamp, in lock-step",
            ha="center", fontsize=F["tag"], color=C["muted"], style="italic")

    # sign legend, stated once
    sign_chip(ax, 27.4, 18.4, +1)
    sign_chip(ax, 30.1, 18.4, -1)
    ax.text(32.1, 18.4, r"spin-2 weights $(w_1, w_2)$ per orbit member",
            ha="left", va="center", fontsize=F["tag"], color=C["muted"])

    arrow(ax, (16.2, Y_G), (24.3, 51.5), rad=0.05, color=C["galaxy"])
    arrow(ax, (16.2, Y_P), (24.3, 25.5), rad=-0.05, color=C["psf"])

    # ================= stage 3: shared backbone =================
    stage(ax, 64, 92, Y_BOT, Y_TOP, "Shared Backbone", C["stage_back"])
    ax.text(78, 65.0, r"weight-tied across all 8 orbit members",
            ha="center", fontsize=F["tag"], color=C["muted"], style="italic")

    xg = conv_stack(ax, 68.5, Y_G, [16, 32], C["galaxy"])
    ax.text(74.5, Y_G + 6.6, "galaxy branch", ha="center", fontsize=F["sub"],
            color=C["galaxy"], fontweight="bold")
    xp = conv_stack(ax, 68.5, Y_P, [16, 32], C["psf"])
    ax.text(74.5, Y_P + 6.6, "PSF branch", ha="center", fontsize=F["sub"],
            color=C["psf"], fontweight="bold")
    ax.text(78.0, Y_G - 7.0, r"Conv $3{\times}3$ $\rightarrow$ LN $\rightarrow$ GeLU"
            "\n" r"$\rightarrow$ AvgPool $2{\times}2$", ha="center",
            fontsize=F["tiny"], color=C["muted"])
    ax.text(78.0, Y_P - 7.6, "(smooth: no ReLU,\nno max-pool)", ha="center",
            fontsize=F["tiny"], color=C["muted"])

    arrow(ax, (58.0, 51.5), (67.6, Y_G), color=C["galaxy"])
    arrow(ax, (58.0, 25.5), (67.6, Y_P), color=C["psf"])

    # ================= stage 4: fusion =================
    stage(ax, 94, 114, Y_BOT, Y_TOP, "Cross-Attention Fusion", C["stage_fuse"])
    box(ax, (97.0, 31.0), 14.0, 12.0, "transformer\nfusion", C["stage_fuse"],
        sub=r"$d=64$,  4 heads")
    ax.text(104.0, 47.8, r"galaxy $\rightarrow Q$", ha="center", fontsize=F["sub"],
            color=C["galaxy"], fontweight="bold")
    ax.text(104.0, 26.2, r"PSF $\rightarrow K,\,V$", ha="center", fontsize=F["sub"],
            color=C["psf"], fontweight="bold")
    arrow(ax, (xg + 0.9, Y_G), (104.0, 45.4), color=C["galaxy"], rad=-0.10)
    arrow(ax, (xp + 0.9, Y_P), (104.0, 28.6), color=C["psf"], rad=0.10)

    # ================= stage 5: realign + Reynolds =================
    stage(ax, 116, 139.5, Y_BOT, Y_TOP, "Reynolds Average", C["stage_reyn"])
    box(ax, (117.5, 32.0), 9.0, 10.0, r"$g_i^{-1}$", C["stage_orbit"],
        sub="realign", fs=F["math"])
    arrow(ax, (111.2, Y_MID), (117.3, Y_MID), color=C["stage_fuse"])

    ax.text(127.7, 60.0,
            r"$\Psi_c=\frac{1}{8}\sum_{i} w_c(g_i)\,g_i^{-1}F(g_i\!\cdot\!x)$",
            ha="center", fontsize=F["math"], color=C["text"])

    box(ax, (129.5, 44.5), 8.4, 7.4, r"$\Psi_1,\Psi_2$", C["stage_reyn"],
        sub="spin-2", fs=F["math"])
    box(ax, (129.5, 19.5), 8.4, 7.4, r"$\Psi_{\rm inv}$", C["stage_reyn"],
        sub="invariant", ls="--", fs=F["math"])
    # split off the realign block: signed sum upward, sign-free sum downward
    arrow(ax, (126.4, 40.2), (129.4, 46.6), color=C["stage_orbit"], rad=-0.18)
    arrow(ax, (126.4, 33.8), (129.4, 25.4), color=C["stage_orbit"], rad=0.18,
          ls="--")
    ax.text(133.7, 15.2, r"$D_4$ Gaussian window $+$ GAP", ha="center",
            fontsize=F["tiny"], color=C["muted"])

    # ================= stage 6: heads =================
    stage(ax, 141, 151.5, Y_BOT, Y_TOP, "Heads", C["stage_head"])
    box(ax, (142.3, 44.5), 8.4, 7.4, r"$g_1,g_2$", C["stage_head"],
        sub="odd MLP", fs=F["math"])
    box(ax, (142.3, 19.5), 8.4, 7.4, "hlr, flux", C["stage_head"], sub="MLP",
        ls="--", fs=F["sub"])
    arrow(ax, (138.0, 48.2), (142.1, 48.2), color=C["stage_reyn"])
    arrow(ax, (138.0, 23.2), (142.1, 23.2), color=C["stage_reyn"], ls="--")
    ax.text(146.5, 56.4, "exactly spin-2\nby construction", ha="center",
            fontsize=F["tag"], color=C["stage_head"], fontweight="bold")
    ax.text(146.5, 15.2, "auxiliary\ntargets", ha="center", fontsize=F["tiny"],
            color=C["muted"], style="italic")

    fig.tight_layout()
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--out", default=None,
                   help="output stem (no extension). Default ../figures/shearnet_arch_d4")
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--usetex", action="store_true",
                   help="render text with a real LaTeX install (falls back if absent)")
    args = p.parse_args(argv)

    set_style(args.usetex)
    stem = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "figures" / "shearnet_arch_d4"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig, _ = build_figure()
    except RuntimeError as exc:                      # usually a missing LaTeX stack
        if not args.usetex:
            raise
        print(f"LaTeX rendering failed ({exc}); falling back to mathtext.")
        set_style(False)
        fig, _ = build_figure()

    for fmt in args.format:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=args.dpi)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
