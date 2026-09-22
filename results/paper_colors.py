"""Shared accessible estimator colors; component identity uses line style."""
from __future__ import annotations
import numpy as np
from matplotlib.colors import to_rgba

COLORS = {"shearnet": "#003F5C", "ngmix": "#C88700"}
COMPONENT_STYLES = {"11": ("-", "o"), "22": ("--", "s")}
RESPONSE_STYLES = {
    ("shearnet", "11"): ("-", "o"), ("shearnet", "22"): ("--", "s"),
    ("ngmix", "11"): ("-.", "^"), ("ngmix", "22"): (":", "D"),
}
HATCHES = {"shearnet": "", "ngmix": "///"}
# A hue-diverging blue / neutral / gold map with monotonic lightness.
# Unlike white-centered divergence, its two signs remain distinct in grayscale.
def srgb_to_linear(rgb):
    rgb=np.asarray(rgb)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)

def linear_to_srgb(rgb):
    rgb=np.maximum(np.asarray(rgb),0)
    return np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)

def signed_cmap():
    from matplotlib.colors import ListedColormap, to_rgb
    stops=srgb_to_linear([to_rgb(c) for c in ("#17324D","#929292","#FFE8A3")])
    t=np.linspace(0,1,256)
    values=np.stack([np.interp(t,[0,.5,1],stops[:,k]) for k in range(3)],axis=-1)
    return ListedColormap(linear_to_srgb(values),name="signed_blue_gray_gold")

PSF_ELLIPTICITY_CMAP = signed_cmap()
PSF_SIZE_CMAP = "cividis"


def recolor_artists(fig, mapping):
    """Change matching artist colors, including legend copies; retain alpha/data."""
    pairs = [(np.array(to_rgba(a))[:3], np.array(to_rgba(b))[:3])
             for a, b in mapping.items()]
    for artist in fig.findobj():
        for prop in ("color", "facecolor", "edgecolor", "markerfacecolor", "markeredgecolor"):
            get, set_ = getattr(artist, "get_"+prop, None), getattr(artist, "set_"+prop, None)
            if get is None or set_ is None:
                continue
            try:
                old = get()
                if isinstance(old, str) or np.asarray(old).ndim == 1:
                    arr = np.array(to_rgba(old), dtype=float)
                else:
                    arr = np.array(old, dtype=float)
                if not arr.size or arr.shape[-1] not in (3,4):
                    continue
                updated = arr.copy()
                for source, dest in pairs:
                    match = np.all(np.isclose(arr[..., :3], source, atol=1e-6), axis=-1)
                    updated[..., :3] = np.where(np.asarray(match)[..., None], dest, updated[..., :3])
                if not np.array_equal(arr, updated):
                    set_(updated)
            except (ValueError, TypeError, AttributeError):
                continue


def color_psf_maps(axes):
    """Keep values and clipping; use zero-centered ellipticity and sequential size."""
    from matplotlib.colors import TwoSlopeNorm
    for row in np.asarray(axes, dtype=object).reshape(-1, 3):
        for col, ax in enumerate(row):
            for im in ax.images:
                im.set_cmap(PSF_ELLIPTICITY_CMAP if col < 2 else PSF_SIZE_CMAP)
                if col < 2:
                    lo, hi = im.get_clim()
                    limit = max(abs(lo), abs(hi))
                    im.set_norm(TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
                if im.colorbar is not None:
                    im.colorbar.update_normal(im)


def style_leakage_lines(fig):
    from matplotlib.colors import to_rgb
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_linestyle() not in ("None", "", "none"):
                for est in COLORS:
                    if np.allclose(to_rgb(line.get_color()), to_rgb(COLORS[est])):
                        line.set_linestyle("-" if est=="shearnet" else "--")
                        line.set_linewidth(1.8)
