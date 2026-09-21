"""Manuscript notation applied to existing Matplotlib artists.

These helpers change labels only. Upstream SuperBIT routines still calculate
and draw the data, regressions, errors, and PSF maps.
"""
from __future__ import annotations

import numpy as np


def ellipticity_truth(component):
    """FITS components are zero-based; manuscript indices are one-based."""
    if component not in (0, 1):
        raise ValueError("component must be 0 or 1")
    return rf"$e_{component + 1}^{{\rm true}}$"


def label_prediction_residuals(fig, axes, compare, reference, coefficients, component):
    """Use the returned upstream fit, without fitting or altering data again."""
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if len(axes) != len(compare):
        raise ValueError("unexpected number of prediction-residual panels")
    index = component + 1
    for ax, estimator in zip(axes, compare):
        fit = coefficients[f"{estimator}_vs_{reference}"]
        values, errors = fit["quadratic"], fit["errors"]
        annotations = [t for t in ax.texts if t.get_text().startswith("m = ")]
        if len(annotations) != 1:
            raise ValueError("upstream quadratic-fit annotation changed; inspect plot_comparison")
        annotations[0].set_text("\n".join(
            f"{name} = {values[k]:.3f} ± {errors[k]:.3f}"
            for name, k in (("linear", 1), ("quadratic", 0), ("constant", 2))
        ))
        annotations[0].set_fontfamily("serif")
        annotations[0].set_bbox(None)
        ax.set_xlabel(ellipticity_truth(component))
        ax.set_title({"shearnet": "ShearNet", "ngmix": "ngmix"}.get(estimator, estimator))
    axes[0].set_ylabel(rf"$e_{index}-e_{index}^{{\rm true}}$")
    fig.tight_layout()


def label_psf_leakage(fig, *, shapes=("raw",)):
    """Label centered medians in ellipticity or response-corrected shear units.

    The current defaults divide both estimators by their ensemble shear response;
    --shape raw preserves the statistic in the archived manuscript figure.
    """
    if len(fig.axes) != 2:
        raise ValueError("expected the two PSF-ellipticity component panels")
    corrected = ["rgamma" in shape for shape in shapes]
    for i, ax in enumerate(fig.axes, 1):
        if all(corrected):
            symbol = rf"\widehat{{\gamma}}_{i}"
        elif not any(corrected):
            symbol = rf"e_{i}"
        else:
            ax.set_ylabel(f"median centered shape, component {i}")
            continue
        ax.set_ylabel(rf"$\mathrm{{median}}({symbol}-\langle {symbol}\rangle)$")
    fig.tight_layout()


def label_psf_properties(axes, *, residual_only=False):
    """Name the PSF quantities, including residual-only exports."""
    axes = np.asarray(axes, dtype=object).reshape(-1, 3)
    prefix = r"\delta " if residual_only else ""
    for ax, symbol in zip(axes[0], (r"e_1^{\rm PSF}", r"e_2^{\rm PSF}", r"T_{\rm PSF}")):
        ax.set_title(f"${prefix}{symbol}$", pad=10)
