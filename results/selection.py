"""The sample cut, taken from superbit-lensing rather than invented here.

The UT4 ensemble shows ngmix metacal at m1 = +0.165, and the per-bin breakdown
in :mod:`diagnose_metacal` puts all of it in the smallest galaxies: the bin at
hlr = 0.000-0.106" (against a 0.5" FWHM PSF) has m_metacal = +0.70, while the
0.418-1.000" bin has +0.013. Those objects are not resolved, so metacal's
deconvolve/reconvolve has nothing to work with and its noshear shape comes back
inflated. No metacal analysis measures them; SuperBIT's does not either.

SuperBIT's cut lives in

    superbit_lensing.metacalibration.calibrator.DEFUALT_SELECTION_CUT

and is applied in ``superbit_lensing.diagnostics.compute_metacal_quantities``.
The dictionary is imported here verbatim -- the thresholds are theirs, not ours.
The mask itself has to be rebuilt because that function reads ngmix-pipeline
column names (``T_noshear``, ``Tpsf_noshear``, ``s2n_noshear``, ``redshift``)
that a ShearNet evaluation FITS does not have; :data:`COLUMNS` is the whole of
the translation, and the boolean expression below is a line-for-line copy of
their ``noshear_mask``.

THE SELECTION RESPONSE IS MISSING, AND THAT MATTERS
---------------------------------------------------
``compute_metacal_quantities`` does not just cut. It repeats the identical cut
on each sheared counterpart (``T_1p``/``s2n_1p``, ``T_1m``/``s2n_1m``, ...) and
forms

    R11_S = (<g_noshear>[sel 1p] - <g_noshear>[sel 1m]) / (2 * mcal_shear)

then calibrates by ``R_gamma + R_S``. That term exists because a cut on a
*measured* quantity is itself shear-dependent: applying it changes which
galaxies are in the sample differently for a positive and a negative shear, and
the difference is a real multiplicative bias of order a percent.

A ShearNet evaluation FITS stores ``T_ngmix`` and ``s2n_ngmix`` from the
**noshear** fit only -- the per-step ``T_1p``, ``s2n_1p`` are never written --
so R_S **cannot be computed from the files we have**. :func:`superbit_mask`
therefore gives the sample SuperBIT measures but leaves it uncalibrated for
selection, which is a percent-level bias sitting under a 16% one.

That is why :func:`truth_mask` exists. A cut on ``hlr_th``, the half-light
radius that was drawn and rendered, cannot depend on the applied shear, so its
selection response is zero by construction and no correction is owed. It is
only available in a simulation -- which is exactly what these runs are.

    from selection import superbit_mask, truth_mask, superbit_cuts
"""

from __future__ import annotations

import numpy as np

def superbit_cuts() -> dict:
    """SuperBIT's thresholds, imported so the numbers have one definition.

    ``min_Tpsf`` is a resolution floor, not a size floor: :func:`superbit_mask`
    reads it as ``T >= min_Tpsf * Tpsf``, so 1.0 means "at least as large as
    the PSF".

    Imported on demand rather than at module scope so that :func:`truth_mask`,
    which owes SuperBIT nothing, still works in an environment without it.
    """
    from superbit_lensing.metacalibration.calibrator import DEFUALT_SELECTION_CUT

    return dict(DEFUALT_SELECTION_CUT)

#: ShearNet column <- SuperBIT column. ``T`` and ``s2n`` are ngmix's, for both
#: estimators, deliberately: the cut defines the *sample*, and a sample that
#: changed with whichever network is being scored would make the two columns of
#: the bias table incomparable. SuperBIT's own cut is ngmix-based for the same
#: reason. ``Tpsf`` is the adaptive-moment PSF size the harness measures, in
#: arcsec^2 like ngmix's ``T`` (``run.py`` builds the jacobian with the pixel
#: scale), so the ratio is dimensionless and the 1.0 threshold transfers.
COLUMNS = {
    "T_noshear": "T_ngmix",
    "Tpsf_noshear": "Tpsf",
    "s2n_noshear": "s2n_ngmix",
    "gpsf_noshear": "gpsf",
}


class MissingColumn(KeyError):
    """A column the cut needs is not in this table."""


def _column(table, superbit_name: str, fallback: str = None) -> np.ndarray:
    name = COLUMNS[superbit_name]
    if name in table.colnames:
        return np.asarray(table[name], dtype=float)
    if fallback and fallback in table.colnames:
        return np.asarray(table[fallback], dtype=float)
    raise MissingColumn(
        f"{name!r} (SuperBIT's {superbit_name!r}) is not in this table; "
        f"present: {', '.join(sorted(table.colnames)[:12])}..."
    )


def superbit_mask(table, cuts: dict = None, *, quiet: bool = False) -> np.ndarray:
    """SuperBIT's ``noshear_mask``, on a ShearNet per-object table.

    A line-for-line copy of the boolean in
    ``superbit_lensing.diagnostics.compute_metacal_quantities``, minus the
    ``redshift`` term (these are single-plane simulations with no cluster, and
    SuperBIT's own ``min_redshift`` is 0 when ``cluster_redshift`` is None, so
    the term is a no-op rather than an omission).

    Read the module docstring before using this for a published number: it
    returns SuperBIT's sample but not SuperBIT's selection response.
    """
    cuts = superbit_cuts() if cuts is None else dict(cuts)

    T = _column(table, "T_noshear")
    Tpsf = _column(table, "Tpsf_noshear")
    s2n = _column(table, "s2n_noshear", fallback="s2n")

    with np.errstate(invalid="ignore"):
        mask = (
            (T >= cuts["min_Tpsf"] * Tpsf)
            & (T < cuts["max_T"])
            & (T >= cuts["min_T"])
            & (s2n > cuts["min_sn"])
            & (s2n < cuts["max_sn"])
        )

    max_gpsf = cuts.get("max_gpsf")
    if max_gpsf is not None and COLUMNS["gpsf_noshear"] in table.colnames:
        gpsf = np.asarray(table[COLUMNS["gpsf_noshear"]], dtype=float)
        # Their form: a box on each component, not a bound on the modulus.
        mask &= (
            (gpsf[:, 1] > -max_gpsf) & (gpsf[:, 1] < max_gpsf)
            & (gpsf[:, 0] > -max_gpsf) & (gpsf[:, 0] < max_gpsf)
        )

    # calibrator.Calibrator.__init__ drops non-finite and non-positive
    # T/Tpsf and s2n before anything else; the failed fits in these runs come
    # back with T/Tpsf down to -0.96, so this is not academic.
    with np.errstate(invalid="ignore", divide="ignore"):
        mask &= np.isfinite(T / Tpsf) & np.isfinite(s2n) & (T / Tpsf > 0) & (s2n > 0)

    if not quiet:
        print(f"# SuperBIT cut: T/Tpsf >= {cuts['min_Tpsf']}, "
              f"{cuts['min_sn']} < S/N < {cuts['max_sn']}, "
              f"{cuts['min_T']} <= T < {cuts['max_T']}"
              + (f", |gpsf| < {max_gpsf}" if max_gpsf is not None else "")
              + f"  ->  {mask.sum()} of {len(mask)} kept")
    return mask


def truth_mask(table, *, min_hlr: float = None, min_resolution: float = None,
               psf_fwhm: float = None, quiet: bool = False) -> np.ndarray:
    """A cut on rendered truth, whose selection response is zero.

    ``hlr_th`` is the half-light radius the simulation drew and rendered. It
    does not depend on the shear that was applied, so selecting on it selects
    the same galaxies in the plus and minus populations and contributes no
    multiplicative bias -- unlike :func:`superbit_mask`, which owes an R_S that
    these files cannot supply.

    ``min_resolution`` is expressed in units of the PSF half-width at half
    maximum (``psf_fwhm / 2``), so ``min_resolution=1`` keeps galaxies at least
    as large as the PSF and is the truth-space analogue of SuperBIT's
    ``min_Tpsf = 1.0``.
    """
    if "hlr_th" not in table.colnames:
        raise MissingColumn("'hlr_th' is absent; this run wrote no truth radii")
    hlr = np.asarray(table["hlr_th"], dtype=float)

    mask = np.isfinite(hlr)
    if min_hlr is not None:
        mask &= hlr >= min_hlr
    if min_resolution is not None:
        if psf_fwhm is None:
            raise ValueError("min_resolution needs psf_fwhm to convert to arcsec")
        mask &= hlr >= min_resolution * (psf_fwhm / 2.0)

    if not quiet:
        terms = []
        if min_hlr is not None:
            terms.append(f"hlr >= {min_hlr}\"")
        if min_resolution is not None:
            terms.append(f"hlr >= {min_resolution} x (PSF FWHM/2) "
                         f"= {min_resolution * psf_fwhm / 2.0:.3f}\"")
        print(f"# truth cut: {', '.join(terms) or 'finite hlr only'}"
              f"  ->  {mask.sum()} of {len(mask)} kept")
    return mask


def paired_mask(plus, minus, which: str, **kwargs) -> np.ndarray:
    """One mask for both populations of a paired run.

    The plus and minus tables hold the same galaxies under opposite shears, and
    the paired estimator differences them row by row. A mask built on only one
    of them, or two different masks, would difference galaxies against
    different galaxies. Both are required to pass.
    """
    builders = {"superbit": superbit_mask, "truth": truth_mask}
    if which not in builders:
        raise ValueError(f"unknown cut {which!r}; choose from {sorted(builders)}")
    build = builders[which]
    return build(plus, **kwargs) & build(minus, quiet=True, **{
        k: v for k, v in kwargs.items() if k != "quiet"})
