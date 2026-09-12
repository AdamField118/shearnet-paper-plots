"""The numbers the paper's tables quote, derived from one evaluation FITS.

Every table cell in the paper is one of four quantities per (run, estimator):
``m1``, ``c2``, ``|alpha|`` and the shape noise. This module produces all four
from a single file, so a table script decides layout and nothing else.

WHY c2 IS RECOMPUTED RATHER THAN READ FROM SUMMARY
--------------------------------------------------
``SUMMARY`` carries a column called ``c``, and it is NOT the paper's ``c2``
under the configuration these runs used.

``paired_bias`` supports two conventions:

* ``'shearnet'`` -- the raw mean shape in the component that was **not**
  sheared. For a g1-sheared population that is the orthogonal prediction, which
  is exactly the paper's c2.
* ``'lin2026'`` -- Eq. 23 of Lin et al. (2026): the mean shape in the
  **sheared** component, divided by the mean response. For a g1-sheared
  population that is c1.

The fiducial config sets ``c_convention: lin2026``. So ``SUMMARY['c']`` at
component 0 is **c1**, while the paper's sentence is "for a population sheared
along g1 ... the mean orthogonal prediction <g2> measures c2". Reading the
column whose name matches the symbol would put the wrong number in the table
with nothing to indicate it.

So c2 is computed here from the per-object shapes, ring-averaged, in the
orthogonal component. ``m1`` is convention-independent and is read from SUMMARY
directly; :func:`check_m1_reproduces` re-derives it from the same per-object
columns as a cross-check that the file explains its own summary row.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from psf_leakage import panel_fits_for

#: Matches the harness's own ``n_jackknife``, so an error bar derived here means
#: the same thing as one read from SUMMARY.
DEFAULT_NJACK = 20

#: The correction whose m and c the paper reports, per estimator. Both are
#: 'metacal' because the fiducial config sets shearnet_metacal: true, so both
#: estimators measure the same nine reconvolved products and both divide by
#: their own metacal R^gamma.
REPORTED_CORRECTION = {"shearnet": "metacal", "ngmix": "metacal", "anacal": "anacal"}

#: Shape columns to try, best first. The metacal-corrected prediction is what
#: the reported m and c divide; the others are fallbacks for runs that did not
#: measure it.
_SHAPE_CANDIDATES = ("e_{est}_metacal_corrected", "e_{est}_metacal", "e_{est}")


class MissingQuantity(Exception):
    """Raised when a file cannot supply a number, with the reason."""


def _station_suffixes(colnames, base: str) -> list:
    """Ring-station suffixes present for a base column, '' included.

    The harness writes ``<col>``, ``<col>_r45``, ``<col>_r90``, ... when
    shape_noise_cancel is on. Discovering them from the column names keeps this
    working whatever the station scheme is, which is the same reason the
    slimming policy matches on patterns rather than on a hard-coded list.
    """
    pattern = re.compile(rf"^{re.escape(base)}(_r\d+)?$")
    return sorted({m.group(1) or "" for m in
                   (pattern.match(name) for name in colnames) if m})


def _ring_mean(table, base: str) -> np.ndarray:
    """Mean over ring stations of a per-object vector column.

    The ring average is what the SUMMARY rows divide, so anything derived here
    has to average the same way or it is answering a different question.
    """
    suffixes = _station_suffixes(table.colnames, base)
    if not suffixes:
        raise MissingQuantity(f"no column {base!r} (or ring stations of it)")
    stacked = np.stack([np.asarray(table[base + s], dtype=float) for s in suffixes])
    return stacked.mean(axis=0)


def _shape_column(table, estimator: str) -> str:
    for template in _SHAPE_CANDIDATES:
        base = template.format(est=estimator)
        if _station_suffixes(table.colnames, base):
            return base
    raise MissingQuantity(
        f"no shape column for {estimator!r}; tried "
        + ", ".join(t.format(est=estimator) for t in _SHAPE_CANDIDATES)
    )


def _pair_tables(evaluation, component: int):
    """``(plus, minus)`` for one sheared component, or raise.

    The first measured component keeps the historical TAB_P/TAB_M names; a
    second adds TAB_P2/TAB_M2.
    """
    tag = "" if component == 0 else str(component + 1)
    names = (f"TAB_P{tag}", f"TAB_M{tag}")
    for name in names:
        if name not in evaluation.hdu_names:
            raise MissingQuantity(
                f"{name} is absent -- the run wrote no per-object rows for "
                f"component {component} (catalog_level: summary?)"
            )
    return evaluation.table(names[0]), evaluation.table(names[1])


def _jackknife_error(values: np.ndarray, njack: int = DEFAULT_NJACK) -> float:
    """Delete-one-block jackknife error on the mean of ``values``."""
    n = len(values)
    njack = max(2, min(int(njack), n))
    edges = np.linspace(0, n, njack + 1).astype(int)
    total = values.sum()
    samples = []
    for i in range(njack):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        samples.append((total - values[lo:hi].sum()) / (n - (hi - lo)))
    samples = np.asarray(samples)
    nb = len(samples)
    return float(np.sqrt((nb - 1) / nb * np.sum((samples - samples.mean()) ** 2)))


def c2_orthogonal(evaluation, estimator: str, njack: int = DEFAULT_NJACK) -> tuple:
    """The paper's c2: the mean orthogonal prediction on the g1-sheared pair.

    Averaged over the +gamma and -gamma populations and over the ring stations.
    Both averages matter: the +/- mean is what removes the applied shear's own
    contribution from the orthogonal component, and the ring average is what
    cancels intrinsic ellipticity.

    Returns ``(c2, c2_err)``.
    """
    plus, minus = _pair_tables(evaluation, component=0)
    base = _shape_column(plus, estimator)
    e_plus = _ring_mean(plus, base)[:, 1]
    e_minus = _ring_mean(minus, base)[:, 1]
    combined = 0.5 * (e_plus + e_minus)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        raise MissingQuantity(f"every {base} row is non-finite for {estimator!r}")
    return float(combined.mean()), _jackknife_error(combined, njack)


def shape_noise(evaluation, estimator: str) -> tuple:
    """Per-component dispersion of the recovered shear, ``(sigma_g1, sigma_g2)``.

    Measured on the UNROTATED station only, not the ring average. The ring
    exists precisely to cancel intrinsic ellipticity, so a ring-averaged
    dispersion tends to the measurement noise rather than to the shape noise --
    on a perfect ring it goes to zero. The paper's column is the scatter of an
    individual recovered shear, which is the single-station number (order 0.25
    for a realistic population).
    """
    plus, _ = _pair_tables(evaluation, component=0)
    base = _shape_column(plus, estimator)
    if base not in plus.colnames:
        raise MissingQuantity(f"no unrotated station for {base!r}")
    shapes = np.asarray(plus[base], dtype=float)
    finite = np.isfinite(shapes).all(axis=1)
    if not finite.any():
        raise MissingQuantity(f"no finite {base} rows for {estimator!r}")
    return tuple(float(v) for v in shapes[finite].std(axis=0))


def m1_from_summary(evaluation, estimator: str, correction: Optional[str] = None) -> tuple:
    """``(m1, m1_err)`` from SUMMARY. Convention-independent, unlike c."""
    correction = correction or REPORTED_CORRECTION.get(estimator, "metacal")
    row = evaluation.summary_row(estimator, correction, component=0)
    if row is None:
        available = evaluation.corrections(estimator)
        raise MissingQuantity(
            f"SUMMARY has no ({estimator}, {correction}, component 0) row; "
            f"corrections present for {estimator!r}: {available or 'none'}"
        )
    return float(row["m"]), float(row["m_err"])


def check_m1_reproduces(evaluation, estimator: str, correction=None, n_sigma=5.0):
    """Re-derive m1 from the per-object columns and compare against SUMMARY.

    The harness guarantees the file explains its own summary row. If this
    disagrees, either the wrong shape/response column pair is being read here or
    the file is not what it claims, and both are worth knowing before a number
    reaches a table.

    Agreement is judged against SUMMARY's own ``m_err`` rather than a fixed
    relative tolerance: the two estimates share the same sample, but not the
    same arithmetic (SUMMARY jackknifes a ratio of means; this is a plain ratio
    of means), so they differ at the level of the sampling error and a fixed
    rtol either fires constantly on a small m or never fires at all on a large
    one.

    Returns ``(summary_m1, recomputed_m1, agrees)``; recomputed is NaN when the
    per-object columns are absent (catalog_level: summary).
    """
    correction = correction or REPORTED_CORRECTION.get(estimator, "metacal")
    summary_m1, summary_err = m1_from_summary(evaluation, estimator, correction)
    try:
        plus, minus = _pair_tables(evaluation, component=0)
        base = _shape_column(plus, estimator)
        response_base = (f"Rgamma_{estimator}_{correction}"
                         if _station_suffixes(plus.colnames, f"Rgamma_{estimator}_{correction}")
                         else f"R_{estimator}_{correction}")
        e_plus = _ring_mean(plus, base)[:, 0]
        e_minus = _ring_mean(minus, base)[:, 0]
        r_plus = _ring_mean(plus, response_base)[:, 0, 0]
        r_minus = _ring_mean(minus, response_base)[:, 0, 0]
        good = np.isfinite(e_plus) & np.isfinite(e_minus) & np.isfinite(r_plus) & np.isfinite(r_minus)
        shear = float(evaluation.header.get("SHEAR_TR", 0.01))
        numerator = np.mean(0.5 * (e_plus[good] - e_minus[good]))
        denominator = np.mean(0.5 * (r_plus[good] + r_minus[good]))
        recomputed = numerator / denominator / shear - 1.0
    except (MissingQuantity, KeyError):
        return summary_m1, float("nan"), None
    tolerance = n_sigma * max(summary_err, 1e-9)
    agrees = bool(abs(recomputed - summary_m1) <= tolerance)
    return summary_m1, float(recomputed), agrees


def _combine_alpha(a1, a1_err, a2, a2_err) -> tuple:
    """The single ``|alpha|`` an ablation row reports, from the two components.

    The tables have one leakage column, not two. Both components measure the
    same property, so this is the inverse-variance mean of |alpha_1| and
    |alpha_2|. The per-component values stay on the result for the figure, which
    annotates them separately.
    """
    values = np.array([abs(float(a1)), abs(float(a2))])
    errors = np.array([float(a1_err), float(a2_err)])
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        return float(values.mean()), float("nan")
    weights = 1.0 / errors ** 2
    return (float(np.sum(weights * values) / np.sum(weights)),
            float(np.sqrt(1.0 / np.sum(weights))))


def run_numbers(evaluation, estimator: str, njack: int = DEFAULT_NJACK,
                *, panel_fits_out=None, njack_leakage: int = 30) -> dict:
    """Every number a table row needs, with per-field failures kept local.

    ``panel_fits_out`` is where superbit's panel-data FITS is written; it
    defaults to a scratch file, since the tables want the coefficients rather
    than the panel data. ``njack_leakage`` is 30 to match
    ``PSFLeakagePanelMaker``'s own default rather than the harness's 20.

    A missing quantity does not abort the row: the key is set to ``None`` and
    ``problems`` records why, so a partially-complete run still fills the cells
    it can. Half a row with the gap named beats no row.
    """
    out = {"estimator": estimator, "problems": {}}
    if panel_fits_out is None:
        panel_fits_out = Path(tempfile.gettempdir()) / f"panels_{estimator}.fits"

    for key, getter in (
        ("m1", lambda: m1_from_summary(evaluation, estimator)),
        ("c2", lambda: c2_orthogonal(evaluation, estimator, njack)),
    ):
        try:
            value, error = getter()
            out[key], out[f"{key}_err"] = value, error
        except MissingQuantity as exc:
            out[key] = out[f"{key}_err"] = None
            out["problems"][key] = str(exc)

    try:
        out["shape_noise"] = shape_noise(evaluation, estimator)
    except MissingQuantity as exc:
        out["shape_noise"] = None
        out["problems"]["shape_noise"] = str(exc)

    # alpha comes from superbit_lensing's PSFLeakagePanelMaker, via the same
    # panel_fits_for() the figure script uses. The collaboration's binning,
    # jackknife and slope convention are what the paper's leakage number means,
    # so this must be their fit and not a second one that happens to agree.
    try:
        a1, a1_err, a2, a2_err = panel_fits_for(
            evaluation, estimator, panel_fits_out, njac=njack_leakage)
        out["alpha1"], out["alpha1_err"] = float(a1), float(a1_err)
        out["alpha2"], out["alpha2_err"] = float(a2), float(a2_err)
        out["alpha"], out["alpha_err"] = _combine_alpha(a1, a1_err, a2, a2_err)
    except Exception as exc:  # superbit raises several types; the reason is what matters
        out["alpha1"] = out["alpha2"] = out["alpha"] = out["alpha_err"] = None
        out["problems"]["alpha"] = f"{type(exc).__name__}: {exc}"

    try:
        summary_m1, recomputed, agrees = check_m1_reproduces(evaluation, estimator)
        out["m1_recomputed"], out["m1_reproduces"] = recomputed, agrees
        if agrees is False:
            out["problems"]["m1_reproduces"] = (
                f"SUMMARY m1={summary_m1:+.6g} but the per-object columns give "
                f"{recomputed:+.6g}. One of them is not measuring what its name says."
            )
    except MissingQuantity:
        out["m1_recomputed"], out["m1_reproduces"] = float("nan"), None

    return out
