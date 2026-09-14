"""Reader for the ShearNet evaluation FITS.

One run of ``research/shear_bias/run.py`` writes a single file holding everything
the evaluation measured. This module is the one place that knows its layout, so
the figure scripts read named quantities instead of re-deriving column names.

Layout (see ``_write_evaluation_fits`` in ShearNet's ``research/shear_bias/run.py``):

``PRIMARY`` header
    Run configuration, and the timing: ``RENDER_S`` (render seconds) and
    ``INFERENC`` (inference seconds). FITS keywords are 8 characters, so the
    full key lives in each card's comment.
``TAB_P`` / ``TAB_M`` (and ``TAB_P2`` / ``TAB_M2`` for a second component)
    The +/- applied-shear populations, one row per object.
``LEAKAGE``
    The unsheared population: ``gpsf``, ``Tpsf``, ``s2n``, and per estimator
    ``e_<est>_raw_ring``, ``e_<est>_ring``, ``Rpsf_<est>_metacal``, ...
``SUMMARY``
    One row per (estimator, correction, component): ``m``, ``m_err``, ``c``,
    ``c_err``, ``R11``, ``R22``, ``n_used``.
``BINNED``
    The same, split by flux quantile; each row divides by its own within-bin
    response, so these are not the global numbers sliced up.
``LEAKSUM``
    One row per estimator: mean shape, ``R^PSF``, and whether it was applied.

The file is large (~1 GB at production ``n_obs``), so tables are read lazily and
cached, and nothing here loads a table it was not asked for.
"""

from __future__ import annotations

from pathlib import Path

import re

import numpy as np
from astropy.io import fits
from astropy.table import Table

#: Estimator names ShearNet's runner can write (``ESTIMATORS`` in run.py).
ESTIMATORS = ("shearnet", "ngmix", "anacal")

#: Display names for figures. ShearNet's flagship is the D4-equivariant model.
DISPLAY_NAME = {
    "shearnet": "ShearNet-D4",
    "ngmix": "NGmix",
    "anacal": "AnaCal",
}


class Evaluation:
    """Lazy accessor for one evaluation FITS."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"evaluation FITS not found: {self.path}")
        self._cache = {}
        with fits.open(self.path) as hdul:
            self.header = dict(hdul[0].header)
            self.hdu_names = [h.name for h in hdul]

    # -- tables ------------------------------------------------------------

    def table(self, name: str) -> Table:
        """Read one HDU as a Table, caching the result."""
        key = name.upper()
        if key not in self._cache:
            if key not in self.hdu_names:
                raise KeyError(
                    f"{self.path.name} has no {key} HDU; it holds {self.hdu_names}"
                )
            self._cache[key] = Table.read(self.path, hdu=key)
        return self._cache[key]

    @property
    def summary(self) -> Table:
        return self.table("SUMMARY")

    @property
    def binned(self) -> Table:
        return self.table("BINNED")

    @property
    def leakage(self) -> Table:
        return self.table("LEAKAGE")

    @property
    def leaksum(self) -> Table:
        return self.table("LEAKSUM")

    # -- convenience -------------------------------------------------------

    @property
    def timing(self) -> dict:
        """Render and inference wall-clock seconds from the primary header."""
        return {
            "render_seconds": self.header.get("RENDER_S"),
            "inference_seconds": self.header.get("INFERENC"),
        }

    @property
    def shape_noise_cancel(self) -> int:
        """Number of ring stations (1 = off)."""
        return int(self.header.get("SNC", 1) or 1)

    def estimators(self) -> list:
        """Estimators actually present, in a stable display order."""
        try:
            present = set(np.asarray(self.summary["estimator"]).astype(str))
        except KeyError:
            present = set()
        return [e for e in ESTIMATORS if e in present]

    def leakage_estimators(self) -> list:
        """Estimators that have usable leakage columns in this file."""
        cols = set(self.leakage.colnames)
        return [
            e for e in ESTIMATORS
            if self._leakage_shape_col(e, cols) and f"Rpsf_{e}_metacal" in cols
        ]

    #: Each named shape, as the three independent choices it actually is.
    #:
    #:   source  which image the shape was measured on
    #:             "raw"      the ORIGINAL image; metacal never touched it
    #:             "noshear"  metacal's reconvolved image, no artificial shear
    #:   rpsf    Rbar^PSF @ e_PSF subtracted
    #:   rgamma  divided by the ensemble <R^gamma>
    #:
    #: The names say WHICH corrections are in. "corrected" and "calibrated"
    #: did not distinguish two different corrections, which is how a leakage
    #: carrying only R^PSF and one carrying both ended up looking like the same
    #: number under two spellings.
    LEAKAGE_SHAPES = {
        "raw": dict(source="raw", rpsf=False, rgamma=False),
        "raw_rgamma": dict(source="raw", rpsf=False, rgamma=True),
        "noshear": dict(source="noshear", rpsf=False, rgamma=False),
        "noshear_rgamma": dict(source="noshear", rpsf=False, rgamma=True),
        "noshear_rpsf": dict(source="noshear", rpsf=True, rgamma=False),
        "noshear_rgamma_rpsf": dict(source="noshear", rpsf=True, rgamma=True),
    }

    #: (source, rpsf) -> the columns run.py actually writes, best first.
    #:
    #: run.py's names INVERT the metacal convention: it calls metacal's noshear
    #: measurement ``e_<est>_raw`` and the genuinely raw measurement
    #: ``e_<est>_original``. The FITS cannot be renamed without re-running every
    #: evaluation, so the mapping is stated once here and everything above this
    #: line speaks the standard vocabulary.
    COLUMNS_BY_SOURCE = {
        ("raw", False): ("e_{est}_original",),
        ("noshear", False): ("e_{est}_raw_ring", "e_{est}_metacal_raw_ring",
                             "e_{est}_raw"),
        ("noshear", True): ("e_{est}_ring", "e_{est}_metacal_corrected_ring",
                            "e_{est}"),
    }

    @staticmethod
    def _leakage_shape_col(estimator: str, cols) -> str | None:
        """Whether this file has any usable leakage shape for an estimator."""
        for candidate in (f"e_{estimator}_raw_ring", f"e_{estimator}_raw",
                          f"e_{estimator}_original"):
            if candidate in cols:
                return candidate
        return None

    @staticmethod
    def _ring_stations(cols, base: str) -> list:
        """Ring-station column names for one base, '' included, sorted."""
        pattern = re.compile(rf"^{re.escape(base)}(_r\d+)?$")
        return [name for name in sorted(cols) if pattern.match(name)]

    def leakage_inputs(self, estimator: str, shape: str = "raw") -> dict:
        """Arrays for ``superbit_lensing``'s PSF-leakage panel maker.

        ``shape`` names an entry of :data:`LEAKAGE_SHAPES`, which says which
        image the measurement came from and which corrections are applied.
        Returns ``e1_gal``, ``e2_gal``, ``e1_psf``, ``e2_psf``, ``r11_psf``,
        ``r22_psf`` plus ``Tpsf`` and ``s2n``, with non-finite rows dropped.
        """
        if shape not in self.LEAKAGE_SHAPES:
            raise ValueError(f"shape must be one of {sorted(self.LEAKAGE_SHAPES)}, "
                             f"got {shape!r}")
        spec = self.LEAKAGE_SHAPES[shape]
        tab = self.leakage
        cols = set(tab.colnames)

        candidates = [c.format(est=estimator)
                      for c in self.COLUMNS_BY_SOURCE[(spec["source"], spec["rpsf"])]]
        e_gal = shape_col = None
        for candidate in candidates:
            # The ring average comes FIRST. run.py writes no _ring column for
            # the raw measurement, only one per station, and the bare name is
            # the UNROTATED station -- matching it exactly would fit a single
            # station and leave the intrinsic ellipticity in, which is the one
            # thing the ring exists to remove.
            stations = self._ring_stations(cols, candidate)
            if len(stations) > 1:
                e_gal = np.stack([np.asarray(tab[n], dtype=float)
                                  for n in stations]).mean(axis=0)
                shape_col = f"{candidate} (ring of {len(stations)})"
                break
            if candidate in cols:
                e_gal = np.asarray(tab[candidate], dtype=float)
                shape_col = candidate
                break
        if e_gal is None:
            raise KeyError(
                f"LEAKAGE has no {shape!r} shape for {estimator!r}; looked for "
                + ", ".join(candidates)
            )

        rpsf_col = f"Rpsf_{estimator}_metacal"
        if rpsf_col not in cols:
            raise KeyError(f"LEAKAGE has no {rpsf_col} column")

        r_gamma = None
        if spec["rgamma"]:
            # The shear response calibrates the estimator, so it rescales the
            # leakage slope too: alpha of e/<R> is alpha(e)/<R>. ENSEMBLE
            # response, from SUMMARY -- dividing per object would inject the
            # finite-difference noise the harness keeps out of the shape.
            row = self.summary_row(estimator, "metacal", component=0)
            if row is None:
                raise KeyError(
                    f"SUMMARY has no ({estimator!r}, 'metacal') row, so "
                    "<R^gamma> is unavailable for this shape"
                )
            r_gamma = (float(row["R11"]), float(row["R22"]))
            if not all(np.isfinite(r_gamma)) or 0.0 in r_gamma:
                raise KeyError(f"<R^gamma> for {estimator!r} is {r_gamma}")
            e_gal = e_gal / np.asarray(r_gamma, dtype=float)
            shape_col += f" / <R^gamma>=({r_gamma[0]:.4f},{r_gamma[1]:.4f})"

        gpsf = np.asarray(tab["gpsf"], dtype=float)
        rpsf = np.asarray(tab[rpsf_col], dtype=float)

        out = {
            "e1_gal": e_gal[:, 0], "e2_gal": e_gal[:, 1],
            "e1_psf": gpsf[:, 0], "e2_psf": gpsf[:, 1],
            "r11_psf": rpsf[:, 0, 0], "r22_psf": rpsf[:, 1, 1],
            "Tpsf": np.asarray(tab["Tpsf"], dtype=float),
            "s2n": np.asarray(tab["s2n"], dtype=float),
            "shape_column": shape_col,
            "shape": shape,
            "r_gamma": r_gamma,
        }

        finite = np.ones(len(e_gal), dtype=bool)
        for key in ("e1_gal", "e2_gal", "e1_psf", "e2_psf", "r11_psf", "r22_psf"):
            finite &= np.isfinite(out[key])
        for key, value in list(out.items()):
            if isinstance(value, np.ndarray):
                out[key] = value[finite]
        out["n_used"] = int(finite.sum())
        out["n_dropped"] = int((~finite).sum())
        return out

    def summary_row(self, estimator: str, correction: str, component: int = 0):
        """One SUMMARY row, or ``None`` when that combination was not run."""
        tab = self.summary
        mask = (
            (np.asarray(tab["estimator"]).astype(str) == estimator)
            & (np.asarray(tab["correction"]).astype(str) == correction)
            & (np.asarray(tab["component"]).astype(int) == int(component))
        )
        return tab[mask][0] if mask.any() else None

    def corrections(self, estimator: str | None = None) -> list:
        """Correction labels present, optionally restricted to one estimator."""
        tab = self.summary
        mask = np.ones(len(tab), dtype=bool)
        if estimator is not None:
            mask = np.asarray(tab["estimator"]).astype(str) == estimator
        return sorted(set(np.asarray(tab["correction"]).astype(str)[mask]))

    def __repr__(self):
        return (
            f"<Evaluation {self.path.name}: HDUs={self.hdu_names}, "
            f"estimators={self.estimators()}, SNC={self.shape_noise_cancel}>"
        )
