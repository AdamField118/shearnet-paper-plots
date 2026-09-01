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

    @staticmethod
    def _leakage_shape_col(estimator: str, cols) -> str | None:
        """Pick the galaxy-shape column the leakage panels should read.

        ``e_<est>_raw_ring`` is the ring-averaged raw shape and is what the
        ShearNet config documents as the column for the leakage panels and the
        alpha-vs-size regression: the ring average cancels intrinsic ellipticity,
        and *raw* keeps the PSF-response correction out, since that correction is
        exactly the thing the leakage slope is measuring. Fall back to the
        non-ring column when the run had ``shape_noise_cancel`` off.
        """
        for candidate in (f"e_{estimator}_raw_ring", f"e_{estimator}_raw"):
            if candidate in cols:
                return candidate
        return None

    def leakage_inputs(self, estimator: str) -> dict:
        """Arrays needed by ``superbit_lensing``'s PSF-leakage panel maker.

        Returns ``e1_gal``, ``e2_gal``, ``e1_psf``, ``e2_psf``, ``r11_psf``,
        ``r22_psf`` plus ``Tpsf`` and ``s2n``, with non-finite rows dropped.
        """
        tab = self.leakage
        cols = set(tab.colnames)

        shape_col = self._leakage_shape_col(estimator, cols)
        if shape_col is None:
            raise KeyError(
                f"LEAKAGE has no shape column for {estimator!r}; "
                f"looked for e_{estimator}_raw_ring and e_{estimator}_raw"
            )
        rpsf_col = f"Rpsf_{estimator}_metacal"
        if rpsf_col not in cols:
            raise KeyError(f"LEAKAGE has no {rpsf_col} column")

        e_gal = np.asarray(tab[shape_col], dtype=float)
        gpsf = np.asarray(tab["gpsf"], dtype=float)
        rpsf = np.asarray(tab[rpsf_col], dtype=float)

        out = {
            "e1_gal": e_gal[:, 0], "e2_gal": e_gal[:, 1],
            "e1_psf": gpsf[:, 0], "e2_psf": gpsf[:, 1],
            "r11_psf": rpsf[:, 0, 0], "r22_psf": rpsf[:, 1, 1],
            "Tpsf": np.asarray(tab["Tpsf"], dtype=float),
            "s2n": np.asarray(tab["s2n"], dtype=float),
            "shape_column": shape_col,
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
