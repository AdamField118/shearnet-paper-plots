"""Reader for the evaluation catalogs ``shearnet-eval`` writes.

One file per evaluation, ``<run>/evaluations/<name>/<run_name>_<name>.fits``,
schema ``shearnet-eval`` version 1 (ShearNet ``docs/catalog.md``). It holds raw
measurements only: no response, bias, cut or correction. Everything derived
lives in :mod:`shear_stats`; this module only knows the layout.

Layout, as far as these scripts care:

* ``TRUTH``, ``STAMP``, ``SHEARNET``, ``NGMIX`` -- one row per catalog object x
  scene x ring station, all four in the same order,
  ``record_id = (scene_id * n_rotations + rotation_id) * n_objects + catalog_row``.
  So every column reshapes to ``(scene, station, object, ...)``, and the same
  ``[s, k, n]`` is the same stamp in every table.
* ``SCENES`` -- ``scene_id``, ``name``, ``g1``, ``g2``: the applied shear of each
  population (``zero``, ``g1_plus``, ``g1_minus``, ``g2_plus``, ``g2_minus``).
* ``ROTATIONS`` -- the ring stations, in degrees.
* Variant suffixes follow SuperBIT's metacal tables: ``_original`` is the stamp
  as rendered, ``_noshear``, ``_1p`` ... ``_2m_psf`` are ngmix's metacal products,
  and ShearNet is run on the very images ngmix fitted.

Columns are read through a memory map and only for the scenes asked for, so
reading the ``g1`` pair out of a 6 GB file reads 2/5 of the rows it needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from astropy.io import fits

SCHEMA = "shearnet-eval"
SCHEMA_VERSION = 1

#: The unit tests, in the paper's order.
RUNS = ("first", "second", "third", "fourth")

#: Display names in the figures.
DISPLAY_NAME = {"shearnet": "ShearNet-D4", "ngmix": "NGmix"}


class CatalogError(ValueError):
    """The file is not a catalog these scripts can read, and why."""


class Catalog:
    """Lazy, memory-mapped access to one evaluation catalog."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"evaluation catalog not found: {self.path}")
        self._hdul = fits.open(self.path, memmap=True)
        self.header = self._hdul[0].header
        schema = self.header.get("SCHEMA")
        if schema != SCHEMA:
            raise CatalogError(
                f"{self.path.name} has SCHEMA={schema!r}, not {SCHEMA!r}. Files from "
                "the old research/shear_bias/run.py harness (TAB_P/TAB_M/SUMMARY) "
                "are not read any more; evaluate the run with shearnet-eval.")
        if int(self.header.get("SCHEMAV", -1)) != SCHEMA_VERSION:
            raise CatalogError(f"{self.path.name} is schema version "
                               f"{self.header.get('SCHEMAV')}, these scripts read "
                               f"{SCHEMA_VERSION}")
        self.hdu_names = [h.name for h in self._hdul]

        scenes = self._hdul["SCENES"].data
        self.scenes: Dict[str, dict] = {
            str(name).strip(): {"id": int(i), "g1": float(g1), "g2": float(g2)}
            for i, name, g1, g2 in zip(scenes["scene_id"], scenes["name"],
                                       scenes["g1"], scenes["g2"])}
        self.rotations_deg = np.asarray(self._hdul["ROTATIONS"].data["rotation_deg"],
                                        dtype=float)
        self.n_scenes = len(self.scenes)
        self.n_rotations = len(self.rotations_deg)
        self.n_objects = int(self.header["NOBJ"])
        self.step = float(self.header["MCALSTEP"])
        expected = self.n_scenes * self.n_rotations * self.n_objects
        if int(self.header["NRECORD"]) != expected:
            raise CatalogError(f"NRECORD={self.header['NRECORD']} but SCENES x "
                               f"ROTATIONS x NOBJ = {expected}")
        self._check_row_order()

    # ------------------------------------------------------------------
    def _check_row_order(self):
        """Every reshape below assumes the documented order; check it once.

        On a sample of rows: a FITS table is stored row by row, so reading one
        whole column means reading the whole 6 GB file.
        """
        nrecord = int(self.header["NRECORD"])
        rows = np.unique(np.linspace(0, nrecord - 1, 2001).astype(int))
        for table in ("TRUTH", "STAMP", "SHEARNET", "NGMIX"):
            if table not in self.hdu_names:
                continue
            data = self._hdul[table].data
            if len(data) != nrecord:
                raise CatalogError(f"{table} has {len(data)} rows, NRECORD={nrecord}")
            rid = np.asarray(data[rows]["record_id"])
            if not np.array_equal(rid, rows):
                raise CatalogError(f"{table}.record_id is not 0..NRECORD-1 in order")

    def close(self):
        self._hdul.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    @property
    def run_name(self) -> str:
        return str(self.header.get("RUNNAME", self.path.stem))

    def has(self, table: str, column: Optional[str] = None) -> bool:
        if table not in self.hdu_names:
            return False
        return column is None or column in self._hdul[table].columns.names

    def scene_ids(self, names: Iterable[str]) -> list:
        missing = [n for n in names if n not in self.scenes]
        if missing:
            raise CatalogError(f"{self.path.name} has no scene(s) {missing}; it has "
                               f"{sorted(self.scenes)}")
        return [self.scenes[n]["id"] for n in names]

    def columns(self, table: str, names: Iterable[str], scenes: Sequence[str]) -> dict:
        """``{name: array}`` for several columns of ``table``, each shaped
        ``(scene, station, object, ...)``, native-endian.

        A FITS table is stored row by row, so reading one column of a scene
        reads every byte of that scene. This reads each scene's rows ONCE, in a
        single sequential pass, takes every requested column from them and lets
        them go, so the peak is one scene of one table.
        """
        names = list(dict.fromkeys(names))
        if not self.has(table):
            raise CatalogError(f"{self.path.name} has no {table} table "
                               "(that estimator was not measured)")
        missing = [n for n in names if not self.has(table, n)]
        if missing:
            raise CatalogError(f"{table} has no column(s) {missing}")
        per_scene = self.n_rotations * self.n_objects
        hdu = self._hdul[table]
        blocks = {name: [] for name in names}
        for sid in self.scene_ids(scenes):
            rows = np.array(hdu.data[sid * per_scene:(sid + 1) * per_scene], copy=True)
            for name in names:
                values = rows[name]
                blocks[name].append(values.astype(values.dtype.newbyteorder("=")))
            del rows
        out = {}
        for name, parts in blocks.items():
            stacked = np.stack(parts)
            out[name] = stacked.reshape((len(parts), self.n_rotations, self.n_objects)
                                        + stacked.shape[2:])
        return out

    def column(self, table: str, name: str, scenes: Sequence[str]) -> np.ndarray:
        """One column; see :meth:`columns` to read several in one pass."""
        return self.columns(table, [name], scenes)[name]

    def __repr__(self):
        return (f"<Catalog {self.path.name}: run {self.run_name}, {self.n_objects} objects "
                f"x {self.n_scenes} scenes x {self.n_rotations} stations>")


def find_run(root, name: str) -> Optional[Path]:
    """The ``default`` evaluation catalog of unit test ``name`` under ``root``.

    ``root`` can be a directory the catalogs were copied into, or ShearNet's
    ``runs/unit_tests`` directory itself, so nothing has to be copied:

        <root>/<name>.fits
        <root>/d4_unit_<name>_default.fits
        <root>/<name>/evaluations/default/<run_name>_default.fits
        <root>/unit_tests/<name>/evaluations/default/<run_name>_default.fits
    """
    root = Path(root).expanduser()
    for candidate in (root / f"{name}.fits", root / f"d4_unit_{name}_default.fits"):
        if candidate.is_file():
            return candidate
    for run_dir in (root / name, root / "unit_tests" / name):
        found = sorted((run_dir / "evaluations" / "default").glob("*_default.fits"))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise CatalogError(f"more than one default catalog in {run_dir}: {found}")
    return None
