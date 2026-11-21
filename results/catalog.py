"""Reader for the evaluation catalogs ``shearnet-eval`` writes.

A merged estimator catalog uses
schema ``shearnet-eval`` version 2 (ShearNet ``docs/catalog.md``). It holds raw
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

CFITSIO reads only the requested scenes and columns, in bounded row batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import fitsio

SCHEMA = "shearnet-eval"
SCHEMA_VERSION = 2

#: The unit tests, in the paper's order.
RUNS = ("first", "second", "third", "fourth")

#: Display names in the figures.
DISPLAY_NAME = {"shearnet": "ShearNet-D4", "ngmix": "NGmix"}


class CatalogError(ValueError):
    """The file is not a catalog these scripts can read, and why."""


class Catalog:
    """Read selected rows without mapping an entire evaluation catalog."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"evaluation catalog not found: {self.path}")
        self._hdul = fitsio.FITS(str(self.path))
        try:
            self._read_layout()
        except Exception:
            self.close()
            raise

    def _read_layout(self):
        self.header = self._hdul[0].read_header()
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
        self.hdu_names = [h.get_extname() or "PRIMARY" for h in self._hdul]

        scenes = self._hdul["SCENES"].read()
        self.scenes: Dict[str, dict] = {
            str(name).strip(): {"id": int(i), "g1": float(g1), "g2": float(g2)}
            for i, name, g1, g2 in zip(scenes["scene_id"], scenes["name"],
                                       scenes["g1"], scenes["g2"])}
        self.rotations_deg = np.asarray(self._hdul["ROTATIONS"].read_column("rotation_deg"),
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
        keys = ["record_id", "object_uid", "observation_uid"]
        truth = None
        for table in ("TRUTH", "STAMP", "SHEARNET", "NGMIX"):
            if table not in self.hdu_names and table != "TRUTH":
                continue
            hdu = self._hdul[table]
            if hdu.get_nrows() != nrecord:
                raise CatalogError(f"{table} has {hdu.get_nrows()} rows, NRECORD={nrecord}")
            for key in keys:
                if key not in hdu.get_colnames():
                    raise CatalogError(f"{table} is missing {key}")
            sampled = hdu.read(columns=keys, rows=rows)
            if not np.array_equal(sampled["record_id"], rows):
                raise CatalogError(f"{table}.record_id is not 0..NRECORD-1 in order")
            if table == "TRUTH":
                truth = sampled
            for key in keys[1:]:
                if np.any(sampled[key] == "") or not np.array_equal(sampled[key], truth[key]):
                    raise CatalogError(f"{table}.{key} disagrees with TRUTH")
        for key in ("input_fingerprint", "metacal_fingerprint"):
            if not self.has("STAMP", key):
                raise CatalogError(f"STAMP is missing {key}")

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
        return column is None or column in self._hdul[table].get_colnames()

    def scene_ids(self, names: Iterable[str]) -> list:
        missing = [n for n in names if n not in self.scenes]
        if missing:
            raise CatalogError(f"{self.path.name} has no scene(s) {missing}; it has "
                               f"{sorted(self.scenes)}")
        return [self.scenes[n]["id"] for n in names]

    def columns(self, table: str, names: Iterable[str], scenes: Sequence[str],
                objects: slice = slice(None)) -> dict:
        """``{name: array}`` for several columns of ``table``, each shaped
        ``(scene, station, object, ...)``, native-endian.

        Read short consecutive runs of FITS rows before gathering their columns.
        An object slice selects the same catalog rows at every ring station.
        Only the requested columns survive each read.
        """
        names = list(dict.fromkeys(names))
        if not self.has(table):
            raise CatalogError(f"{self.path.name} has no {table} table "
                               "(that estimator was not measured)")
        missing = [n for n in names if not self.has(table, n)]
        if missing:
            raise CatalogError(f"{table} has no column(s) {missing}")
        start, stop, step = objects.indices(self.n_objects)
        if step != 1 or stop <= start:
            raise ValueError("expected a nonempty contiguous range of objects")
        data = self._hdul[table]
        ids = self.scene_ids(scenes)
        out = {}
        for si, sid in enumerate(ids):
            for rot in range(self.n_rotations):
                origin = (sid * self.n_rotations + rot) * self.n_objects
                for lo in range(start, stop, 4096):
                    hi = min(lo + 4096, stop)
                    rows = data.read(columns=names, rows=np.arange(origin + lo, origin + hi))
                    for name in names:
                        v = rows[name]
                        if name not in out:
                            shape = (len(ids), self.n_rotations, stop - start) + v.shape[1:]
                            out[name] = np.empty(shape, dtype=v.dtype.newbyteorder("="))
                        out[name][si, rot, lo - start:hi - start] = v
        return out

    def key_values(self, table: str) -> dict:
        """A ``key``/``value`` table (``TIMING``, ``CONFIG`` ...) as a dict; values
        that are JSON are decoded, the rest are kept as text."""
        if not self.has(table):
            raise CatalogError(f"{self.path.name} has no {table} table")
        out = {}
        for row in self._hdul[table].read(columns=["key", "value"]):
            key, value = row["key"], row["value"]
            try:
                out[str(key)] = json.loads(value)
            except ValueError:
                out[str(key)] = str(value)
        return out

    def column(self, table: str, name: str, scenes: Sequence[str]) -> np.ndarray:
        """One column; see :meth:`columns` to read several in one pass."""
        return self.columns(table, [name], scenes)[name]

    def __repr__(self):
        return (f"<Catalog {self.path.name}: run {self.run_name}, {self.n_objects} objects "
                f"x {self.n_scenes} scenes x {self.n_rotations} stations>")


def find_run(root, name: str) -> Optional[Path]:
    """The merged catalog of unit test ``name`` under ``root``.

    ``root`` can be a directory the catalogs were copied into, or ShearNet's
    ``runs/unit_tests`` directory itself, so nothing has to be copied:

        <root>/<name>.fits
        <root>/<name>/generation-*/merged.fits
        <root>/unit_tests/<name>/generation-*/merged.fits

    Each ShearNet submission is a new generation; the newest one whose merge
    finished (``merge.completed``) is used.
    """
    root = Path(root).expanduser()
    if (root / f"{name}.fits").is_file():
        return root / f"{name}.fits"
    for run_dir in (root / name, root / "unit_tests" / name):
        finished = [g for g in run_dir.glob("generation-*")
                    if (g / "merged.fits").is_file() and (g / "merge.completed").is_file()]
        if finished:
            newest = max(finished, key=lambda g: (g / "merge.completed").stat().st_mtime)
            return newest / "merged.fits"
    return None
