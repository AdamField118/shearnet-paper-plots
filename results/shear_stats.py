"""Selection, responses and shear bias, from one ``shearnet-eval`` catalog.

Every number in Table 2 and Figures 4 and 5 comes through here, so the rules
are written down once.

THE SAMPLE CUT (``PAPER_CUT``)
------------------------------
On **ngmix's own Gaussian fit** of each metacal product ``t``::

    flags_t == 0,   T_t / Tpsf_t > 1,   s2n_t > 10

``T_t``, ``Tpsf_t``, ``s2n_t``, ``flags_t`` are the ``NGMIX`` columns with that
suffix: ``T`` is the fitted pre-PSF galaxy size, ``Tpsf`` is ngmix's fit of the
PSF *that product was fitted with* (for metacal, the dilated reconvolution
PSF -- the ``Tpsf_noshear`` SuperBIT divides by), ``s2n`` is ngmix's S/N. Never
the truth radius, never the stamp S/N, never a catalog proxy.

The **same ngmix-defined sample** is used for both estimators, so ShearNet and
ngmix are compared on identical galaxies. A record also has to be measurable by
both (every flag zero, every shape and response finite), which removes nothing
in the paper runs.

WHERE THE CUT ACTS
------------------
Only where m, c and R are formed, per applied-shear population, exactly as in
SuperBIT's ``compute_R_S``:

* the **sample** is the ``noshear`` selection of that population;
* the **shear response** ``<R^gamma>`` is averaged over that sample;
* the **selection response** is metacal's::

      R^S[:, j] = ( <e>_{S(jp)} - <e>_{S(jm)} ) / (2 step)

  where ``S(1p)`` is the cut applied to the ``1p`` product's own fit, and ``e``
  is the estimator's *unsheared* shape averaged over that selection;
* the response that calibrates the shear is ``R = <R^gamma> + R^S``.

Each population (``g1_plus``, ``g1_minus``) is selected on its own, as a survey
would be, so the cut is not forced to be even in the shear; ``R^S`` is what
corrects for that. Every ring station is a record of its own: the means run
over all selected stations, which is the paper's "means include the evaluation
ring orientations".

THE SHAPES
----------
* ShearNet: ``SHEARNET.g_original`` -- the network on the stamp as rendered,
  metacal never touching it -- calibrated by ShearNet's own metacal ``R^gamma``
  (the network run on ngmix's nine reconvolved images) plus its ``R^S``.
* ngmix: ``NGMIX.g_noshear`` -- the reconvolved metacal shape -- with the
  population PSF correction ``<e> - <R^PSF><e^PSF>`` (``e^PSF`` is
  ``STAMP.psf_g``), calibrated by its own ``R^gamma + R^S``. Every ngmix mean
  is corrected, the ones inside ``R^S`` included; no ShearNet value ever is.

UNCERTAINTIES
-------------
Delete-one-block jackknife over **objects**: the catalog rows are split into
``njack`` contiguous blocks, and each sample drops one block from every scene
and every ring station at once and re-forms the whole statistic -- means,
``R^gamma``, ``R^S`` and the ratio. So the shape/response covariance, the
pairing of the +/- populations and the correlation of the rotated copies of a
galaxy are all carried, for both estimators the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from catalog import Catalog

ESTIMATORS = ("shearnet", "ngmix")
METACAL_SHEARED = ("1p", "1m", "2p", "2m")
SELECTION_TYPES = ("noshear",) + METACAL_SHEARED
PSF_TYPES = ("1p_psf", "1m_psf", "2p_psf", "2m_psf")
#: The populations m1 and c2 are measured on.
G1_PAIR = ("g1_plus", "g1_minus")

#: Which column is "the shape" of each estimator, and whether the population
#: PSF correction applies to it.
SHAPE_VARIANT = {"shearnet": "original", "ngmix": "noshear"}
PSF_CORRECTED = {"shearnet": False, "ngmix": True}
TABLE = {"shearnet": "SHEARNET", "ngmix": "NGMIX"}

DEFAULT_NJACK = 20


@dataclass(frozen=True)
class SelectionCut:
    """A cut on ngmix's fit of each metacal product. Both inequalities strict."""

    min_t_ratio: float = 1.0
    min_s2n: float = 10.0

    def describe(self) -> str:
        return (f"ngmix T_t/Tpsf_t > {self.min_t_ratio:g} and s2n_t > {self.min_s2n:g} "
                "(flags_t == 0) on each metacal product t's own fit; sample = "
                "noshear selection; R = <R^gamma> + R^S")


#: The cut the paper applies.
PAPER_CUT = SelectionCut(min_t_ratio=1.0, min_s2n=10.0)


def cut_from_name(name: str, min_t_ratio=None, min_s2n=None) -> Optional[SelectionCut]:
    """``"metacal"`` -> the paper cut (thresholds overridable); ``"none"`` -> None."""
    if name == "none":
        return None
    if name != "metacal":
        raise ValueError(f"cut must be 'metacal' or 'none', got {name!r}")
    return SelectionCut(PAPER_CUT.min_t_ratio if min_t_ratio is None else min_t_ratio,
                        PAPER_CUT.min_s2n if min_s2n is None else min_s2n)


# ----------------------------------------------------------------------
# per-record quantities
# ----------------------------------------------------------------------

def _finite_rows(array: np.ndarray, n_lead: int = 3) -> np.ndarray:
    axes = tuple(range(n_lead, array.ndim))
    return np.isfinite(array).all(axis=axes) if axes else np.isfinite(array)


def _cut_columns(mtype: str) -> list:
    return [f"T_{mtype}", f"Tpsf_{mtype}", f"s2n_{mtype}", f"flags_{mtype}"]


def ngmix_passes(ngmix: dict, mtype: str, cut: SelectionCut) -> np.ndarray:
    """Whether each record passes ``cut`` on ngmix's fit of product ``mtype``.

    ``ngmix`` maps ``NGMIX`` column names to arrays (``Catalog.columns``).
    """
    T, Tpsf, s2n, flags = (ngmix[name] for name in _cut_columns(mtype))
    ok = (flags == 0) & np.isfinite(T) & np.isfinite(Tpsf) & (Tpsf > 0) & np.isfinite(s2n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok &= (T / Tpsf > cut.min_t_ratio) & (s2n > cut.min_s2n)
    return ok


def _response(cols: dict, types, step) -> np.ndarray:
    """``R[..., i, j] = (g_{j+1}p - g_{j+1}m)[i] / (2 step)`` from the given products."""
    p1, m1, p2, m2 = (cols[f"g_{t}"] for t in types)
    return np.stack([(p1 - m1), (p2 - m2)], axis=-1) / (2.0 * step)


class Sample:
    """Everything the statistics need for some scenes, read in one pass per table.

    ``select[t]`` is the cut on ngmix's fit of product ``t`` AND-ed with
    ``valid``; with ``cut=None`` every ``select[t]`` is ``valid`` itself, so
    ``R^S`` is exactly zero and the sample is everything measurable.

    ``shape[est]`` is the estimator's reported shape (``SHAPE_VARIANT``);
    ``g[est][v]`` holds ``g_original`` and ``g_noshear`` for Figure 5.
    """

    def __init__(self, cat: Catalog, scenes: Sequence[str],
                 cut: Optional[SelectionCut] = PAPER_CUT,
                 estimators: Sequence[str] = ESTIMATORS):
        self.cat = cat
        self.scenes = list(scenes)
        self.cut = cut
        self.estimators = list(estimators)
        self.step = cat.step

        flag_variants = ("original", "noshear") + METACAL_SHEARED + PSF_TYPES
        wanted = {table: set() for table in ("SHEARNET", "NGMIX", "STAMP")}
        for est in self.estimators:
            wanted[TABLE[est]] |= {f"g_{v}" for v in flag_variants}
            wanted[TABLE[est]] |= {f"flags_{v}" for v in flag_variants}
        for mtype in SELECTION_TYPES:
            wanted["NGMIX"] |= set(_cut_columns(mtype))
        wanted["STAMP"].add("psf_g")
        cols = {table: cat.columns(table, sorted(names), self.scenes)
                for table, names in wanted.items() if names}

        self.shape, self.g, self.r_gamma, self.r_psf = {}, {}, {}, {}
        valid = np.ones((len(self.scenes), cat.n_rotations, cat.n_objects), dtype=bool)
        for est in self.estimators:
            c = cols[TABLE[est]]
            variant = SHAPE_VARIANT[est]
            self.g[est] = {"original": c["g_original"], "noshear": c["g_noshear"]}
            self.shape[est] = c[f"g_{variant}"]
            self.r_gamma[est] = _response(c, METACAL_SHEARED, self.step)
            self.r_psf[est] = _response(c, PSF_TYPES, self.step)
            for v in {variant, *METACAL_SHEARED, *PSF_TYPES}:
                valid &= c[f"flags_{v}"] == 0
            valid &= _finite_rows(self.shape[est])
            valid &= _finite_rows(self.r_gamma[est]) & _finite_rows(self.r_psf[est])
        ngmix = cols["NGMIX"]
        # ngmix's noshear fit is what the cut reads, so it must exist for
        # every record either estimator is measured on
        valid &= ngmix["flags_noshear"] == 0
        self.s2n_noshear = ngmix["s2n_noshear"]
        self.psf_g = cols["STAMP"]["psf_g"]
        valid &= _finite_rows(self.psf_g)
        self.valid = valid

        self.select = {}
        for mtype in SELECTION_TYPES:
            self.select[mtype] = (valid if cut is None
                                  else valid & ngmix_passes(ngmix, mtype, cut))

    def scene_index(self, name: str) -> int:
        return self.scenes.index(name)

    def counts(self) -> dict:
        """Records measurable / selected per scene -- what the cut did."""
        out = {}
        for i, scene in enumerate(self.scenes):
            out[scene] = {"records": int(self.valid[i].size),
                          "measurable": int(self.valid[i].sum()),
                          **{f"selected_{t}": int(self.select[t][i].sum())
                             for t in SELECTION_TYPES}}
        return out


# ----------------------------------------------------------------------
# jackknife over objects
# ----------------------------------------------------------------------

class _Blocks:
    """Per-(scene, object block) sums, so every jackknife sample is a subtraction."""

    def __init__(self, n_objects: int, njack: int):
        njack = max(2, min(int(njack), n_objects))
        self.starts = np.unique(np.linspace(0, n_objects, njack + 1).astype(int)[:-1])
        self.sums: Dict[str, np.ndarray] = {}

    def add(self, key: str, values: np.ndarray, mask: np.ndarray) -> None:
        """``values`` ``(scene, station, object, ...)`` summed over ``mask``."""
        m = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
        per_object = np.where(m, values, 0.0).sum(axis=1)
        self.sums[key] = np.add.reduceat(per_object, self.starts, axis=1)

    def count(self, key: str, mask: np.ndarray) -> None:
        self.sums[key] = np.add.reduceat(mask.sum(axis=1).astype(float), self.starts, axis=1)

    def estimate(self, fn: Callable[[dict], dict]) -> dict:
        """``{name: (value, jackknife error)}`` for every output of ``fn(totals)``."""
        totals = {k: v.sum(axis=1) for k, v in self.sums.items()}
        value = fn(totals)
        n = len(self.starts)
        samples = [fn({k: totals[k] - v[:, b] for k, v in self.sums.items()})
                   for b in range(n)]
        out = {}
        for name, central in value.items():
            stack = np.stack([np.asarray(s[name], dtype=float) for s in samples])
            mean = stack.mean(axis=0)
            error = np.sqrt((n - 1) / n * ((stack - mean) ** 2).sum(axis=0))
            out[name] = (np.asarray(central, dtype=float), error)
        return out


def _population_means(sample: Sample, est: str, blocks: _Blocks) -> None:
    sel = sample.select
    blocks.count("n", sel["noshear"])
    blocks.add("e", sample.shape[est], sel["noshear"])
    blocks.add("Rg", sample.r_gamma[est], sel["noshear"])
    blocks.add("Rp", sample.r_psf[est], sel["noshear"])
    blocks.add("gpsf", sample.psf_g, sel["noshear"])
    for mtype in METACAL_SHEARED:
        blocks.count(f"n_{mtype}", sel[mtype])
        blocks.add(f"e_{mtype}", sample.shape[est], sel[mtype])
        blocks.add(f"gpsf_{mtype}", sample.psf_g, sel[mtype])


def _calibrated(t: dict, step: float, psf_corrected: bool) -> dict:
    """Per-scene ``<e>`` (PSF-corrected if asked), ``<R^gamma>``, ``R^S``, ``<R^PSF>``."""
    n = t["n"][:, None]
    Rg = t["Rg"] / n[..., None]
    Rp = t["Rp"] / n[..., None]
    def corrected(shape_sum, psf_sum, count):
        """Mean shape of a selection, R^PSF-corrected when the estimator is."""
        mean = shape_sum / count[:, None]
        if psf_corrected:
            mean = mean - np.einsum("sij,sj->si", Rp, psf_sum / count[:, None])
        return mean

    RS = np.zeros_like(Rg)
    for j, (plus, minus) in enumerate((("1p", "1m"), ("2p", "2m"))):
        # R^S of the shape actually reported: for ngmix the R^PSF-corrected one
        mean_p = corrected(t[f"e_{plus}"], t[f"gpsf_{plus}"], t[f"n_{plus}"])
        mean_m = corrected(t[f"e_{minus}"], t[f"gpsf_{minus}"], t[f"n_{minus}"])
        RS[:, :, j] = (mean_p - mean_m) / (2.0 * step)
    e = corrected(t["e"], t["gpsf"], t["n"])
    return {"e": e, "Rg": Rg, "RS": RS, "R": Rg + RS, "Rp": Rp}


def responses(sample: Sample, est: str, njack: int = DEFAULT_NJACK,
              pair: Sequence[str] = G1_PAIR) -> dict:
    """Table 2: ``<R^gamma>``, ``R^S``, ``R = <R^gamma> + R^S`` and ``<R^PSF>``,
    each the mean of the two populations of ``pair``, with jackknife errors."""
    p, q = (sample.scene_index(s) for s in pair)
    blocks = _Blocks(sample.cat.n_objects, njack)
    _population_means(sample, est, blocks)

    def fn(t):
        c = _calibrated(t, sample.step, PSF_CORRECTED[est])
        return {key: 0.5 * (c[key][p] + c[key][q]) for key in ("Rg", "RS", "R", "Rp")}

    out = blocks.estimate(fn)
    out["n_selected"] = int(sample.select["noshear"][[p, q]].sum())
    return out


def shear_bias(sample: Sample, est: str, njack: int = DEFAULT_NJACK,
               pair: Sequence[str] = G1_PAIR, component: int = 0) -> dict:
    """Equation 3 on the selected sample, with ``R_i = <R^gamma_ii> + R^S_ii``.

    ``component`` is the sheared one (0 for the g1 pair): ``m`` is measured along
    it and ``c`` on the other component. Returns ``{"m": (value, err),
    "c": (value, err), ...}``.
    """
    p, q = (sample.scene_index(s) for s in pair)
    a, b = component, 1 - component
    key = "g1" if a == 0 else "g2"
    g_plus = sample.cat.scenes[pair[0]][key]
    g_minus = sample.cat.scenes[pair[1]][key]
    half_separation = 0.5 * (g_plus - g_minus)
    if half_separation == 0:
        raise ValueError(f"{pair} do not differ in component {a + 1}")

    blocks = _Blocks(sample.cat.n_objects, njack)
    _population_means(sample, est, blocks)

    def fn(t):
        c = _calibrated(t, sample.step, PSF_CORRECTED[est])
        e, R, Rg = c["e"], c["R"], c["Rg"]
        return {
            "m": (e[p, a] - e[q, a]) / (half_separation * (R[p, a, a] + R[q, a, a])) - 1.0,
            "c": (e[p, b] + e[q, b]) / (R[p, b, b] + R[q, b, b]),
            # the same estimator without R^S, so the size of the selection
            # correction is visible next to the number it changes
            "m_without_RS": (e[p, a] - e[q, a])
            / (half_separation * (Rg[p, a, a] + Rg[q, a, a])) - 1.0,
            "R_aa": 0.5 * (R[p, a, a] + R[q, a, a]),
            "RS_aa": 0.5 * (c["RS"][p, a, a] + c["RS"][q, a, a]),
        }

    out = blocks.estimate(fn)
    out["n_selected"] = int(sample.select["noshear"][[p, q]].sum())
    return out


# ----------------------------------------------------------------------
# Figure 3 and Figure 5 inputs
# ----------------------------------------------------------------------

def responses_per_object(sample: Sample, est: str, pair: Sequence[str] = G1_PAIR) -> dict:
    """Per-object diagonal responses and ngmix S/N, for Figure 3.

    Each object's ``R^gamma_ii`` and ``R^PSF_ii`` are averaged over the records
    of ``pair`` x every ring station that are in ``sample.select['noshear']``
    (with ``cut=None``: every measurable record); its S/N is ngmix's
    ``s2n_noshear`` averaged over the same records.
    """
    idx = [sample.scene_index(s) for s in pair]
    mask = sample.select["noshear"][idx]
    count = mask.sum(axis=(0, 1)).astype(float)
    keep = count > 0
    s2n = sample.s2n_noshear[idx]

    def mean(values):
        total = np.where(mask, values, 0.0).sum(axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return total / count

    out = {"s2n": mean(s2n)[keep], "n_objects": int(keep.sum())}
    for name, matrix in (("gamma", sample.r_gamma[est]), ("psf", sample.r_psf[est])):
        m = matrix[idx]
        out[name] = {"11": mean(m[..., 0, 0])[keep], "22": mean(m[..., 1, 1])[keep]}
    return out


def leakage_inputs(sample: Sample, est: str, scene: str = "zero",
                   variant: str = "original") -> dict:
    """Ring-averaged shapes and PSF ellipticities of one population, for Figure 5.

    With a cut, an object is kept only if **every** ring station of it passes
    the ngmix noshear cut, so the ring average still cancels the intrinsic
    shape; the same objects are used for both estimators. ``variant`` is the
    shape: ``original`` is the raw measurement on the stamp as rendered (no
    response division, no PSF correction), as the figure states.
    """
    i = sample.scene_index(scene)
    keep = sample.select["noshear"][i].all(axis=0)
    shape = sample.g[est][variant][i]
    e_ring = shape[:, keep].mean(axis=0)
    gpsf = sample.psf_g[i][:, keep].mean(axis=0)
    rpsf = sample.r_psf[est][i][:, keep].mean(axis=0)
    finite = np.isfinite(e_ring).all(axis=1)
    return {
        "e1_gal": e_ring[finite, 0], "e2_gal": e_ring[finite, 1],
        "e1_psf": gpsf[finite, 0], "e2_psf": gpsf[finite, 1],
        "r11_psf": rpsf[finite, 0, 0], "r22_psf": rpsf[finite, 1, 1],
        "n_objects": int(finite.sum()), "n_total": int(keep.size),
    }


#: Figure 5's shape per estimator, the same pipeline its m and c come from:
#: ngmix's metacal noshear shape (R^PSF-corrected), ShearNet's original-image
#: shape (never R^PSF-corrected).
LEAKAGE_VARIANT = {"shearnet": "original", "ngmix": "noshear"}


def calibrated_leakage_inputs(sample: Sample, est: str, scene: str = "zero",
                              njack: int = DEFAULT_NJACK) -> dict:
    """Figure 5's input, calibrated the way m and c are.

    * every shape is divided by the estimator's own diagonal
      ``R = <R^gamma> + R^S``, measured on the selected records of ``scene``;
    * ngmix's is ``g_noshear`` and is **R^PSF-corrected**; ShearNet's is
      ``g_original`` and is **not**. The correction itself is
      ``superbit_lensing``'s (``PSFLeakagePanelMaker(correct_psf_leakage=True)``:
      ``e - R^PSF_ii e^PSF_i`` with ``R^PSF_ii`` the mean in each ``e^PSF``
      percentile bin, LITB III Appendix D), so this only sets
      ``correct_psf_leakage`` and hands over ``R^PSF / R`` -- which makes the
      result ``(e - R^PSF e^PSF) / R``, LITB III Eq. 24's order.
    """
    data = leakage_inputs(sample, est, scene=scene, variant=LEAKAGE_VARIANT[est])
    R = responses(sample, est, njack=njack, pair=(scene, scene))["R"][0]
    r1, r2 = float(R[0, 0]), float(R[1, 1])
    data["e1_gal"] = data["e1_gal"] / r1
    data["e2_gal"] = data["e2_gal"] / r2
    data["r11_psf"] = data["r11_psf"] / r1
    data["r22_psf"] = data["r22_psf"] / r2
    data["correct_psf_leakage"] = bool(PSF_CORRECTED[est])
    data["R"] = (r1, r2)
    data["shape"] = (f"(g_{LEAKAGE_VARIANT[est]} - R^PSF e^PSF) / R" if PSF_CORRECTED[est]
                     else f"g_{LEAKAGE_VARIANT[est]} / R")
    return data
