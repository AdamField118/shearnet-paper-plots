"""A small ``shearnet-eval`` catalog with known answers, for the tests.

Same schema, same row order and the same column names as a real evaluation, but
the measurements are a linear model whose answers are known:

* every metacal product ``t`` of an estimator measures
  ``R (e_true + delta_t) + noise`` with ``delta_t`` the metacal shear of that
  product, so ``R^gamma`` is exactly ``R``;
* the ``*_psf`` products move the shape by ``rho`` per unit PSF shear, so
  ``R^PSF = rho``, and ngmix's noshear shape leaks ``rho * e^PSF`` -- the
  population PSF correction removes it exactly;
* the original-image shapes leak ``alpha * e^PSF`` (the Figure 5 slope);
* ngmix's S/N on product ``t`` is ``S0 (1 + kappa g_t,1)``: an S/N cut then
  prefers galaxies that *look* sheared along +g1, which is a selection bias of
  exactly the kind metacal's ``R^S`` measures. Without ``R^S``, m is wrong;
  with it, m is the injected value;
* noise is the same in every scene (pair-matched), as in the real evaluation.

    python make_fixture.py -o /tmp/fixture.fits --n 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

SCENES = (("zero", 0.0, 0.0), ("g1_plus", 0.01, 0.0), ("g1_minus", -0.01, 0.0),
          ("g2_plus", 0.0, 0.01), ("g2_minus", 0.0, -0.01))
ROTATIONS = (0.0, 45.0, 90.0, 135.0)
STEP = 0.01
METACAL = ("noshear", "1p", "1m", "2p", "2m")
DELTA = {"noshear": (0, 0), "1p": (STEP, 0), "1m": (-STEP, 0),
         "2p": (0, STEP), "2m": (0, -STEP)}
PSF = {"1p_psf": (0, +1), "1m_psf": (0, -1), "2p_psf": (1, +1), "2m_psf": (1, -1)}

#: The injected truth.
TRUTH = {
    "ngmix": {"R": 0.75, "R_original": 0.80, "rho": 0.22, "alpha": (0.020, 0.060),
              "noise": 0.06},
    "shearnet": {"R": 0.98, "R_original": 0.98, "rho": -0.15, "alpha": (-0.006, -0.005),
                 "noise": 0.03},
    "kappa": 2.0,          # S/N's dependence on ngmix's measured g1
    "Tpsf": 0.1,
}


def build(n: int, seed: int = 0, shearnet_m: float = 0.0, kappa=None) -> fits.HDUList:
    """The catalog. ``shearnet_m`` scales ShearNet's original-image response, so
    its m is ``shearnet_m`` (with no selection; to first order with one)."""
    rng = np.random.default_rng(seed)
    K, S = len(ROTATIONS), len(SCENES)
    kappa = TRUTH["kappa"] if kappa is None else kappa

    # per object
    e_mag = np.clip(rng.normal(0.2, 0.1, n), 0, 0.7)
    e_src = e_mag * np.exp(2j * rng.uniform(0, np.pi, n))
    psf_g = rng.uniform(-0.05, 0.05, (n, 2)) + np.array([0.01, -0.005])
    s2n0 = np.exp(rng.normal(np.log(14.0), 0.5, n))
    T0 = rng.uniform(0.05, 0.3, n)
    # per object and station, the same in every scene
    noise = {est: rng.normal(0, TRUTH[est]["noise"], (K, n, 2))
             for est in ("ngmix", "shearnet")}

    rows = S * K * n
    scene_id = np.repeat(np.arange(S), K * n)
    rotation_id = np.tile(np.repeat(np.arange(K), n), S)
    catalog_row = np.tile(np.arange(n), S * K)

    theta = np.deg2rad(np.asarray(ROTATIONS))[rotation_id]
    e_rot = e_src[catalog_row] * np.exp(2j * theta)
    g_applied = np.array([(g1, g2) for _, g1, g2 in SCENES])[scene_id]
    e_true = np.stack([e_rot.real, e_rot.imag], axis=1) + g_applied
    gpsf = psf_g[catalog_row]

    keys = {"record_id": np.arange(rows), "catalog_row": catalog_row,
            "scene_id": scene_id.astype(np.int16), "rotation_id": rotation_id.astype(np.int16)}
    tables = {"TRUTH": dict(keys, e_prepsf=e_true, g_applied=g_applied),
              "STAMP": dict(keys, psf_g=gpsf, psf_flags=np.zeros(rows, np.int32),
                            s2n_stamp=s2n0[catalog_row] * 4.0)}

    for est, table in (("ngmix", "NGMIX"), ("shearnet", "SHEARNET")):
        p = TRUTH[est]
        nu = noise[est][rotation_id, catalog_row]
        cols = dict(keys)
        r_orig = p["R_original"] * (1.0 + (shearnet_m if est == "shearnet" else 0.0))
        cols["g_original"] = r_orig * e_true + np.asarray(p["alpha"]) * gpsf + nu
        cols["flags_original"] = np.zeros(rows, np.int32)
        leak = p["rho"] * gpsf if est == "ngmix" else 0.0
        for t in METACAL:
            cols[f"g_{t}"] = p["R"] * (e_true + np.asarray(DELTA[t])) + leak + nu
            cols[f"flags_{t}"] = np.zeros(rows, np.int32)
        for t, (component, sign) in PSF.items():
            shift = np.zeros(2)
            shift[component] = sign * STEP * p["rho"]
            cols[f"g_{t}"] = cols["g_noshear"] + shift
            cols[f"flags_{t}"] = np.zeros(rows, np.int32)
        if est == "ngmix":
            for t in ("original",) + METACAL:
                cols[f"T_{t}"] = T0[catalog_row]
                cols[f"Tpsf_{t}"] = np.full(rows, TRUTH["Tpsf"])
                cols[f"s2n_{t}"] = s2n0[catalog_row] * (1.0 + kappa * cols[f"g_{t}"][:, 0])
        tables[table] = cols

    primary = fits.PrimaryHDU()
    for key, value in (("SCHEMA", "shearnet-eval"), ("SCHEMAV", 1), ("RUNNAME", "fixture"),
                       ("EVALNAME", "default"), ("NOBJ", n), ("NSCENE", S), ("NROT", K),
                       ("NRECORD", rows), ("MCALSTEP", STEP), ("MCALPSF", "dilate")):
        primary.header[key] = value
    hdus = [primary]
    for name, cols in tables.items():
        hdus.append(fits.BinTableHDU.from_columns(
            [fits.Column(name=k, array=v, format=_fmt(v)) for k, v in cols.items()],
            name=name))
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="scene_id", array=np.arange(S), format="K"),
        fits.Column(name="name", array=np.array([s[0] for s in SCENES]), format="10A"),
        fits.Column(name="g1", array=np.array([s[1] for s in SCENES]), format="D"),
        fits.Column(name="g2", array=np.array([s[2] for s in SCENES]), format="D")],
        name="SCENES"))
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name="rotation_id", array=np.arange(K), format="K"),
        fits.Column(name="rotation_deg", array=np.asarray(ROTATIONS), format="D")],
        name="ROTATIONS"))
    return fits.HDUList(hdus)


def _fmt(array) -> str:
    array = np.asarray(array)
    width = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    if array.dtype.kind == "f":
        code = "D"
    else:
        code = {8: "K", 4: "J", 2: "I"}[array.dtype.itemsize]
    return f"{width}{code}" if width > 1 else code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.n, args.seed).writeto(args.out, overwrite=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
