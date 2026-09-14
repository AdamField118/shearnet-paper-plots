"""Synthesise an evaluation FITS with the production schema, at small N.

The table and figure scripts here cannot be exercised without a file to read,
and the real ones are ~800 MB and live on the cluster. This writes a file with
the same HDUs, the same column names and the same shapes, at a size that fits in
a test.

It is a schema fixture, not a physics simulation: the numbers are drawn to be
plausible and to carry a KNOWN injected bias, so a script can be checked against
an answer rather than only against "it ran".

    python make_fixture.py --out /tmp/fake/evaluation.fits --n 5000
    python make_fixture.py --out /tmp/fake --suite     # one per paper run

The injected truth is recorded in the primary header (``FAKE_M1``, ``FAKE_C2``,
``FAKE_A1``, ``FAKE_A2``) so a test can assert recovery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

STATIONS = ("", "_r45", "_r90", "_r135")
ESTIMATORS = ("shearnet", "ngmix")


def _pair(rng, n, sign, shear, m1, c2, alpha, beta, gpsf, tpsf, sigma_e):
    """One population's columns, with a known m1, c2 and leakage built in."""
    col = {}
    col["gpsf"], col["Tpsf"] = gpsf, tpsf
    col["s2n"] = rng.lognormal(3.0, 0.6, n)
    col["hlr_th"] = rng.lognormal(-0.7, 0.4, n)
    col["flux_th"] = rng.lognormal(9.0, 0.8, n)

    dt = tpsf - tpsf.mean()
    # ONE intrinsic ellipticity per object, rotated round the ring. Drawing a
    # fresh one per station would leave sigma_e/2 of scatter in the ring mean
    # instead of cancelling it -- which is the whole point of the ring, and
    # would make every ring-dependent quantity here untestable.
    base_intrinsic = rng.normal(0, sigma_e, n) + 1j * rng.normal(0, sigma_e, n)
    for station, angle in zip(STATIONS, (0.0, 45.0, 90.0, 135.0)):
        # Spin-2: the shape picks up exp(2 i theta), so the 90-degree station
        # flips its sign and the four stations sum to zero exactly.
        intrinsic = base_intrinsic * np.exp(2j * np.deg2rad(angle))
        applied = sign * shear
        e1 = intrinsic.real + (1.0 + m1) * applied + alpha * gpsf[:, 0] + beta * dt
        e2 = intrinsic.imag + c2 + alpha * gpsf[:, 1] + beta * dt
        shape = np.column_stack([e1, e2]) + rng.normal(0, 1e-3, (n, 2))
        # run.py writes stamps.labels[:, :2], which is the COMPOSED shape --
        # the intrinsic ellipticity with the applied shear on it -- not the
        # applied shear alone. It therefore varies object to object, and a
        # figure that regresses against it (fig:prediction-residuals) needs
        # that spread: a constant truth column makes the quadratic fit inside
        # superbit_lensing.plot_comparison singular.
        col[f"g_th{station}"] = np.column_stack([
            intrinsic.real + applied, intrinsic.imag])
        for est in ESTIMATORS:
            col[f"e_{est}{station}"] = shape + rng.normal(0, 1e-3, (n, 2))
            col[f"e_{est}_uncorrected{station}"] = shape + rng.normal(0, 1e-3, (n, 2))
            col[f"e_{est}_metacal_raw{station}"] = shape + rng.normal(0, 1e-3, (n, 2))
            col[f"e_{est}_metacal_corrected{station}"] = shape
            eye = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
            col[f"R_{est}_sim{station}"] = eye + rng.normal(0, 0.01, (n, 2, 2))
            col[f"R_{est}_metacal{station}"] = eye + rng.normal(0, 0.01, (n, 2, 2))
            col[f"Rgamma_{est}_metacal{station}"] = eye + rng.normal(0, 0.01, (n, 2, 2))
            col[f"Rpsf_{est}_sim{station}"] = rng.normal(0, 0.05, (n, 2, 2))
            col[f"Rpsf_{est}_metacal{station}"] = rng.normal(0, 0.05, (n, 2, 2))
            col[f"flag_{est}{station}"] = np.zeros(n, dtype=np.int32)
    for est in ESTIMATORS:
        col[f"Rbarpsf_{est}_metacal"] = np.broadcast_to(
            np.eye(2) * 0.27, (n, 2, 2)).copy()
    # ngmix's own noshear fit quantities, unsuffixed by ring station exactly as
    # run.py writes them. selection.superbit_mask reads these three, so without
    # them the sample cut cannot be exercised against the schema at all. T is
    # tied to hlr_th so the cut removes the small end, and a handful of failed
    # fits are given the negative T the real files contain.
    T = 2.0 * col["hlr_th"] ** 2
    T[rng.random(n) < 0.02] = -0.5
    col["T_ngmix"] = T
    col["s2n_ngmix"] = col["s2n"]
    col["flux_ngmix"] = col["flux_th"]
    return col


def _leakage(rng, n, alpha, beta, gpsf, tpsf, sigma_e):
    col = {"gpsf": gpsf, "Tpsf": tpsf, "s2n": rng.lognormal(3.0, 0.6, n)}
    dt = tpsf - tpsf.mean()
    for est in ESTIMATORS:
        # The ring-averaged raw shape: intrinsic ellipticity has cancelled, so
        # only the leakage survives. This is the column the alpha fit reads.
        ring = np.column_stack([alpha * gpsf[:, 0] + beta * dt,
                                alpha * gpsf[:, 1] + beta * dt])
        # Residual scatter AFTER the ring average. It must be small enough that
        # the injected leakage is recoverable, or the fixture cannot verify the
        # alpha fit -- only that it runs. alpha * sigma(e_PSF) is order
        # alpha * 0.042, so the per-bin error has to sit well below that.
        ring = ring + rng.normal(0, 2.0e-4, (n, 2))
        col[f"e_{est}_raw_ring"] = ring
        col[f"e_{est}_ring"] = ring.copy()
        for station in STATIONS:
            noisy = ring + rng.normal(0, sigma_e, (n, 2))
            col[f"e_{est}{station}"] = noisy
            col[f"e_{est}_raw{station}"] = noisy
            col[f"e_{est}_original{station}"] = noisy
            col[f"Rpsf_{est}_metacal{station}"] = rng.normal(0, 0.05, (n, 2, 2))
        col[f"Rpsf_{est}_metacal"] = rng.normal(0, 0.05, (n, 2, 2))
        col[f"Rbarpsf_{est}_metacal"] = np.broadcast_to(
            np.eye(2) * 0.27, (n, 2, 2)).copy()
    return col


def write_fixture(path, n=5000, seed=0, m1=-0.012, c2=3.0e-5,
                  alpha=2.0e-2, beta=1.0e-3, shear=0.01, sigma_e=0.25):
    """Write one evaluation FITS with a known injected bias."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gpsf = rng.normal(0, 0.042, (n, 2))
    tpsf = rng.normal(0.35, 0.085, n)

    primary = fits.PrimaryHDU()
    for key, value, comment in (
        ("SEED", 150, "seed"), ("SAMPLES", n, "samples"),
        ("SHEAR_TR", shear, "shear_true"), ("COMPONEN", 0, "component"),
        ("N_JACKKN", 20, "n_jackknife"), ("SNC", len(STATIONS), "shape_noise_cancel"),
        ("RENDER_S", 812.5, "render_seconds"), ("INFERENC", 6.31, "inference_seconds"),
        ("NGMIX_SE", 4180.0, "ngmix_seconds"), ("NGMIX_NP", 18, "ngmix_nproc"),
        ("CATLEVEL", "paper", "catalog level"),
        ("FAKE_M1", m1, "injected m1"), ("FAKE_C2", c2, "injected c2"),
        ("FAKE_A1", alpha, "injected alpha"), ("FAKE_B1", beta, "injected beta"),
    ):
        primary.header[key] = (value, comment)

    hdus = [primary]
    for tag, component in (("", 0), ("2", 1)):
        for label, sign in (("P", 1.0), ("M", -1.0)):
            cols = _pair(rng, n, sign, shear, m1, c2, alpha, beta, gpsf, tpsf, sigma_e)
            hdu = fits.BinTableHDU(Table(cols), name=f"TAB_{label}{tag}")
            hdu.header["COMPONEN"] = (component, "sheared component")
            hdus.append(hdu)
    hdus.append(fits.BinTableHDU(Table(_leakage(rng, n, alpha, beta, gpsf, tpsf, sigma_e)),
                                 name="LEAKAGE"))

    summary = {k: [] for k in ("estimator", "correction", "component", "m", "m_err",
                               "c", "c_err", "R11", "R22", "n_used")}
    for est in ESTIMATORS:
        for corr in ("metacal", "sim", "none"):
            for component in (0, 1):
                summary["estimator"].append(est)
                summary["correction"].append(corr)
                summary["component"].append(component)
                summary["m"].append(m1 if corr != "none" else m1 - 0.05)
                summary["m_err"].append(2.0e-4)
                # lin2026: the SHEARED component's mean over <R>. Deliberately
                # not c2, so a script that reads this by name is caught.
                summary["c"].append(1.7e-4)
                summary["c_err"].append(1.0e-5)
                summary["R11"].append(1.0)
                summary["R22"].append(1.0)
                summary["n_used"].append(n)
    hdus.append(fits.BinTableHDU(Table(summary), name="SUMMARY"))
    hdus.append(fits.BinTableHDU(Table({
        "estimator": [e for e in ESTIMATORS for _ in range(4)],
        "correction": ["metacal"] * 4 * len(ESTIMATORS),
        "bin": list(range(4)) * len(ESTIMATORS),
        "m": rng.normal(m1, 2e-3, 4 * len(ESTIMATORS)),
        "m_err": np.full(4 * len(ESTIMATORS), 5e-4),
        "c": rng.normal(c2, 1e-5, 4 * len(ESTIMATORS)),
        "c_err": np.full(4 * len(ESTIMATORS), 2e-5),
    }), name="BINNED"))
    hdus.append(fits.BinTableHDU(Table({
        "estimator": list(ESTIMATORS),
        "corrected": [False] * len(ESTIMATORS),
        "mean_e1": [1e-4] * len(ESTIMATORS), "mean_e1_err": [1e-5] * len(ESTIMATORS),
        "mean_e2": [2e-5] * len(ESTIMATORS), "mean_e2_err": [1e-5] * len(ESTIMATORS),
        "mean_e1_raw": [1e-4] * len(ESTIMATORS), "mean_e2_raw": [2e-5] * len(ESTIMATORS),
        "Rpsf11": [0.27] * len(ESTIMATORS), "Rpsf11_err": [0.01] * len(ESTIMATORS),
        "Rpsf22": [0.25] * len(ESTIMATORS), "Rpsf22_err": [0.01] * len(ESTIMATORS),
        "n_used": [n] * len(ESTIMATORS),
    }), name="LEAKSUM"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


#: Enough of the campaign to exercise every table, with a different injected m1
#: per run so a mis-wired table shows up as the wrong row rather than as a crash.
SUITE = {
    "first": -0.0400, "second": -0.0250, "third": -0.0170, "fourth": -0.0120,
    "tier1/no_psf_response": -0.0135,
    "tier2/01_galaxy_only": -0.0900, "tier2/02_psf_branch_concat": -0.0600,
    "tier2/03_transformer_fusion": -0.0400, "tier2/04_auxiliary_targets": -0.0350,
    "tier2/05_d4_augmentation": -0.0300, "tier2/06_d4_equivariant": -0.0200,
    "tier3/no_gamma_response": -0.0450, "tier3/no_isotropy": -0.0150,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", action="store_true",
                        help="write one fixture per paper run under --out")
    args = parser.parse_args(argv)

    if args.suite:
        for index, (name, m1) in enumerate(SUITE.items()):
            path = args.out / name / "evaluation.fits"
            write_fixture(path, n=args.n, seed=args.seed + index, m1=m1,
                          alpha=2.0e-2 * (1 + 0.1 * index))
            print(f"wrote {path}")
    else:
        print(f"wrote {write_fixture(args.out, n=args.n, seed=args.seed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
