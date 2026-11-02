"""tab:timing -- what measuring shear costs each estimator, split into its parts.

Both estimators are calibrated by metacal (:mod:`shear_stats`, THE SHAPES), so
both need metacal's nine reconvolved products of every observation (noshear,
the four sheared ones for R^gamma and R^S, the four PSF-sheared ones for
R^PSF). The table keeps apart the two costs that this mixes:

* **inference on one galaxy stamp** (with its PSF stamp):
  ShearNet on the GPU, the ShearNet job's benchmark
  (``TIMING`` ``shearnet_benchmark_best_seconds_per_stamp``: the network alone
  on stamps already on the device, compiled and warmed up, every pass waited
  for, at its fastest batch size); ngmix on one CPU core, its fit of the
  original stamp, ``NGMIX.fit_cpu_seconds`` (size guess, PSF fit and galaxy
  fit). The evaluation's own ShearNet time, which also copies every batch of
  stamps to the GPU, is printed alongside;
* **making the metacal stamps**, the same for both, on one CPU core:
  ``NGMIX.metacal_image_cpu_seconds`` (all nine products of one observation).

and then the totals for every observation of the catalog -- nine stamps each --
with and without the metacal stamps, on NERSC Perlmutter-sized hardware:
ngmix on one CPU node (``CPU_NODE_CORES`` cores), ShearNet on one A100 with its
share of a GPU node (``GPU_SLICE_CORES`` cores, which make its metacal
stamps). The totals are the measured per-stamp costs times the work, divided
over the hardware, assuming it is kept busy; the jobs' own run times are longer,
since they also simulated every image. ngmix's fits on the nine products are
``metacal_cpu_seconds`` minus the stamps' cost.

PSF fits: the evaluation fitted each distinct PSF stamp of a galaxy once and
shared it between that galaxy's observations, which a survey would not see.
Every ngmix time here therefore counts the PSF fit of every stamp
(``*_psf_uncached_cpu_seconds`` in place of the shared ``*_psf_cpu_seconds``),
as ngmix's ``MetacalBootstrapper`` fits it -- the same footing as ShearNet,
which reads every PSF stamp.

Rendering the simulation and compiling the network are excluded.

    python timing_table.py --fits ../evaluations/fourth.fits --out ../output/tab_timing.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catalog import Catalog

#: metacal products each measurement needs
PRODUCTS = 9
#: one NERSC Perlmutter CPU node
CPU_NODE_CORES = 128
#: the CPU cores of a Perlmutter GPU node that go with one of its four A100s
GPU_SLICE_CORES = 16

COLUMNS = ["fit_cpu_seconds", "fit_psf_cpu_seconds", "fit_psf_uncached_cpu_seconds",
           "metacal_cpu_seconds", "metacal_image_cpu_seconds", "metacal_psf_cpu_seconds",
           "metacal_psf_uncached_cpu_seconds"]


def measure(catalog: Catalog) -> dict:
    """Mean seconds per stamp / per observation, and the totals over the catalog."""
    timing = catalog.key_values("TIMING")
    c = {k: float(np.mean(v)) for k, v in
         catalog.columns("NGMIX", COLUMNS, list(catalog.scenes)).items()}
    n = int(catalog.header["NRECORD"])
    shearnet_stamp = timing["shearnet.shearnet_benchmark_best_seconds_per_stamp"]
    pipeline_stamp = (timing["shearnet.shearnet_inference_seconds"]
                      / timing["shearnet.shearnet_images"])
    ngmix_stamp = (c["fit_cpu_seconds"] - c["fit_psf_cpu_seconds"]
                   + c["fit_psf_uncached_cpu_seconds"])
    images = c["metacal_image_cpu_seconds"]
    ngmix_products = (c["metacal_cpu_seconds"] - images - c["metacal_psf_cpu_seconds"]
                      + c["metacal_psf_uncached_cpu_seconds"])
    shearnet_products = PRODUCTS * shearnet_stamp
    hours = 1.0 / 3600.0
    return {
        "observations": n, "gpu": timing["shearnet.device_kind"],
        "cpu": timing["ngmix.cpu_model"], "ngmix_cores": timing["ngmix.cpus"],
        "batch_size": timing["shearnet.shearnet_benchmark_best_batch_size"],
        "pipeline_stamp": pipeline_stamp,
        # per stamp, and the metacal stamps of one observation
        "shearnet_stamp": shearnet_stamp, "ngmix_stamp": ngmix_stamp,
        "images": images, "shearnet_products": shearnet_products,
        "ngmix_products": ngmix_products,
        # every observation, on the NERSC-sized hardware, in hours
        "ngmix_fits_h": n * ngmix_products / CPU_NODE_CORES * hours,
        "ngmix_total_h": n * (images + ngmix_products) / CPU_NODE_CORES * hours,
        "shearnet_fits_h": n * shearnet_products * hours,
        # the GPU and its cores work at the same time, so the slower sets the pace
        "shearnet_images_h": n * images / GPU_SLICE_CORES * hours,
        "shearnet_total_h": max(n * shearnet_products, n * images / GPU_SLICE_CORES) * hours,
    }


def latex(t: dict) -> list:
    ms = lambda seconds: f"{1e3 * seconds:.3g} ms"  # noqa: E731  3 significant figures
    h = lambda hours: (f"{hours:.3g} h" if hours >= 1  # noqa: E731
                       else f"{60 * hours:.3g} min")
    node = f"1 CPU node ({CPU_NODE_CORES} cores)"
    slice_ = f"1 A100 + {GPU_SLICE_CORES} cores"
    n = f"{t['observations']:,}".replace(",", "{,}")
    return [
        r"\multicolumn{3}{l}{\emph{One galaxy stamp (with its PSF stamp)}} \\",
        rf"\textsc{{ShearNet}}-D4 inference & 1 A100 & {ms(t['shearnet_stamp'])} \\",
        rf"\textsc{{ngmix}} fit & 1 CPU core & {ms(t['ngmix_stamp'])} \\",
        rf"\multicolumn{{3}}{{l}}{{\emph{{Metacal stamps of one observation ({PRODUCTS})}}}} \\",
        rf"making them (both estimators) & 1 CPU core & {ms(t['images'])} \\",
        r"\hline",
        rf"\multicolumn{{3}}{{l}}{{\emph{{All {n} observations, {PRODUCTS} stamps each}}}} \\",
        rf"\textsc{{ShearNet}}-D4, inference only & 1 A100 & {h(t['shearnet_fits_h'])} \\",
        rf"\textsc{{ShearNet}}-D4, with metacal stamps & {slice_} & {h(t['shearnet_total_h'])} \\",
        rf"\textsc{{ngmix}}, fits only & {node} & {h(t['ngmix_fits_h'])} \\",
        rf"\textsc{{ngmix}}, with metacal stamps & {node} & {h(t['ngmix_total_h'])} \\",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    catalog = Catalog(args.fits)
    t = measure(catalog)
    lines = [
        f"% tab:timing from {args.fits.name} ({catalog.run_name})",
        "% columns: what & hardware & time",
        f"% means over {t['observations']} observations; ShearNet on {t['gpu']}, "
        f"ngmix on {t['ngmix_cores']} cores of {t['cpu']}",
        "% excluded: rendering and JAX compilation; every ngmix stamp fits its own PSF",
        f"% ShearNet per stamp: {1e3 * t['shearnet_stamp']:.4f} ms alone at batch size "
        f"{t['batch_size']}, {1e3 * t['pipeline_stamp']:.4f} ms in the evaluation "
        "(with host-device copies)",
        f"% per stamp, ngmix's fit takes {t['ngmix_stamp'] / t['shearnet_stamp']:.0f}x "
        "ShearNet's inference",
        f"% inference only: ShearNet on 1 A100 is {t['ngmix_fits_h'] / t['shearnet_fits_h']:.1f}x "
        f"faster than ngmix on {CPU_NODE_CORES} cores",
        f"% with metacal stamps: {t['ngmix_total_h'] / t['shearnet_total_h']:.2f}x "
        f"(ShearNet's {GPU_SLICE_CORES} cores need {t['shearnet_images_h']:.1f} h for the "
        f"stamps, its GPU {t['shearnet_fits_h']:.1f} h)",
    ] + latex(t)
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\n% wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
