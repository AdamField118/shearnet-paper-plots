"""tab:timing -- what one shear measurement costs each estimator.

Both estimators are calibrated by metacal (:mod:`shear_stats`, THE SHAPES), so
both need metacal's nine reconvolved products of every observation (noshear,
the four sheared ones for R^gamma and R^S, the four PSF-sheared ones for
R^PSF). They differ only in what measures those products:

* making the nine products, on a CPU core: ``NGMIX.metacal_image_cpu_seconds``;
* ngmix fitting them, PSF fits included, on a CPU core:
  ``NGMIX.metacal_cpu_seconds`` minus the products' cost;
* ShearNet predicting on them, on the GPU: the catalog's ``TIMING``
  ``shearnet_inference_seconds / shearnet_images``, times nine.

Every number is per observation, averaged over every observation of the
catalog. Rendering the simulation and compiling the network are excluded. The
CPU seconds are one core's; ngmix's PSF fits are shared between the
observations of a galaxy, as a survey pipeline would share them, and counted
where they ran.

    python timing_table.py --fits ../evaluations/fourth.fits --out ../output/tab_timing.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catalog import Catalog

#: metacal products each measurement needs
PRODUCTS = 9


def measure(catalog: Catalog) -> dict:
    timing = catalog.key_values("TIMING")
    cols = catalog.columns("NGMIX", ["metacal_cpu_seconds", "metacal_image_cpu_seconds"],
                           list(catalog.scenes))
    images = float(np.mean(cols["metacal_image_cpu_seconds"]))
    fits = float(np.mean(cols["metacal_cpu_seconds"])) - images
    per_image = timing["shearnet.shearnet_inference_seconds"] / timing["shearnet.shearnet_images"]
    return {"images_cpu": images, "ngmix_cpu": fits, "shearnet_gpu": PRODUCTS * per_image,
            "device": timing["shearnet.device"], "observations": int(catalog.header["NRECORD"])}


def latex(t: dict) -> list:
    ms = lambda seconds: f"{1e3 * seconds:.2f}"  # noqa: E731
    ngmix_total = t["images_cpu"] + t["ngmix_cpu"]
    return [
        rf"metacal products ({PRODUCTS}) & CPU core & {ms(t['images_cpu'])} \\",
        rf"\textsc{{ngmix}} fits of the products & CPU core & {ms(t['ngmix_cpu'])} \\",
        rf"\textsc{{ShearNet}}-D4 on the products & GPU & {ms(t['shearnet_gpu'])} \\",
        r"\hline",
        rf"\textsc{{ngmix}} total & CPU core & {ms(ngmix_total)} \\",
        rf"\textsc{{ShearNet}}-D4 total & CPU core + GPU & {ms(t['images_cpu'])} + "
        rf"{ms(t['shearnet_gpu'])} \\",
        rf"Ratio of CPU time & & {ngmix_total / t['images_cpu']:.1f}$\times$ \\",
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
        "% columns: step & device & milliseconds per observation",
        f"% means over {t['observations']} observations; ShearNet on {t['device']}",
        "% excluded: rendering and JAX compilation; CPU times are one core's",
        f"% ngmix's fits are {t['ngmix_cpu'] / t['shearnet_gpu']:.0f}x ShearNet's GPU time "
        "on the same products",
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
