"""Draw only the joint PSF distribution from the two completed UT4 catalogs."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from catalog import Catalog, find_run
from psf_distribution import corner_comparison
from shear_stats import PAPER_CUT, Sample, cut_from_name, leakage_inputs


def selected_pairs(path, cut=PAPER_CUT, chunk=4096):
    """Use the leakage fit's sample and ring average, without fitting any slopes."""
    parts = []
    with Catalog(path) as cat:
        for start in range(0, cat.n_objects, chunk):
            sample = Sample(cat, ["zero"], cut=cut, objects=slice(start, start + chunk))
            values = leakage_inputs(sample, "shearnet", variant="noshear")
            parts.append(np.column_stack((values["e1_psf"], values["e2_psf"])))
    result = np.concatenate(parts)
    if not result.size:
        raise ValueError(f"{path}: no complete rings pass the leakage selection")
    return result


def main(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True, help="directory with fourth and fourth_worstpsf")
    p.add_argument("--out", type=Path, default=Path("output/paper/psf_ellipticity_corner.pdf"))
    p.add_argument("--cut", choices=["none", "metacal"], default="metacal")
    p.add_argument("--min-t-ratio", type=float)
    p.add_argument("--min-s2n", type=float)
    args = p.parse_args(argv)
    paths = [find_run(args.runs, name) for name in ("fourth", "fourth_worstpsf")]
    if None in paths:
        p.error("both fourth and fourth_worstpsf must have a finished merged catalog under --runs")
    if paths[0].samefile(paths[1]):
        p.error("fourth and fourth_worstpsf resolve to the same file")
    cut = cut_from_name(args.cut, args.min_t_ratio, args.min_s2n)
    fig = corner_comparison(*(selected_pairs(path, cut) for path in paths))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
