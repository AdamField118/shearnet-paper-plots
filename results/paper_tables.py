"""Turn a directory of evaluation FITS into the paper's table rows.

Point it at a tree of finished runs and it emits LaTeX for every table it can
fill, and names what is still missing:

    python paper_tables.py --runs /path/to/runs
    python paper_tables.py --runs /path/to/runs --table unit-test-bias
    python paper_tables.py --runs /path/to/runs --status

A run is a directory whose name matches a row below, containing
``evaluation.fits`` (or ``<name>/benchmarking/evaluation.fits``). Runs that have
not finished are reported as pending rather than silently omitted -- a table
quietly missing a row is how a half-finished campaign gets written up as a whole
one.

Every number comes from :mod:`paper_numbers`, which reads SUMMARY for m1,
recomputes c2 because the run's ``c_convention`` makes SUMMARY's ``c`` the wrong
component, and takes alpha from ``superbit_lensing``'s own PSFLeakagePanelMaker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_fits import Evaluation
from paper_numbers import run_numbers

#: (run directory name, LaTeX row label) in the order the table prints them.
TABLES = {
    "unit-test-bias": {
        "caption": "tab:unit-test-bias -- m1 and c2 across the simulation ladder",
        "estimators": ("shearnet", "ngmix"),
        "columns": ("m1", "c2"),
        "rows": [("first", "UT1"), ("second", "UT2"), ("third", "UT3"),
                 ("fourth", r"\textbf{UT4}")],
    },
    "abl-main": {
        "caption": "tab:abl-main -- architecture ladder and response objectives",
        "estimators": ("shearnet",),
        "columns": ("m1", "c2", "alpha", "shape_noise"),
        "rows": [
            ("tier2/01_galaxy_only", "Galaxy image only (no PSF branch)"),
            ("tier2/02_psf_branch_concat", r"{}+ PSF branch, concat fusion"),
            ("tier2/03_transformer_fusion", r"{}+ transformer fusion"),
            ("tier2/04_auxiliary_targets", r"{}+ auxiliary targets $(r_{1/2}, F)$"),
            ("tier2/05_d4_augmentation", r"{}+ $D_4$ augmentation (not equivariant)"),
            ("tier2/06_d4_equivariant", r"{}+ $D_4$ equivariance (exact)"),
            ("tier2/07_inloop_rendering", r"{}+ in-loop rendering (fresh noise)"),
            ("tier2/08_learned_pooling_head", r"{}+ learned pooling head ($K = 4$)"),
            ("fourth", r"\textbf{{}+ relative 2D RoPE fusion encoding}"),
            (None, None),  # \hline
            ("fourth", r"\textbf{Fiducial model (all terms)}"),
            ("tier3/no_gamma_response", r"{}$-$ shear response $\lambda_\gamma$"),
            ("tier1/no_psf_response", r"{}$-$ PSF response $\lambda_{\rm PSF}$"),
            ("tier3/no_shift_response", r"{}$-$ translation $\lambda_{\rm shift}$"),
            ("tier3/no_complement", r"{}$-$ complement $\lambda_\perp$"),
            ("tier3/no_psf_orbit", r"{}$-$ PSF orbit $\lambda_{\rm orbit}$"),
            ("tier3/no_isotropy", r"{}$-$ isotropy $\lambda_{\rm iso}$"),
            (None, None),
            ("tier3/orbit_k2", r"Orbit $K = 2$ instead of $K = 4$"),
            ("tier3/target_identity", r"Target \texttt{identity} instead of \texttt{analytic}"),
        ],
    },
    "abl-secondary": {
        "caption": "tab:abl-secondary -- backbone, training transforms, loss",
        "estimators": ("shearnet",),
        "columns": ("m1", "alpha", "shape_noise"),
        "rows": [
            ("fourth", r"\textbf{Fiducial $(16,48,64)$, depths $(1,2,1)/(0,1,1)$}"),
            ("tier4/no_multiscale_block", r"{}$-$ multi-scale context block"),
            ("tier4/untrimmed_stem", r"{}$+$ untrimmed stem width $(32,48,64)$"),
            ("tier4/full_resolution_psf_block", r"{}$+$ full-resolution PSF block"),
            (None, None),
            ("tier4/no_label_normalization", r"{}$-$ label normalization (raw labels)"),
            ("tier4/no_ema", r"{}$-$ EMA of weights"),
            ("tier4/no_image_standardization", r"{}$-$ input-image standardization"),
            (None, None),
            ("fourth", r"\textbf{Mean squared error (adopted)}"),
            ("tier4/loss_mae", "Mean absolute error"),
            ("tier4/loss_huber", r"Huber ($\delta = 1$)"),
        ],
    },
}

#: How each column is scaled and formatted for LaTeX, matching the paper's
#: column headers (``m_1 (10^{-3})``, ``c_2 (10^{-5})``, ``|alpha| (10^{-3})``).
FORMAT = {
    "m1": (1e3, "{:+.2f}"),
    "c2": (1e5, "{:+.2f}"),
    "alpha": (1e3, "{:.2f}"),
    "shape_noise": (1.0, "{:.3f}"),
}


def find_run(root: Path, name: str):
    """The evaluation FITS for one run name, or None if it has not finished."""
    for candidate in (root / name / "evaluation.fits",
                      root / name / "benchmarking" / "evaluation.fits",
                      root / f"{name.replace('/', '_')}.fits"):
        if candidate.is_file():
            return candidate
    return None


def cell(numbers: dict, column: str) -> str:
    """One formatted LaTeX cell, or ``\\pending`` when the number is missing."""
    scale, template = FORMAT[column]
    if column == "shape_noise":
        value = numbers.get("shape_noise")
        return r"\pending" if value is None else template.format(value[0])
    value, error = numbers.get(column), numbers.get(f"{column}_err")
    if value is None:
        return r"\pending"
    body = template.format(value * scale)
    if error is not None and error == error:  # not NaN
        body += r" \pm " + template.format(error * scale).lstrip("+")
    return f"${body}$"


def build_table(spec: dict, root: Path, njack: int = 20):
    """``(latex_lines, missing_run_names)`` for one table."""
    lines, missing, cache = [], [], {}
    for name, label in spec["rows"]:
        if name is None:
            lines.append(r"\hline")
            continue
        path = find_run(root, name)
        if path is None:
            missing.append(name)
            blanks = [r"\pending"] * (len(spec["columns"]) * len(spec["estimators"]))
            lines.append(f"{label} & " + " & ".join(blanks) + r" \\")
            continue
        if name not in cache:
            evaluation = Evaluation(path)
            cache[name] = {est: run_numbers(evaluation, est, njack=njack)
                           for est in spec["estimators"]}
        cells = [cell(cache[name][est], column)
                 for est in spec["estimators"] for column in spec["columns"]]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return lines, missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True,
                        help="directory holding one subdirectory per run")
    parser.add_argument("--table", choices=sorted(TABLES), default=None,
                        help="just one table (default: all of them)")
    parser.add_argument("--status", action="store_true",
                        help="only report which runs are present")
    parser.add_argument("--njack", type=int, default=20)
    args = parser.parse_args(argv)

    wanted = [args.table] if args.table else sorted(TABLES)

    if args.status:
        every = {name for key in TABLES for name, _ in TABLES[key]["rows"] if name}
        found = {n: find_run(args.runs, n) for n in sorted(every)}
        ready = [n for n, p in found.items() if p]
        pending = [n for n, p in found.items() if not p]
        print(f"{len(ready)}/{len(found)} runs present under {args.runs}\n")
        for name in ready:
            print(f"  ready    {name}")
        for name in pending:
            print(f"  PENDING  {name}")
        return 0

    for key in wanted:
        spec = TABLES[key]
        lines, missing = build_table(spec, args.runs, njack=args.njack)
        print(f"% {spec['caption']}")
        print(f"% columns: " + ", ".join(
            f"{est} {col}" for est in spec["estimators"] for col in spec["columns"]))
        if missing:
            print(f"% NOT YET AVAILABLE ({len(missing)}): " + ", ".join(missing))
        print("\n".join(lines))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
