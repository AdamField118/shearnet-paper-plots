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

``tab:abl-main`` and ``tab:abl-secondary`` were emitted here until the ablation
campaign was cut from the paper. They are gone rather than disabled: a table
this still produced would be one the paper has no slot for, and the twenty-four
runs behind it no longer exist. ``results/paper_manifest.py`` records the cut.

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
