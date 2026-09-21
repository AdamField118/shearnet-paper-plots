"""Every float and every panel the paper needs, and what produces it.

The current paper has five figures and three tables. Some are drawn from the
evaluation FITS, some from training logs, some are schematics with no data at
all, and two tables were cut from the campaign. Without one list it is not
possible to answer the only question that matters before submission -- is
anything missing, and is anything here that the paper does not ask for -- and
the answer drifts every time a script is added.

    python paper_manifest.py                       # what is covered
    python paper_manifest.py --runs ../evaluations # also: which runs are present
    python paper_manifest.py --strict              # exit non-zero if incomplete

SOURCES
  fits      derived from an evaluation FITS by a script in this directory
  logs      from training records, not from the benchmark
  static    a schematic or a hand-written table; no measurement
  cut       removed from the paper; nothing should produce it
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class Item:
    __slots__ = ("label", "source", "script", "note")

    def __init__(self, label, source, script=None, note=""):
        self.label = label
        self.source = source
        self.script = script
        self.note = note

    @property
    def path(self):
        return None if self.script is None else REPO / self.script

    @property
    def status(self):
        if self.source == "cut":
            # A cut item is satisfied by ABSENCE. A script still emitting one
            # is not harmless: it puts a table in the output that the paper no
            # longer has a slot for, and someone will paste it back in.
            return "still produced" if self.script and self.path.is_file() else "cut"
        if self.source in ("logs", "static"):
            return "not from the benchmark"
        if self.script is None:
            return "MISSING"
        return "ok" if self.path.is_file() else "MISSING"


#: The paper's labels, in the order they appear. Keep this list in step with
#: main.tex -- it is the definition of "what the paper needs".
MANIFEST = [
    Item("fig:d4", "static", "architecture/shearnet_d4_architecture_4plots.py", "architecture"),
    Item("fig:psf-properties", "static", "psf/psf_properties.py", "observed PSF row only"),
    Item("fig:response_snr", "fits", "results/response_vs_snr.py", "diagonal responses vs SNR"),
    Item("fig:unit-test-bias", "fits", "results/unit_test_bias.py", "m and c vs UT1--UT4"),
    Item("fig:psf-leakage", "fits", "results/psf_leakage.py", "raw shapes: --shape raw"),
    Item("tab:unit-tests", "static", None, "simulation definitions"),
    Item("tab:response-diag", "fits", "results/response_diagnostics.py", "mean response matrices"),
    Item("tab:timing", "fits", "results/timing_table.py", "original-image timing"),
    Item("fig:prediction-residuals", "cut", None, "replaced by the UT bias comparison"),
    Item("fig:snr-size", "cut", None, "removed at advisor review"),
    Item("tab:abl-main", "cut", None, "no ablation claims"),
    Item("tab:abl-secondary", "cut", None, "no ablation claims"),
]

#: Scripts that are tools, not paper deliverables. They must not be swept up by
#: a driver that runs every .py with --fits: two of them take different
#: arguments entirely and would fail the run.
NOT_DELIVERABLES = {
    "results/paper_tables.py": "optional numeric companion to the bias figure",
    "results/prediction_residuals.py": "optional diagnostic, removed from manuscript",
    "results/snr_size_dependence.py": "optional diagnostic, removed from manuscript",
    "results/paper_labels.py": "notation helper",
    "results/unit_test_bias_test.py": "bias-figure tests",
    "results/test_figures.py": "figure and statistic tests",
    "tools/reproduce_advisor_figures.py": "checked archival replay of supplied figures",

    "results/evaluation_fits.py": "library: the FITS reader",
    "results/paper_numbers.py": "library: m, c and alpha",
    "results/selection.py": "library: the sample cut",
    "results/make_fixture.py": "test fixture generator (--out, not --fits)",
    "results/diagnose_metacal.py": "diagnostic for the metacal bias, not a figure",
    "results/paper_manifest.py": "this audit",
    "results/plotstyle.py": "library: the TeX-availability probe",
    "results/test_run_all_paths.py": "tests for the driver's path handling",
}


def _runs_present(root: Path):
    from paper_tables import find_run

    return {name: find_run(root, name) for name in ("first", "second", "third", "fourth")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=None,
                        help="directory of evaluation FITS, to also report coverage")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if anything is missing or over-produced")
    args = parser.parse_args(argv)

    width = max(len(item.label) for item in MANIFEST)
    problems = []

    for source, title in (("fits", "MEASURED FROM AN EVALUATION FITS"),
                          ("logs", "FROM TRAINING RECORDS, NOT THE BENCHMARK"),
                          ("static", "SCHEMATIC OR HAND-WRITTEN"),
                          ("cut", "CUT FROM THE PAPER")):
        items = [i for i in MANIFEST if i.source == source]
        print(f"\n{title}")
        print("-" * 74)
        for item in items:
            status = item.status
            if status in ("MISSING", "still produced"):
                problems.append(item)
            mark = {"ok": "  ", "cut": "  ", "MISSING": "->",
                    "still produced": "->"}.get(status, "  ")
            print(f"{mark} {item.label:<{width}}  {status:<22} {item.script or '-'}")
            if item.note:
                print(f"   {'':<{width}}  {item.note}")

    print("\nTOOLS, NOT DELIVERABLES")
    print("-" * 74)
    for script, why in sorted(NOT_DELIVERABLES.items()):
        exists = (REPO / script).is_file()
        print(f"   {script:<34} {'present' if exists else 'absent':<8} {why}")

    stray = sorted(
        str(p.relative_to(REPO))
        for p in REPO.rglob("*.py")
        if ".git" not in p.parts
        and str(p.relative_to(REPO)) not in NOT_DELIVERABLES
        and not any(i.script == str(p.relative_to(REPO)) for i in MANIFEST)
    )
    if stray:
        print("\nNOT ACCOUNTED FOR  (neither a paper deliverable nor a declared tool)")
        print("-" * 74)
        for script in stray:
            print(f"-> {script}")
        problems.extend(stray)

    if args.runs:
        print(f"\nRUNS UNDER {args.runs}")
        print("-" * 74)
        import sys

        sys.path.insert(0, str(HERE))
        for name, path in _runs_present(args.runs).items():
            print(f"   {name:<10} {'present' if path else 'MISSING':<8} {path or ''}")

    print()
    if problems:
        print(f"{len(problems)} item(s) need attention.")
        return 1 if args.strict else 0
    print("Every paper item is accounted for, and nothing else is produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
