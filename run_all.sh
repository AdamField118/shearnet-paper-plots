#!/bin/bash
#
# Produce every figure and table the paper draws from the evaluation catalogs.
#
#   ./run_all.sh                                    # dry run: print, produce nothing
#   ./run_all.sh --go --fits evaluations/fourth.fits --runs evaluations
#   ./run_all.sh --go --runs ~/ShearNet/runs/unit_tests \
#                --fits ~/ShearNet/runs/unit_tests/fourth/evaluations/default/d4_unit_fourth_default.fits
#
# The catalogs are the ones `shearnet-eval` writes (schema shearnet-eval v1).
# --runs may be a directory of first.fits ... fourth.fits, or ShearNet's
# runs/unit_tests itself, so nothing has to be copied.
#
# THE CUT. Table 2 and Figures 4 and 5 are measured on the sample that passes
# ngmix's own fit: T/Tpsf > 1 and s2n > 10 (flags == 0), applied right before m,
# c and R are formed, with metacal's selection response R^S in the calibration
# (results/shear_stats.py says exactly how). Figure 3 is NEVER cut: it shows the
# whole population, because it is what motivates the cut. --cut none turns the
# cut off everywhere else too.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GO=0
FITS=""
RUNS=""
OUTDIR="$ROOT/output"
CUT_ARGS=(--cut metacal)

usage() {
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --go                  actually run (default is a dry run)
  --fits PATH           the fiducial (UT4) catalog: Table 2, Figures 3 and 5
  --runs DIR            the four unit-test catalogs: Figure 4
  --out DIR             where figures and .tex fragments go (default: output/)
  --cut metacal|none    the sample cut (default metacal); never applied to Figure 3
  --min-t-ratio X       T/Tpsf threshold of the metacal cut (default 1)
  --min-s2n X           s2n threshold of the metacal cut (default 10)
  -h, --help            this

fig:psf-properties and the architecture schematic do not read a catalog: run
psf/psf_properties.py and architecture/shearnet_d4_architecture_4plots.py
directly. tab:timing is produced outside this repository.
USAGE
}

# Each deliverable runs with its own directory as the working directory, so a
# path given relative to where YOU invoked this would resolve against results/
# instead. Everything a script receives is made absolute here, once.
abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s\n' "$PWD/$1" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --go) GO=1; shift ;;
        --fits) FITS="$(abspath "$2")"; shift 2 ;;
        --runs) RUNS="$(abspath "$2")"; shift 2 ;;
        --out) OUTDIR="$(abspath "$2")"; shift 2 ;;
        --cut) CUT_ARGS[1]="$2"; shift 2 ;;
        --min-t-ratio) CUT_ARGS+=(--min-t-ratio "$2"); shift 2 ;;
        --min-s2n) CUT_ARGS+=(--min-s2n "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# Check the inputs before running anything: one clear message rather than one
# traceback per script describing the same typo.
if [[ -n "$FITS" && ! -f "$FITS" ]]; then
    echo "--fits does not exist: $FITS" >&2
    parent="$(dirname "$FITS")"
    if [[ -d "$parent" ]]; then
        echo "FITS files in $parent:" >&2
        find -L "$parent" -maxdepth 1 -name "*.fits" -printf '  %f\n' 2>/dev/null | sort >&2
    else
        echo "  (its directory does not exist either)" >&2
    fi
    exit 1
fi
if [[ -n "$RUNS" && ! -d "$RUNS" ]]; then
    echo "--runs is not a directory: $RUNS" >&2
    exit 1
fi

echo "--- what the paper asks for ---"
python "$ROOT/results/paper_manifest.py" ${RUNS:+--runs "$RUNS"} | tail -n 2
echo

# label | directory | script | extra arguments | takes the cut?
DELIVERABLES=(
    "tab:response-diag|results|response_diagnostics.py|--out $OUTDIR/tab_response_diag.tex|cut"
    "fig:response_snr|results|response_vs_snr.py|--out $OUTDIR/response_vs_snr.pdf|"
    "fig:psf-leakage|results|psf_leakage.py|--shape raw --out $OUTDIR/psf_leakage|cut"
)

FAILED=()

run() {
    local label="$1" dir="$2" script="$3"; shift 3
    if [[ "$GO" -eq 0 ]]; then
        printf '  %-26s %s/%s %s\n' "$label" "$dir" "$script" "$*"
        return 0
    fi
    printf '\n=== %s ===\n' "$label"
    # Deliberately not fatal: one deliverable failing must not cost the others.
    if ! ( cd "$ROOT/$dir" && python "$script" "$@" ); then
        echo "!!! $label FAILED (see above)" >&2
        FAILED+=("$label")
    fi
}

if [[ "$GO" -eq 1 ]]; then
    mkdir -p "$OUTDIR"
else
    echo "DRY RUN -- nothing will be produced. Add --go."
fi

if [[ -n "$FITS" ]]; then
    for entry in "${DELIVERABLES[@]}"; do
        IFS='|' read -r label dir script extra takes_cut <<< "$entry"
        args=(--fits "$FITS")
        # shellcheck disable=SC2206
        [[ -n "$extra" ]] && args+=($extra)
        [[ -n "$takes_cut" ]] && args+=("${CUT_ARGS[@]}")
        run "$label" "$dir" "$script" "${args[@]}"
    done
else
    echo "  (no --fits given; skipping the single-run deliverables)"
fi

# fig:unit-test-bias spans the four unit tests, so it takes a directory.
if [[ -n "$RUNS" ]]; then
    run "fig:unit-test-bias" "results" "unit_test_bias.py" --runs "$RUNS" \
        "${CUT_ARGS[@]}" --out "$OUTDIR/unit_test_bias.pdf"
else
    echo "  (no --runs given; skipping fig:unit-test-bias)"
fi

echo
if [[ "$GO" -eq 1 ]]; then
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "Output in $OUTDIR, but ${#FAILED[@]} deliverable(s) FAILED:"
        printf '  %s\n' "${FAILED[@]}"
        exit 1
    fi
    echo "Done. Every deliverable produced. Output in $OUTDIR"
else
    echo "Nothing produced. Re-run with --go."
fi
