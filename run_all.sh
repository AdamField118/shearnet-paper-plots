#!/bin/bash
#
# Produce every figure and table the paper draws from the evaluation catalogs.
#
#   ./run_all.sh                                    # dry run: print, produce nothing
#   ./run_all.sh --go --fits evaluations/fourth.fits --runs evaluations
#   ./run_all.sh --go --runs ~/weak_lensing/ShearNet/runs/unit_tests
#
# The catalogs are ShearNet's merged catalogs (schema shearnet-eval v2).
# --runs may be a directory of first.fits ... fourth.fits, or ShearNet's
# runs/unit_tests itself (the newest finished generation of each), so nothing
# has to be copied. Without --fits, the fourth unit test found there is used.
#
# THE CUT. Table 2 and Figures 4 and 5 are measured on the sample that passes
# ngmix's own fit: T/T_PSF > 1 and s2n > 10 (flags == 0), applied right before m,
# c and R are formed, with metacal's selection response R^S in the calibration
# (results/shear_stats.py says exactly how). Figure 3 is NEVER cut: it shows the
# whole population, because it is what motivates the cut. --cut none turns the
# cut off everywhere else too.
#
# THE SHAPES. Both estimators are calibrated the same way: their metacal noshear
# shape divided by their own R = <R^gamma> + R^S. The paper subtracts metacal's PSF response
# for ngmix only. Both estimators can retain residual leakage.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GO=0
FITS=""
RUNS=""
ABELL=""
OUTDIR="$ROOT/output"
CUT_ARGS=(--cut metacal)

usage() {
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --go                  actually run (default is a dry run)
  --fits PATH           the fiducial (UT4) catalog: Table 2, Figures 3 and 5
                        (default: the fourth unit test under --runs)
  --abell3411 FITS      Abell 3411 UT4; defaults to fourth_worstpsf under --runs
  --runs DIR            the four unit-test catalogs: Figure 4
  --out DIR             where figures and .tex fragments go (default: output/)
  --cut metacal|none    the sample cut (default metacal); never applied to Figure 3
  --min-t-ratio X       T/T_PSF threshold of the metacal cut (default 1)
  --min-s2n X           s2n threshold of the metacal cut (default 10)
  -h, --help            this

Figure 2 reproduces published Figure C1 of LITB III. The paired leakage
plots and PSF ellipticity corner plot require both UT4 catalogs.
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
        --abell3411) ABELL="$(abspath "$2")"; shift 2 ;;
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
find_catalog() {
    python -c 'import sys; sys.path.insert(0, sys.argv[1])
from catalog import find_run
print(find_run(sys.argv[2], sys.argv[3]) or "")' "$ROOT/results" "$RUNS" "$1"
}
if [[ -n "$RUNS" ]]; then
    [[ -n "$FITS" ]] || FITS="$(find_catalog fourth)"
    [[ -n "$ABELL" ]] || ABELL="$(find_catalog fourth_worstpsf)"
fi
if [[ -n "$ABELL" && ( ! -f "$ABELL" || ! -f "$FITS" ) ]]; then
    echo "The Abell 3411 comparison needs both merged UT4 catalogs." >&2
    exit 1
fi
echo "--- what the paper asks for ---"
python "$ROOT/results/paper_manifest.py" ${RUNS:+--runs "$RUNS"} | tail -n 2
echo

# label | directory | script | extra arguments | takes the cut?
DELIVERABLES=(
    "tab:response-diag|results|response_diagnostics.py|--out $OUTDIR/tab_response_diag.tex|cut"
    "fig:response_snr|results|response_vs_snr.py|--out $OUTDIR/response_vs_snr.pdf|"
    "tab:timing|results|timing_table.py|--out $OUTDIR/tab_timing.tex|"
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
        [[ -n "$extra" ]] && args+=(--out "${extra#--out }")
        [[ -n "$takes_cut" ]] && args+=("${CUT_ARGS[@]}")
        run "$label" "$dir" "$script" "${args[@]}"
    done
    if [[ -n "$ABELL" ]]; then
        run "fig:psf-leakage + corner" results psf_leakage.py --fits "$FITS" \
            --abell3411 "$ABELL" --rpsf paper+off "${CUT_ARGS[@]}" --out "$OUTDIR/psf_leakage_distribution"
    else
        run "fig:psf-leakage" results psf_leakage.py --fits "$FITS" "${CUT_ARGS[@]}" --out "$OUTDIR/psf_leakage"
        echo "  (no Abell 3411 catalog; skipping the paired PSF comparison)"
    fi
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
