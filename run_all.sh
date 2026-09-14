#!/bin/bash
#
# Produce every figure and table the paper needs, and nothing else.
#
#   ./run_all.sh                                    # dry run: print, produce nothing
#   ./run_all.sh --go --fits evaluations/fourth.fits --runs evaluations
#   ./run_all.sh --go --fits ... --cut both --min-resolution 1.0
#
# The previous version ran `find . -name "*.py"` and handed every match
# `--fits`. That worked while every script was a figure, and stopped working
# once results/ also held the FITS reader, the bias library, the sample cut, a
# fixture generator that takes `--out` rather than `--fits`, and the metacal
# diagnostic: five things that are not paper deliverables, two of which would
# have failed the run outright. The list below is explicit for that reason, and
# results/paper_manifest.py checks it against the paper's own labels.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GO=0
FITS=""
RUNS=""
OUTDIR="$ROOT/output"
CUT_ARGS=()
TABLE_ARGS=()

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --go                  actually run (default is a dry run)
  --fits PATH           evaluation FITS for the fiducial run (UT4)
  --runs DIR            directory of per-run FITS, for tab:unit-test-bias
  --out DIR             where figures and .tex fragments go (default: output/)
  --cut WHICH           none | superbit | truth | both  (passed to each script
                        that takes a sample cut)
  --min-resolution R    size floor in PSF half-widths, with --cut truth/both
  --min-hlr H           size floor in arcsec
  --psf-fwhm F          PSF FWHM in arcsec (default 0.5)
  -h, --help            this

fig:psf-properties and the architecture schematics are NOT run here: they read
the PSFEx model and synthetic profiles, not an evaluation FITS. Run
psf/psf_properties.py and architecture/shearnet_d4_architecture_4plots.py
directly.
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
        --cut) CUT_ARGS+=(--cut "$2"); shift 2 ;;
        --min-resolution) CUT_ARGS+=(--min-resolution "$2"); shift 2 ;;
        --min-hlr) CUT_ARGS+=(--min-hlr "$2"); shift 2 ;;
        --psf-fwhm) CUT_ARGS+=(--psf-fwhm "$2"); shift 2 ;;
        --correction) TABLE_ARGS+=(--correction "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# Check the inputs before running anything. Seven scripts each raising the same
# FileNotFoundError is seven tracebacks describing one typo, with the only
# useful part -- which path, and what is actually there -- buried in the last.
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

# label | directory | script | extra arguments | takes a sample cut?
DELIVERABLES=(
    "tab:response-diag|results|response_diagnostics.py|--out $OUTDIR/tab_response_diag.tex|cut"
    "tab:timing|results|timing_table.py|--out $OUTDIR/tab_timing.tex|"
    "fig:psf_leakage|results|psf_leakage.py||"
    "fig:snr_size|results|snr_size_dependence.py||cut"
    "fig:prediction-residuals|results|prediction_residuals.py|--out $OUTDIR/prediction_residuals.pdf|cut"
    "fig:response_snr|results|response_vs_snr.py|--out $OUTDIR/response_vs_snr.pdf|cut"
)

FAILED=()

run() {
    local label="$1" dir="$2" script="$3"; shift 3
    if [[ "$GO" -eq 0 ]]; then
        printf '  %-26s %s/%s %s\n' "$label" "$dir" "$script" "$*"
        return 0
    fi
    printf '\n=== %s ===\n' "$label"
    # Deliberately not fatal. One deliverable failing -- a missing column, an
    # optional dependency absent -- must not cost the other five, and the run
    # that produced nothing because the first script died is worse than a run
    # that produced five figures and named the sixth.
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
        if [[ -n "$takes_cut" && ${#CUT_ARGS[@]} -gt 0 ]]; then
            args+=("${CUT_ARGS[@]}")
        fi
        run "$label" "$dir" "$script" "${args[@]}"
    done
else
    echo "  (no --fits given; skipping the six single-run deliverables)"
fi

# tab:unit-test-bias spans the four rungs, so it takes a directory of runs
# rather than one file.
if [[ -n "$RUNS" ]]; then
    tb_args=(--runs "$RUNS")
    # The same cut and corrections as every other deliverable. Without them
    # this table reads SUMMARY over the whole population and reports both
    # estimators through metacal, which is neither the sample nor the pipeline
    # the rest of the paper uses.
    [[ ${#CUT_ARGS[@]} -gt 0 ]] && tb_args+=("${CUT_ARGS[@]}")
    [[ ${#TABLE_ARGS[@]} -gt 0 ]] && tb_args+=("${TABLE_ARGS[@]}")
    run "tab:unit-test-bias" "results" "paper_tables.py" "${tb_args[@]}"
else
    echo "  (no --runs given; skipping tab:unit-test-bias)"
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
