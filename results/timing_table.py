"""tab:timing -- shape-inference throughput, GPU against a CPU pool.

The caption is specific about what the number is and is not. It covers the
catalog measurement step only: image rendering, JAX compilation and the extra
perturbed images metacalibration needs are all excluded, for both methods. It
is a wall-clock comparison between different hardware, not a
hardware-normalized algorithmic speedup, and the table says so.

Everything here comes from the primary header, which ``run.py`` fills from the
timing pass::

    RENDER_S   seconds to render the population (EXCLUDED from the table)
    INFERENC   seconds of ShearNet inference on one GPU
    NGMIX_SE   seconds of ngmix fitting across the CPU pool, when recorded
    NGMIX_NP   pool size
    SAMPLES    objects measured

Nothing is recomputed and nothing is timed here: a timing produced on this
laptop would describe this laptop. If a field the paper needs is absent from
the header the row is emitted as ``\\pending`` rather than guessed, because a
plausible-looking throughput is worse than a visible gap.

    python timing_table.py --fits ../evaluations/fourth.fits
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_fits import Evaluation

#: Header key -> what it measures. Spellings are the 8-character FITS
#: truncations of run.py's config names.
KEYS = {
    "inference": ("INFERENC", "ShearNet inference, one GPU"),
    "ngmix": ("NGMIX_SE", "ngmix fits, CPU pool"),
    "nproc": ("NGMIX_NP", "CPU pool size"),
    "samples": ("SAMPLES", "objects measured"),
    "render": ("RENDER_S", "rendering (excluded from the table)"),
}


def collect(evaluation) -> dict:
    header = evaluation.header
    out = {name: header.get(key) for name, (key, _) in KEYS.items()}
    samples = out.get("samples")
    for method in ("inference", "ngmix"):
        seconds = out.get(method)
        out[f"{method}_rate"] = (
            float(samples) / float(seconds)
            if seconds and samples and float(seconds) > 0 else None
        )
    inference, ngmix = out.get("inference_rate"), out.get("ngmix_rate")
    out["speedup"] = (ngmix and inference and inference / ngmix) or None
    return out


def _cell(value, template="{:.1f}"):
    return r"\pending" if value is None else template.format(value)


def latex_rows(values: dict):
    samples = values.get("samples")
    nproc = values.get("nproc")
    pool = f" ({int(nproc)} cores)" if nproc else ""
    return [
        r"\textsc{ShearNet}-D4 (1 GPU) & "
        + _cell(values.get("inference"), "{:.2f}") + " & "
        + _cell(values.get("inference_rate"), "{:.0f}") + r" \\",
        rf"\textsc{{ngmix}} (CPU pool{pool}) & "
        + _cell(values.get("ngmix"), "{:.2f}") + " & "
        + _cell(values.get("ngmix_rate"), "{:.0f}") + r" \\",
        r"\hline",
        r"Ratio & & " + _cell(values.get("speedup"), r"{:.0f}$\times$") + r" \\",
    ], samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    evaluation = Evaluation(args.fits)
    values = collect(evaluation)
    rows, samples = latex_rows(values)

    missing = [name for name in ("inference", "ngmix", "samples")
               if values.get(name) is None]

    lines = [
        f"% tab:timing from {args.fits.name}",
        r"% columns: method & wall-clock (s) & objects s^-1",
        f"% objects measured: {int(samples) if samples else 'unknown'}",
        r"% EXCLUDED for both methods: rendering, JAX compilation, and the",
        r"% perturbed images used for the metacalibration responses.",
        f"% rendering took {_cell(values.get('render'), '{:.1f}')} s and is not in the table.",
    ]
    if missing:
        lines.append("% NOT IN THIS FILE'S HEADER: " + ", ".join(
            f"{name} ({KEYS[name][0]})" for name in missing))
    lines += rows
    text = "\n".join(lines)
    print(text)

    if missing:
        print(f"\n% {len(missing)} field(s) absent -- rows emitted as \\pending.")
        print("% The header records what run.py measured; a number invented here")
        print("% would describe whatever machine this script ran on.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"% wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
