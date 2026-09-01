#!/bin/bash

source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/bin/activate
conda activate shearnet-plots

ROOT="$(pwd)"

# runs psf properties before the rest, due to ordering requirements
cd "$ROOT"/psf/ && python psf_properties.py --psf "$ROOT"/psf_data/emp_psfs_best/psfex-output/superbit_psf_emp01.psf && cd "$ROOT"

# recursively submit every *.py

find "$ROOT" \
    \( -type d \( -name "psf" -o -name "architecture" \) -prune \) -o \
    \( -type f -name "*.py" -print0 \) |
while IFS= read -r -d '' file; do
    dir="$(dirname "$file")"
    script="$(basename "$file")"

    echo "Running: $file"

    (
        cd "$dir" || exit 1
        python "$script" --fits "$ROOT"/evaluation.fits
    )
done

# folow up with unique scripts excluded
cd "$ROOT"/architecture/ && python shearnet_d4_architecture_4plots.py && cd "$ROOT"
