#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=../figures
ppi=300
mkdir -p "$out"
for source in panels/*.typ; do
  panel=$(basename "$source" .typ)
  stem="$out/shearnet_arch_d4_$panel"
  typst compile --input "panel=$panel" shearnet.typ "$stem.pdf"
  typst compile --input "panel=$panel" --ppi "$ppi" shearnet.typ "$stem.png"
done
