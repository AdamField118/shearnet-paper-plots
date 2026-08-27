# shearnet-paper-plots

Figure-generation code for the ShearNet paper. Each subdirectory produces one
family of figures; rendered output is written to `figures/` (git-ignored, so the
repo stays free of binary churn — regenerate rather than commit).

## Setup

```bash
conda env create -f environment.yaml
conda activate shearnet-plots
```

The environment pins `conda-forge` only (`nodefaults`), so it builds without
needing to accept the Anaconda commercial channel Terms of Service.

## Figures

### `architecture/`

`shearnet_d4_architecture.py` draws the flagship D4-equivariant `D4ForkLike`
model as a left-to-right dataflow diagram: the joint $D_4$ orbit over the
(galaxy, PSF) stamp pair, the shared smooth two-branch backbones, cross-attention
fusion, realignment to the reference frame, and the sign-weighted Reynolds
average that yields the spin-2 equivariant features feeding the shear head.

```bash
cd architecture
python shearnet_d4_architecture.py                  # -> ../figures/shearnet_arch_d4.{pdf,png}
python shearnet_d4_architecture.py --format png     # png only
python shearnet_d4_architecture.py -o /tmp/arch     # custom output stem
```

Pure matplotlib — no LaTeX toolchain required. Styling constants (`PALETTE`,
`FONT`, block geometry) live at the top of the module.
