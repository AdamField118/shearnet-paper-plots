# shearnet-paper-plots

Figure-generation code for the ShearNet paper. Each subdirectory produces one
family of figures; rendered output is written to `figures/` (git-ignored, so the
repo stays free of binary churn — regenerate rather than commit).

## ShearNet dependency

Most of this repo does **not** import ShearNet. One notebook does
(`architecture/draw_single_stamp.ipynb`), because it renders stamps through
ShearNet's own dataset generator rather than reimplementing the simulation.

> **Pinned ShearNet version**
>
> | | |
> |---|---|
> | repository | `s-Sayan/ShearNet` |
> | commit | **`f60d0c447c666eabdbaa388e9ec0b8be8a15bb52`** (`f60d0c4`) |
> | commit subject | *working to standardize the way we measure responses across the board* |
> | date pinned | 2026-08-27 |

Pin it explicitly before installing:

```bash
git clone https://github.com/s-Sayan/ShearNet.git
cd ShearNet
git checkout f60d0c447c666eabdbaa388e9ec0b8be8a15bb52
pip install -e .          # into the shearnet-plots-sim env, see below
```

**Why an exact commit, not a version range.** ShearNet is not released to PyPI and
its `pyproject.toml` version (`0.1.0`) has not moved across many breaking changes to
the dataset and response APIs. The notebook depends on API that is specific and
recent — `DatasetSpec.from_config`, the `paths.*` unit-test config schema, and
per-object seeding in `generate_dataset` — so a commit hash is the only meaningful
pin. The notebook records the commit it actually ran against into every `.npz` it
writes, so a stamp file can always be traced back to the code that produced it.

When you bump the pin, update the table above and re-run the notebook: the stored
provenance and this README should never disagree.

## Environments

Two environments, because the plotting stack is light and the simulation stack is
not. Neither uses Anaconda's commercial channels (`conda-forge` + `nodefaults`), so
they build without accepting the Anaconda Terms of Service.

**`environment.yaml` → `shearnet-plots`** — numpy/matplotlib/scipy/astropy. Enough
to render every figure from data that already exists.

```bash
conda env create -f environment.yaml
conda activate shearnet-plots
```

**`environment-shearnet.yaml` → `shearnet-plots-sim`** — the above plus Jupyter,
GalSim, ngmix and JAX. Needed only to run the stamp-drawing notebook. Install the
pinned ShearNet into it afterwards, as shown above.

```bash
conda env create -f environment-shearnet.yaml
conda activate shearnet-plots-sim
pip install -e /path/to/ShearNet
```

## Figures

### `architecture/`

**`shearnet_d4_architecture.py`** draws the flagship D4-equivariant `D4ForkLike`
model as a dataflow diagram: the joint $D_4$ orbit over the (galaxy, PSF) stamp
pair, the shared smooth two-branch backbones, cross-attention fusion, realignment
to the reference frame, and the sign-weighted Reynolds average that produces the
spin-2 equivariant features feeding the shear head.

```bash
cd architecture
python shearnet_d4_architecture.py                  # -> ../figures/shearnet_arch_d4.{pdf,png}
python shearnet_d4_architecture.py --format png     # png only
python shearnet_d4_architecture.py --usetex         # real LaTeX text if available
python shearnet_d4_architecture.py -o /tmp/arch     # custom output stem
```

Pure matplotlib — no LaTeX toolchain required (`--usetex` is opt-in and falls back
to mathtext when TeX is absent). Styling constants (`PALETTE`/`C`, `F`, block
geometry) live at the top of the module.

**`draw_single_stamp.ipynb`** renders **exactly one** galaxy through ShearNet's own
`generate_dataset`, using the same PSF model, detection catalog and random seed as
the fiducial run, and saves the dirty (PSF-convolved, noisy) galaxy stamp together
with its PSF stamp to `architecture/stamps/single_stamp.npz`. Run it in the
`shearnet-plots-sim` environment.

Every simulation setting is read from a ShearNet config file
(`research/unit_tests/fourth/config.yaml` by default) via `DatasetSpec.from_config`,
the same object the training CLI builds — so the notebook cannot drift from the real
run. Set `SHEARNET_REPO` and `CONFIG_PATH` in the parameters cell to point at your
checkout; `POPULATION` chooses the training catalog/seed or the held-out eval
catalog/seed.

Two things worth knowing:

- **`samples=1` is exact, not an approximation.** `generate_dataset` seeds every
  object independently, so a one-object call returns byte-for-byte the stamp that
  object `0` of the full 300k-object run would get. The notebook asserts this at the
  end by re-rendering a larger population and comparing.
- **A missing catalog fails loudly.** ShearNet silently falls back to a *synthetic
  random* catalog when the detection catalog is absent, which would quietly yield a
  galaxy that is not from the real population. The notebook checks both the PSF model
  and the catalog exist and raises before generating.

`architecture/stamps/` is git-ignored. Once you have generated the real stamp file
you may want to commit it anyway, so the architecture figure can be rebuilt from the
light `shearnet-plots` environment alone, without ShearNet or cluster data access.
