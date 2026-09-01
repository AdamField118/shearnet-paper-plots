# shearnet-paper-plots

Figure-generation code for the ShearNet paper. Each subdirectory produces one
family of figures; rendered output is written to `figures/` (git-ignored, so the
repo stays free of binary churn — regenerate rather than commit).

## House rule: defer to `superbit_lensing`

**Any plot that the SuperBIT pipeline already knows how to make is made by calling
`superbit_lensing`, not by reimplementing it here.** These figures sit next to the
SuperBIT weak-lensing paper, and a second implementation of the same diagnostic is a
second thing that can silently disagree with it. Scripts in this repo are thin: they
resolve inputs, cache expensive intermediate products, and save output. The moment
fitting, colour maps, colour limits and panel layout come from upstream.

The practical consequence is that upstream's choices win even when they are
inconvenient — see the note on hard-coded colour limits under `psf/` below. If one of
those choices needs to change, change it in `superbit_lensing` so both papers move
together.

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
pip install -e .          # into the shearnet-plots env, see below
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

## superbit_lensing dependency

The PSF figures import `superbit_lensing` (see the house rule above).

> **Pinned superbit-lensing version**
>
> | | |
> |---|---|
> | repository | `superbit-collaboration/superbit-lensing` |
> | commit | **`69a4e5d9812239b01c9c58bbc0cf09bf01e0508b`** (`69a4e5d`) |
> | commit subject | *long due update* |
> | date pinned | 2026-08-28 |

```bash
git clone https://github.com/superbit-collaboration/superbit-lensing.git
cd superbit-lensing
git checkout 69a4e5d9812239b01c9c58bbc0cf09bf01e0508b
pip install -e .
```

It is not on PyPI either, so the same commit-pin reasoning applies. If you would
rather not install it, point `SUPERBIT_LENSING_DIR` at the checkout and the scripts
will add it to `sys.path`:

```bash
export SUPERBIT_LENSING_DIR=/path/to/superbit-lensing
```

Its import chain pulls in several packages beyond ShearNet's own dependency list —
`psfex`, `piff`, `treecorr`, `pyregion`, `shapely`, `colossus`, `astroquery`, `ipdb`.
`psfex` is not on PyPI; install it from conda-forge (`conda install -c conda-forge
psfex`).

## Environment

One environment, `environment.yml` → **`shearnet-plots`**, covering both the plotting
stack and the simulation stack (Jupyter, GalSim, ngmix, JAX). It is a full export with
exact versions, so it reproduces a known-good solve rather than re-resolving, and it
builds from `conda-forge` without needing the Anaconda Terms of Service.

```bash
conda env create -f environment.yml
conda activate shearnet-plots
```

ShearNet and superbit_lensing are **not** in the environment file — install them
separately at the commits pinned above, so their versions stay explicit decisions.

## Figures

### `architecture/`

**`shearnet_d4_architecture_4plots.py`** draws the flagship D4-equivariant `D4ForkLike`
model as a dataflow diagram: the joint $D_4$ orbit over the (galaxy, PSF) stamp
pair, the shared smooth two-branch backbones, cross-attention fusion, realignment
to the reference frame, and the sign-weighted Reynolds average that produces the
spin-2 equivariant features feeding the shear head.

```bash
cd architecture
python shearnet_d4_architecture_4plots.py                  # -> ../figures/shearnet_arch_d4.{pdf,png}
python shearnet_d4_architecture_4plots.py --format png     # png only
python shearnet_d4_architecture_4plots.py --usetex         # real LaTeX text if available
python shearnet_d4_architecture_4plots.py -o /tmp/arch     # custom output stem
```

Pure matplotlib — no LaTeX toolchain required (`--usetex` is opt-in and falls back
to mathtext when TeX is absent). Styling constants (`PALETTE`/`C`, `F`, block
geometry) live at the top of the module.

**`draw_single_stamp.ipynb`** renders **exactly one** galaxy through ShearNet's own
`generate_dataset`, using the same PSF model, detection catalog and random seed as
the fiducial run, and saves the dirty (PSF-convolved, noisy) galaxy stamp together
with its PSF stamp to `architecture/stamps/single_stamp.npz`. Run it in the
`shearnet-plots` environment, with ShearNet installed as above.

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

`architecture/stamps/single_stamp.npz` is committed deliberately, so the architecture
figure can be rebuilt by anyone without ShearNet, GalSim, or cluster data access.
Re-run the notebook only when the stamp itself should change.

### `results/`

Figures driven by the evaluation FITS that `research/shear_bias/run.py` writes
(`benchmarking/evaluation.fits`, ~1 GB at production `n_obs`). One file holds
everything: `TAB_P`/`TAB_M` (per-object ± shear populations), `LEAKAGE`, `SUMMARY`,
`BINNED`, `LEAKSUM`, and the run configuration plus timing (`RENDER_S`,
`INFERENC`) in the primary header.

**`evaluation_fits.py`** is the only module that knows that layout. Everything else
asks it for named quantities, so a schema change is a one-file edit.

**`psf_leakage.py`** — mean recovered shear against PSF ellipticity, with the fitted
leakage slope α per estimator (paper Figure 6).

```bash
python psf_leakage.py --fits .../evaluation.fits
python psf_leakage.py --fits .../evaluation.fits --estimators shearnet ngmix
```

The binning, the jackknife α/β fit and the drawing are `superbit_lensing`'s
(`PSFLeakagePanelMaker`, `save_all_panels_to_fits`, `plot_psf_leakage_comparison`).
This script only selects columns. It reads `e_<est>_raw_ring`: the ring average
cancels intrinsic ellipticity, and *raw* keeps the PSF-response correction out,
since that correction is exactly what the slope is measuring.

**`snr_size_dependence.py`** — multiplicative bias in bins of S/N and half-light
radius, against the Stage IV band (paper Figure 8).

```bash
python snr_size_dependence.py --fits .../evaluation.fits
python snr_size_dependence.py --fits .../evaluation.fits --bin-by s2n hlr_th flux_th
```

Note the `BINNED` HDU **cannot** supply this figure: `bin_by` is fixed to `"flux"`
in `run.py` at the pinned commit, so `BINNED` only has flux bins. The per-object
`TAB_P`/`TAB_M` tables do carry `s2n` and `hlr_th`, so the bins are formed here and
the bias in each is computed by **ShearNet's own** `shearnet.methods.anacal.paired_bias`
with `bin_values`. `m` is a pair-matched ratio of means with a jackknife error and a
specific `c` convention; reimplementing it here would be a second thing that can
disagree with the `SUMMARY` table in the same file. (If `bin_by` grows S/N and size
options upstream, this script should switch to reading `BINNED` instead.)

**`make_fixture.py`** writes a few-MB FITS with the production schema and *known
injected answers* — a leakage slope per estimator and an S/N-dependent bias — so the
figure scripts can be developed and tested without the real file. A script that
recovers the injected value is reading the file correctly.

```bash
python make_fixture.py -o /tmp/eval_fixture.fits --n 40000
python psf_leakage.py --fits /tmp/eval_fixture.fits          # recovers 0.012 / 0.043
```

### `psf/`

**`psf_properties.py`** reproduces **Figure C4** of the SuperBIT weak-lensing paper
for the PSF ShearNet is trained on: a 3×3 grid of $(e_1, e_2, T)$ columns against
observed-PSFEx / ngmix-EM5-model / residual rows, over CCD coordinates.

Per the house rule, it calls `superbit_lensing.em5.compute_em5_psfex_maps` and
`superbit_lensing.em5.plot_em5_psfex_maps`. The grid sampling, adaptive-moment and
EM5 fits, colour maps, colour limits and layout are all upstream's; this script only
resolves the PSF path, caches, and saves.

```bash
cd psf

# take the PSF from a ShearNet config (reads paths.psfex_model_file)
python psf_properties.py --config ~/ShearNet/research/unit_tests/fourth/config.yaml \
                        --cache maps/fiducial.npz

# or point straight at a model
python psf_properties.py --psf /path/to/model.psf --cache maps/fiducial.npz

# replot from the cache; no refitting
python psf_properties.py --cache maps/fiducial.npz
```

Reading the PSF path from the ShearNet config keeps the figure tied to the PSF the
network actually saw, rather than a path copied by hand.

Two things worth knowing:

- **The fit is the slow part.** It fits adaptive moments *and* an EM5 mixture at every
  grid point — minutes at the default `--step 200` on a 9600×6400 detector. Always
  pass `--cache`; replots are then instant. `psf/maps/` is git-ignored, since the maps
  are deterministic given a `.psf` file.
- **The colour limits are hard-coded upstream** at $|e| \le 0.06$, $|\delta e| \le
  0.015$, $|\delta T| \le 0.005$, tuned to the SuperBIT survey PSF. A PSF with
  stronger ellipticity will saturate them and show flat blocks of colour. This script
  deliberately does not override that — if the limits need to change, change them in
  `superbit_lensing` so this figure and the SuperBIT paper stay consistent.
