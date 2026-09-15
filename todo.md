# ShearNet paper — writing TODO

Numbers are frozen. Everything below is transcription and prose; nothing here
requires another measurement run.

Line numbers refer to `main.tex` as of the draft in `ShearNet_Paper.zip`.

**Convention for every number in the paper**

| estimator | reported as | meaning |
|---|---|---|
| ShearNet | `rgamma` | raw image, metacal never touching it, divided by its shear response |
| ngmix | `metacal` | the full estimator: `R^PSF` subtracted, then divided by `R^gamma` |

`m`, `c` and `alpha` all follow that one choice per estimator
(`paper_numbers.REPORTED_CORRECTION`), so no row mixes pipelines.

---

## 0. One command, then stop

- [ ] `cd results && python paper_tables.py --runs ../evaluations`
      — fills the `m1` column. **This is the last number.**

      ShearNet's `m1` is *always* recomputed from the per-object columns, never
      read from SUMMARY: the paper reports ShearNet under `rgamma`, and the
      benchmark writes SUMMARY rows only for `metacal`/`anacal`/`sim`/`none`.
      Until that fallback existed, every ShearNet cell printed `\pending`. The
      output now says which source answered, per estimator:

      `% m1 from: ngmix -> summary, shearnet -> recomputed`

      If ShearNet ever reads `summary` there, something has changed the
      reported correction and the number is no longer the one §1 describes.
- [ ] `./run_all.sh --go --fits evaluations/fourth.fits --runs evaluations`
      — regenerates every figure with the frozen settings.
- [ ] Optional, for the paper's fonts rather than mathtext:
      `~/.TinyTeX/bin/*/tlmgr install cm-super type1cm dvipng`

---

## 1. Tables

### `tab:response-diag` (L878–884)

- [ ] **Change the column header.** It currently reads
      `Objective disabled | Fiducial model | Target`. The first column was the
      λ_PSF ablation arm, which is cut. `response_diagnostics.py` emits
      `ShearNet | ngmix`, so the header must be
      `\colhead{Quantity} & \colhead{\textsc{ShearNet}-D4} & \colhead{\textsc{ngmix}} & \colhead{Target}`.
- [ ] **Delete rows 4 and 7** — `\langle D_{11}-D_{22}\rangle` and the
      translation response. Both are training-time diagnostics with no column
      in the evaluation FITS, so neither can ever be filled from a run.
- [ ] Paste (UT4, uncut, N_jack = 20):

```
R^gamma_11 - 1     -0.0927 +/- 0.0006    -0.3589 +/- 0.0026     0
R^gamma_22 - 1     -0.0973 +/- 0.0006    -0.3428 +/- 0.0022     0
R^gamma_(12,21)   (+0.0110, +0.0126)    (-0.0010, +0.0103)    (0,0)
R^PSF_11           -0.3749 +/- 0.0006    +0.2814 +/- 0.0025     0
R^PSF_22           -0.2772 +/- 0.0005    +0.2685 +/- 0.0023     0
```

- [ ] **Narrow the caption.** It currently promises the translation response
      and the isotropy combinations. Both rows are gone; say the table reports
      `R^gamma` against the analytic target and `R^PSF` against zero, and that
      `R^PSF` is *reported, not applied* (see §4).

### `tab:unit-test-bias` (L936–939)

- [ ] Paste the `m1` column from step 0. The `c2` column is already measured:

```
        ShearNet c2 (1e-5)     ngmix c2 (1e-5)
UT1       -43.53 +/- 1.56       +24.78 +/-  2.06
UT2        +0.76 +/- 2.15      +907.69 +/-  5.19
UT3        -2.33 +/- 5.89     +1180.51 +/- 10.18
UT4        -4.86 +/- 8.29     +1176.61 +/- 13.20
```

- [ ] Add a footnote that the four rungs are measured on 2x10^5 objects each
      with no selection cut (see §4).

### `tab:timing` (L1119–1122)

- [ ] **DEFERRED** — waiting on `evaluation_timed.fits`. ShearNet's half is
      known (22.82 s for 2x10^5 objects on one GPU, 8764 gal s^-1); the ngmix
      row and the ratio come from that run. Leave `\pending` until then.

---

## 2. Figures

All seven producers exist and run. After step 0, replace the
`\figureplaceholder` boxes:

- [ ] L405 — `fig:psf-properties` (run `psf/psf_properties.py` separately; it
      reads the PSFEx model, not a run)
- [ ] L906 — `fig:response_snr` (now drawn with `plots_from_fits.ipynb`'s own
      binning and axes — percentile bins, median with a +/-1 SE band, the 1/2/5
      decade ticks — rather than a second version of the same plot)
- [ ] L966 — `fig:prediction-residuals` (the point cloud is now rasterized, so
      the PDF is ~0.8 MB instead of 6.2 MB and actually opens; the axes, the fit
      curve and all text are still vector. The numbers in it never changed.)
- [ ] L1013 — `fig:psf_leakage`
- [ ] L1043 — `fig:snr_size`
- [ ] L1081 — second `fig:response_snr` placeholder (the draft has two; keep one)

**One cosmetic thing to decide on `fig:prediction-residuals`.** The shaded
tolerance band is `error_allowed = 0.01`, and the UT4 residuals span about
+/-1.2, so the band is a hairline at `y = 0` sitting underneath the red dashed
zero line — invisible at print size. Either raise it to something meaningful for
this figure (`--error-allowed 0.1`) or do not mention a tolerance band in the
caption. Do not describe a band the reader cannot see.

---

## 3. Prose, with the numbers to use

- [ ] **L889 — response diagnostics.** ShearNet's `R^gamma` is 0.907/0.903
      against an analytic ensemble target of 1; ngmix's is 0.641/0.657. Off
      diagonals are <= 0.013 for ShearNet and consistent with zero for both.
      **Drop the λ_iso sentence** — there is no ablation arm to support it.

- [ ] **L920 — the ladder.** ngmix's `m1` jumps at **UT3** (+16 -> +109 x10^-3),
      the rung that introduces the COSMOS size distribution. Name UT3 as the
      first source of realism that significantly changes the bias.

- [ ] **L977 — UT4 residuals.** ShearNet slope -0.1201 +/- 0.0008, ngmix
      -0.3835 +/- 0.0007. These are `R^gamma - 1` measured a completely
      different way (a per-object regression against truth rather than a
      finite difference), and they agree with `tab:response-diag` to three
      digits. Worth one sentence as an independent consistency check.

- [ ] **L1037 / L1052 — S/N and size.** ShearNet `m` spans
      [+0.017, +0.332] across S/N and [+0.025, +0.249] across size; ngmix spans
      [+0.069, +0.619] and [-0.005, +0.757]. The size panel is the evidence for
      the sample limitation in §4 — point at it explicitly.

- [ ] **L1146 — PSF leakage.** Report alpha for both, each under its own
      pipeline:

```
                        |alpha_1|            |alpha_2|
ShearNet (raw/rgamma)   ~0.0104              ~0.0121
ngmix (noshear/rgamma)  +0.0167 +/- 0.0023   +0.0751 +/- 0.0019
```

      **Lead with isotropy, not magnitude.** |alpha_1/alpha_2| is 0.87 for
      ShearNet and 0.22 for ngmix. An isotropic leakage is one coefficient to
      calibrate and averages down under field rotation; an anisotropic one does
      neither. That is the architectural claim the D_4 design supports.
      RMS leakage is ~4.8x lower for ShearNet.

- [ ] **L1130 — Discussion opening.** Three measured differences: c2
      consistent with zero vs ~1.2x10^-2; RMS leakage 4.8x lower and isotropic;
      ~8800 gal s^-1 on one GPU.

- [ ] **L1182 — Conclusions.** Same three, one sentence each.

- [ ] **L412** — representative PSFEx filename in the `fig:psf-properties`
      caption.
- [ ] **L425** — S/N range and median: 56 to 2039, median ~135.
- [ ] **L822** — training wall-clock and GPU model.

### The one exception worth a sentence

- [ ] **UT1 is the only rung where ShearNet's c2 is worse than ngmix's**
      (-43.5 +/- 1.6 vs +24.8 +/- 2.1, i.e. 27 sigma from zero). UT1 is the
      constant-radius, no-PSF-variation rung. State it rather than let a reader
      find it in the table; a one-line acknowledgement costs nothing and a
      silent anomaly costs credibility.

---

## 4. Limitations — write these, do not bury them

Three of these read as rigour when you raise them and as sloppiness when a
referee does.

- [ ] **`R^gamma` inflates the leakage, necessarily.** alpha of `e/<R>` is
      `alpha(e)/<R>`, so calibrating *raises* alpha: ngmix 0.0107 -> 0.0167,
      ShearNet 0.0095 -> 0.0104. Both estimators are quoted calibrated, so the
      comparison is like-for-like; say so explicitly so the raw numbers being
      smaller is not mistaken for a better result being hidden.

- [ ] **Metacalibration degrades ShearNet.** Its reconvolved noshear prediction
      carries 2.2x the leakage of its raw prediction (0.0095 -> 0.0209) and a
      worse `m`. ngmix is essentially unaffected (0.0078 -> 0.0107). ShearNet is
      therefore reported *without* metacal, and the deconvolve/reconvolve step
      is the reason.

- [ ] **`R^PSF` is reported but never applied.** Measured at +0.281 (ngmix) and
      -0.375 (ShearNet) against leakages of 0.011 and 0.0095 in the very shapes
      it would correct — 25x to 40x too large. Applying it drives alpha to
      -0.43 and +0.37 with the sign flipped, i.e. it injects leakage rather
      than removing it. Metacal's `*_psf` products shear the *reconvolution*
      PSF, which is a response to PSF model error rather than to the observed
      PSF's ellipticity; treating the two as the same coefficient is the
      suspected cause. State this as an open issue.

- [ ] **No selection cut is applied.** The catalog contains unresolvable
      sources: failed Sersic fits are set to `hlr = 1e-6` arcsec and kept
      (`detection_catalog_split.ipynb`, cell 1), against a 0.5 arcsec FWHM PSF.
      This is why `m` is large for **both** estimators at UT3-UT4, and why the
      size panel of `fig:snr_size` shows the trend it does. The fix — rerunning
      SExtractor on COSMOS 2015 with SuperBIT noise and PSF to get a
      detection-limited sample — is future work, not a post-hoc threshold.
      **Also update §sec:dataset**, which currently claims half-light radii
      "comparable to the PSF full-width at half-maximum".

- [ ] **The train/eval split shares galaxies.** The notebook augments each
      source galaxy into 10 orientations and *then* permutes, so
      P(all 10 copies land in eval) = 0.4^10 and **99.99% of eval galaxies have
      training copies at other orientations**. Same radius, same flux, same axis
      ratio. This is the one a referee will find first. State it plainly and say
      the rebuild (split on the source galaxy, before augmentation) is underway.

- [ ] **The +/- populations share a noise realisation.** `shear_pair` renders
      both signs from one seed, so pixel noise largely cancels in the paired
      difference. That is deliberate variance reduction, but without stating it
      a 7x10^-4 error bar on 21k objects is not believable.

---

## 5. Delete what has no data

- [ ] §Architecture and Response-Objective Ablations (L1230) — reduce to a
      sentence.
- [ ] §Secondary Ablations (L1297) — reduce to a sentence.
- [ ] §Hyperparameter Search (L1342) — reduce to a sentence; the text already
      says the hyperparameters were fixed a priori.
- [ ] `fig:robustness` (L1203) — no producer; needs its own runs. Cut or mark
      future work.
- [ ] `fig:response-training` (L1425) — needs training records, not the
      benchmark. Cut or mark future work.
- [ ] Then `python results/paper_manifest.py --strict` and confirm nothing is
      orphaned.

---

## Deferred (not blocking the draft)

- `tab:timing` — waiting on `evaluation_timed.fits`.
- The `R^PSF` discrepancy — worth understanding, but §4 states it honestly and
  the paper does not depend on it.
- Retraining on a clean split — v2 / referee response.
