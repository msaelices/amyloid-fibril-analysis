# amyloid-fibril-analysis

Automatic tracing and morphological analysis of amyloid fibrils in cryo-EM
(`.mrc`) micrographs.

Given noisy 2D cryo-EM micrographs, the pipeline detects and traces individual
fibrils, then computes per-fibril morphology descriptors and aggregates them per
patient. A **deterministic classical baseline** runs with no training; a
**U-Net** trained from your existing manual traces does the job properly.

## What it computes

For every traced fibril (an ordered centerline resampled to physical units):

| Metric | Definition |
| --- | --- |
| `length_nm` | Arc length of the centerline. |
| `tortuosity` | Arc length / end-to-end (chord) distance. `1.0` = perfectly straight. |
| `mean_abs_curvature_per_nm` | Mean of \|κ\| along the fibril, κ = dθ/ds. |
| `max_curvature_per_nm` | Maximum \|κ\| (on the smoothed centerline). |
| `total_abs_turning_rad` | ∫\|dθ\| — total absolute turning along the fibril. |
| `total_abs_turning_per_nm` | ∫\|dθ\| / length — turning normalized by length. |
| `local_dir_change_per_nm` | Local direction change per unit length (mean \|Δθ\|/Δs). |

Physical units come from the pixel size in the `.mrc` header (overridable).
Curvature is computed on a smoothed, arc-length-resampled centerline, because raw
pixel discretization makes curvature noisy.

> **The annotated images are screenshots, not micrographs**, so their pixel size
> is calibrated rather than read. Matching 5 screenshots to their `.mrc`
> originals (`scripts/match_mrc.py`) gives **0.3299 nm/px**, 4x the 0.826 Å/px of
> the micrographs. The `afa` commands read `.mrc` files and take the pixel size
> from the header; the scripts that work on the screenshots need it passed, as
> `--pixel-size-nm 0.3299`, or lengths and curvature come out in screen pixels.

## Outputs

1. `per_image.csv` — one row per fibril, with image id, patient id, filament id
   and all metrics.
2. `per_patient.csv` — mean, SD and 95% confidence interval of each metric,
   grouped by patient.
3. Overlay PNGs — the traced centerlines drawn on top of the raw micrograph, for
   visual validation.

## Approach (why it's built this way)

Reading `.mrc` and computing the metrics is the easy, deterministic part. The
hard part is **finding the fibrils** in low-SNR micrographs with crossings.

- **Detection.** The classical vesselness baseline is weak here: coverage 0.14
  against **0.78** for the trained U-Net on the same 8 held-out images. Use the
  learned detector.
- **Annotation.** The manual traces were drawn *beside* each fibril, so they are
  snapped onto the ridge first (`trace/snap.py`), and `scripts/qc_traces.py`
  renders each snapped trace for checking by eye. Only a few fibrils per image
  were traced, so unlabelled pixels are treated as unknown, not background.
- **Tracing.** Skeletonize, pair branch ends per junction by collinearity,
  collapse crossing bridges, and join fragments across gaps **only where the
  detector's map supports it** (`trace/tracer.py`, `trace/bridge.py`).
- **Validation.** Dice at pixel level, matching plus coverage at fibril level,
  per-metric morphology error (`validate.py`). Expect a **semi-automatic**
  workflow for publication quality: the model proposes, you confirm.

## New to this code? Read in this order

1. **[`docs/traps.md`](docs/traps.md)** — six mistakes already made here, five
   of which the test suite could not catch. Highest value per minute here.
2. **[`docs/approach.md`](docs/approach.md)** — what the pipeline does.
3. **[`docs/decisions.md`](docs/decisions.md)** — why the alternatives lost,
   with the measurement behind each.
4. **The tests, before the source.** Each states one failure mode in a sentence:
   traces are drawn beside the fibrils, skeletonizing an X gives two junctions,
   geometry welds collinear fibrils.
5. **[`reports/README.md`](reports/README.md)** — every training run, and which
   comparisons between them are invalid.

[`CONTRIBUTING.md`](CONTRIBUTING.md) has setup, conventions, and a drill for
taking ownership of code you did not write.

## Project layout

```
src/afa/
  io/            # .mrc reading (pixel size), annotation loading (ImageJ ROI / CSV / burned-in)
  segment/       # classical vesselness baseline, U-Net, dataset, tiled inference
  trace/         # snapping, skeleton -> graph -> centerlines, gap bridging, resampling
  morphology/    # the metric definitions (deterministic, unit-tested)
  validate.py    # automatic vs manual: Dice, coverage, fibril matching, metric error
  stats.py       # per-patient means/SD/95% CI
  viz.py         # overlay rendering
  pipeline.py    # end-to-end orchestration
  cli.py         # `afa` command-line interface
configs/         # YAML pipeline configs
scripts/         # training, validation, cross-validation, trace QC, .mrc matching, imports
reports/         # tracked training run history (weights are not tracked)
tests/           # unit tests (metrics verified against analytic shapes)
data/            # gitignored -- never commit patient data
```

## Install

```bash
# uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # deterministic core + dev tools
uv pip install -e ".[dev,dl]"     # also install torch for the U-Net
```

## Quick start

```bash
# 1. Metrics + overlay from EXISTING manual traces (no detection needed)
afa metrics-from-rois data/raw/img001.mrc data/annotations/img001_RoiSet.zip \
    --patient P1 --out outputs/

# 2. Fully automatic trace with the classical baseline
afa trace data/raw/img001.mrc --patient P1 --out outputs/

# 3. Aggregate everything into per-patient stats
afa summarize outputs/per_image.csv --out outputs/per_patient.csv
```

## Status

The whole pipeline runs end to end: annotation import, trace snapping, U-Net
training and tiled inference, tracing, metrics, per-patient statistics and a
validation harness. 93 tests, CI green.

Cross-validated over all 41 annotated images (5 folds, 111 manual fibrils, every
image scored by a model that never saw it):

| | U-Net |
| --- | --- |
| One-to-one recall | **0.252** (28/111), 95% CI [0.175, 0.344] |
| Coverage of manual fibrils | 0.735 |
| Objects produced per image | 97, where 1 to 8 were traced |
| Length error on matched fibrils | 17% |
| Tortuosity error on matched fibrils | 11% |
| Curvature error on matched fibrils | 95% to 500% |

Three things to keep in mind when reading those numbers.

- **Detection works; tracing is the remaining problem.** The fibrils are found,
  but come out in pieces, and every break and spur adds a bend that is not
  there. That is why curvature is unusable while length and tortuosity are not
  (issue #7).
- **Expect about 0.25 on new data, not more.** The tracer settings were chosen
  on these same 41 images. A nested estimate that re-chooses them on four fifths
  and scores the held-out fifth gives 0.281 +/- 0.018, against 0.351 when the
  same images are used for both (`docs/traps.md` section 5,
  [`reports/README.md`](reports/README.md)). A defensible number needs new
  annotation (issue #8).
- **The classical baseline was only scored on an 8-image split**: coverage 0.14
  and 0/20 recall, against 0.78 and 7/20 for the U-Net on the same images.

The most promising next step is not in this code. RELION 5.1's amyloid tracer
detects the 4.75 Å cross-β repeat directly, which may replace the detector,
but it needs the full-resolution `.mrc` files: screenshots cannot resolve
anything finer than 6.6 Å (issue #21). See the [open issues](../../issues) for
the rest.

## License

MIT — see [LICENSE](LICENSE).
