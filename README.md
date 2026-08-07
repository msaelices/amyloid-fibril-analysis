# amyloid-fibril-analysis

Automatic tracing and morphological analysis of amyloid fibrils in cryo-EM
(`.mrc`) micrographs.

Given noisy 2D cryo-EM micrographs, the pipeline detects and traces individual
fibrils, then computes per-fibril morphology descriptors and aggregates them per
patient. It ships with a **deterministic classical baseline** that runs with no
training, and a clean interface for a **learned (U-Net) segmenter** trained from
your existing manual traces.

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

All physical units come from the pixel size stored in the `.mrc` header
(overridable). Curvature is computed on a smoothed, arc-length-resampled
centerline because raw pixel discretization makes curvature noisy.

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

- **Detection.** A classical vesselness/ridge filter (`segment/classical.py`)
  gives a zero-training baseline. Because you already have ~20-30 manually
  traced images, the recommended path is a small **U-Net** trained on masks
  rasterized from those traces (`segment/unet.py`), which learns to ignore
  background noise far better than any fixed threshold.
- **Tracing.** The probability map is skeletonized and turned into a graph;
  junctions (fibril crossings) are resolved by **orientation continuity** — at a
  4-way crossing, opposite branches with the smoothest direction are linked
  (`trace/tracer.py`).
- **Validation.** Automatic traces are matched against your manual ground truth
  to report length/tortuosity error and pixel-level overlap. Expect a
  **semi-automatic** workflow for publication quality: the model proposes, you
  confirm/split at ambiguous crossings.

See [`docs/approach.md`](docs/approach.md) for the full rationale.

## Project layout

```
src/afa/
  io/            # .mrc reading (pixel size), annotation loading (ImageJ ROI / CSV / burned-in)
  segment/       # classical vesselness baseline + U-Net interface
  trace/         # skeleton -> graph -> orientation-aware centerlines, resampling & smoothing
  morphology/    # the metric definitions (deterministic, unit-tested)
  stats.py       # per-patient means/SD/95% CI
  viz.py         # overlay rendering
  pipeline.py    # end-to-end orchestration
  cli.py         # `afa` command-line interface
configs/         # YAML pipeline configs
tests/           # unit tests (metrics verified against analytic shapes)
data/            # raw/, annotations/, processed/ (gitignored -- never commit patient data)
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

Deterministic core (io, metrics, stats, viz) and the classical tracing baseline
are implemented and tested. The U-Net trainer is a documented interface ready to
be filled in once a sample `.mrc` + its traces are available. This is an early
scaffold — see the issues / `docs/approach.md` for the roadmap.

## License

MIT — see [LICENSE](LICENSE).
