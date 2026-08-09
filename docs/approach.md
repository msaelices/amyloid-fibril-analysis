# Approach & rationale

What the pipeline does. For why the alternatives lost see
[`decisions.md`](decisions.md); for the mistakes already made here see
[`traps.md`](traps.md), which is the one to read first.

## The two separable problems

1. **Metrics from a trace** — deterministic and exact. Given an ordered
   centerline and a pixel size, every descriptor is a closed formula. With
   manual traces this alone produces the CSVs and overlays.
2. **Finding the traces** — the hard vision problem. Low SNR, faint filaments,
   constant crossings. Where the R&D and the labour savings live.

Keeping them apart means the reporting is trustworthy independently of the
necessarily imperfect detector. Nothing in `morphology/metrics.py` knows how a
centerline was produced.

## Annotation

The manual traces were drawn *beside* each fibril so the line would not obscure
it, median offset ~10 px. They are an initialization, not ground truth:
`trace/snap.py` moves each onto the ridge by dynamic programming over per-vertex
offsets. Skipping that labels background for training and reports the curvature
of an offset curve.

Two consequences run through everything:

- **The annotation is partial** — 2 to 8 fibrils per micrograph out of many
  visible. Unlabelled is not negative, so ridge-like unannotated pixels are
  marked *ignore* rather than background. This distorts every metric restricted
  to the complement (`traps.md` §1).
- **The images are screenshots, not micrographs**, so there is no pixel size and
  lengths come out in screen pixels. `tortuosity` and `total_abs_turning` are
  dimensionless and exact regardless; ratios between patients are valid for
  every metric. Absolute nm needs the `.mrc` files (issue #4).

## Detection

A multiscale vesselness filter (`segment/classical.py`) is the zero-training
baseline and is weak here: coverage 0.14 against **0.78** for the U-Net.

The working detector is a small U-Net (`segment/torch_unet.py`, exposed through
`segment/unet.py`), trained from the snapped traces rasterized to a fibril-width
mask, augmented with flips and 90-degree rotations. Inference is tiled and
blended with a raised-cosine window (`segment/tiling.py`): a micrograph does not
fit in one forward pass, and butt-joined patches leave seams the skeletonizer
turns into false breaks.

Splits are by whole image and three-way (`segment/dataset.py`) — patches from
one micrograph overlap and share noise statistics, and since the checkpoint is
selected on validation, reporting belongs on a split nothing selects against.

## Tracing

1. Threshold the probability map and skeletonize.
2. Build a skeleton graph with `skan`.
3. **Collapse crossing bridges.** Skeletonizing an X yields two Y-junctions
   joined by a short bridge, not one node. Short junction-to-junction branches
   merge; length filtering applies only to free-ended branches, the real noise.
4. **Pair branch ends per junction** by collinearity rather than walking
   greedily, so a fibril does not lose a crossing to whichever branch was
   reached first.
5. **Bridge gaps** between fragments facing each other, *checked against the
   detector's map*: geometry alone welds collinear fibrils (`traps.md` §3).
6. Resample and smooth — **only chains actually joined**, since repeating that
   halves curvature while leaving length alone (`traps.md` §2).

## Metric definitions

Ordered points `p_i`, segment vectors `d_i = p_{i+1} - p_i`, lengths
`Δs_i = |d_i|`, tangent angles `θ_i = atan2(d_i.y, d_i.x)`.

- **length** `L = Σ Δs_i`
- **tortuosity** `L / |p_last - p_first|`
- **turning** `Δθ_i = wrap(θ_{i+1} - θ_i)`, wrapped to (-π, π]
- **curvature** `κ_i = Δθ_i / s_i`, with `s_i = (Δs_{i-1} + Δs_i)/2`
- **max / mean abs curvature** `max |κ_i|`, `mean |κ_i|`
- **total absolute turning** `Σ |Δθ_i|` (radians), and `Σ |Δθ_i| / L`

Curvature has units 1/length, so the pixel size is applied first. Smoothing is
required: raw pixel-stepped centerlines produce spurious high curvature.

## Statistics per patient

`mean`, `SD` and a 95% CI per metric, t-based by default.

**For publication:** fibrils within an image or patient are not independent, so
the naive CI understates uncertainty. `--bootstrap` resamples at the image
level; a mixed-effects model is the rigorous route.

## Validation

- **Pixel level.** Dice and IoU, reported both restricted to labelled pixels and
  over the whole image, with the excluded fraction. Neither alone is honest.
- **Fibril level.** One-to-one Hungarian matching on symmetric centerline
  distance, plus fragmentation-tolerant **coverage**. Together they separate
  "missed it" from "found it in twenty pieces".
- **Morphology.** Per-metric error on matched fibrils.
- Precision is `precision_lower_bound`: under partial annotation an unmatched
  detection is usually a real fibril nobody traced.

Render overlays and look at a handful. And note the sample — 20 test fibrils
cannot establish most differences worth arguing about (`traps.md` §5).
