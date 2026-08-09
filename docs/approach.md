# Approach & rationale

What the pipeline does and why it is shaped this way. For *why the alternative
was rejected* see [`decisions.md`](decisions.md); for the mistakes already made
here see [`traps.md`](traps.md), which is the one to read first if you are new.

## The two separable problems

1. **Metrics from a trace** — deterministic, exact, unit-tested. Given an
   ordered centerline and a pixel size, every requested descriptor is a closed
   formula. If you already have manual traces, this alone produces the CSVs and
   overlays you need today.
2. **Finding the traces** — the hard computer-vision problem. Low SNR, faint
   filaments, and constant crossings. This is where the R&D and the labour
   savings live.

Keeping them separate means the reporting is trustworthy and testable
independently of the necessarily imperfect detector. Nothing in
`morphology/metrics.py` knows how a centerline was produced.

## Annotation: the traces are drawn *beside* the fibrils

The first batch of manual traces arrived as PowerPoint freeform shapes, drawn
deliberately alongside each fibril so the drawn line would not obscure it.
Median offset about 10 px.

They are therefore an *initialization*, not ground truth.
`trace/snap.py` moves each one onto the fibril ridge by dynamic programming over
per-vertex offsets before anything downstream uses it. Skipping this would label
empty background for training, and would report the curvature of an offset curve
(`k / (1 - d*k)`) rather than of the fibril.

Two further consequences run through everything:

- **The annotation is partial.** Only 2 to 8 fibrils per micrograph were traced
  out of many visible. Unlabelled is not negative, so ridge-like unannotated
  pixels are marked *ignore* and excluded from the loss rather than asserted to
  be background. This distorts every metric restricted to the complement; see
  `traps.md` §1.
- **The images are screenshots, not micrographs.** No pixel size is available,
  so lengths and curvature come out in screen pixels. `tortuosity` and
  `total_abs_turning` are dimensionless and exact today; ratios between patients
  are valid for every metric. Absolute values in nm need the original `.mrc`
  files (issue #4).

## Detection

A multiscale vesselness filter (Frangi, `segment/classical.py`) is the
zero-training baseline, so the pipeline runs end to end without a model. It is
weak on this data: coverage 0.14 of the manual fibril length against 0.78 for
the learned detector.

The working detector is a small U-Net (`segment/torch_unet.py`, exposed through
`segment/unet.py`), trained from the snapped traces rasterized to a fibril-width
mask, with flips and 90-degree rotations for augmentation. Inference is tiled and
blended with a raised-cosine window (`segment/tiling.py`), because a micrograph
does not fit in one forward pass and butt-joined patches leave seams the
skeletonizer turns into false fibril breaks.

Training, splitting and the ignore mask live in `segment/dataset.py`. Splits are
by whole image and three-way: patches from one micrograph overlap and share
noise statistics, and the checkpoint is selected on validation, so reporting
belongs on a test split nothing selects against.

## Tracing (probability map → ordered centerlines)

1. Threshold the probability map and skeletonize.
2. Build a graph of the skeleton with `skan`.
3. **Collapse crossing bridges.** Skeletonizing an X yields two Y-junctions
   joined by a short bridge, not one node. Short junction-to-junction branches
   are merged into a single node; length filtering applies only to free-ended
   branches, which are the actual noise.
4. **Pair branch ends per junction** to maximize total collinearity, rather than
   walking greedily, so a fibril crossing a dense field does not lose a junction
   to whichever branch was reached first.
5. **Bridge gaps** between fragments whose ends face each other, *checked against
   the detector's probability map*. Geometry alone cannot distinguish a break
   inside one fibril from the space between two collinear ones, and welds the
   latter (`traps.md` §3).
6. Resample to uniform arc length and smooth — but **only chains that were
   actually joined**, since repeating that step halves curvature while leaving
   length alone (`traps.md` §2).

## Metric definitions

Ordered points `p_i`, segment vectors `d_i = p_{i+1} - p_i`, segment lengths
`Δs_i = |d_i|`, tangent angles `θ_i = atan2(d_i.y, d_i.x)`.

- **length** `L = Σ Δs_i`
- **tortuosity** `L / |p_last - p_first|`
- **local direction change** `Δθ_i = wrap(θ_{i+1} - θ_i)`, wrapped to (-π, π]
- **curvature** `κ_i = Δθ_i / s_i`, with `s_i = (Δs_{i-1} + Δs_i)/2`
- **max curvature** `max |κ_i|`
- **mean abs curvature** `mean |κ_i|`
- **total absolute turning** `Σ |Δθ_i|` (radians)
- **total absolute turning per length** `Σ |Δθ_i| / L`

Curvature has units 1/length, so the pixel size is applied before these are
computed. Smoothing is required: raw pixel-stepped centerlines produce spurious
high curvature.

## Statistics per patient

`mean`, `SD` and a 95% confidence interval per metric, grouped by patient. The
default is the t-based interval.

**Caveat for publication.** Fibrils within one image or patient are not
independent, so the naive CI understates uncertainty. `--bootstrap` resamples at
the image level; a mixed-effects model is the fully rigorous route.

## Validation

- **Pixel level.** Dice and IoU, reported *both* restricted to labelled pixels
  and over the whole image, with the excluded fraction. Neither alone is honest.
- **Fibril level.** One-to-one matching by Hungarian assignment on symmetric
  centerline distance, plus **coverage**, which is fragmentation-tolerant.
  Together they separate "missed the fibril" from "found it in twenty pieces".
- **Morphology.** Per-metric error on matched fibrils.
- Precision is reported as `precision_lower_bound`, because under partial
  annotation an unmatched detection is usually a real fibril nobody traced.

Always render overlays and look at a handful. And note the sample: 20 test
fibrils cannot establish most of the differences worth arguing about
(`traps.md` §5).
