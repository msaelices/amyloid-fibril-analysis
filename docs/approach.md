# Approach & rationale

## The two separable problems

1. **Metrics from a trace** — deterministic, exact, unit-tested. Given an
   ordered centerline and a pixel size, every requested descriptor is a closed
   formula. If you already have manual traces, this alone produces the CSVs and
   overlays you need today.
2. **Finding the traces** — the hard computer-vision problem. Low SNR, faint
   filaments, and crossings make this non-trivial. This is where the R&D and the
   labor savings live.

Keeping them separate means the reporting pipeline is trustworthy and testable
independently of the (necessarily imperfect) detector.

## Detection

Cryo-EM micrographs have very low SNR. A fixed threshold on a vesselness filter
(Frangi) picks up noise and the dense dark aggregates. It is provided as a
zero-training **baseline** (`segment/classical.py`) so the pipeline runs
end-to-end from day one.

The recommended detector is a small **U-Net** (`segment/unet.py`):

- **Training data.** Rasterize the ~20-30 manual traces into binary masks (line
  drawn with a fibril-width kernel). Each ~4k×4k micrograph yields thousands of
  patches; fibrils are rotation-invariant, so flips/rotations augment heavily.
- **Split.** Hold out ~5-8 images the model never sees, to report honest
  generalization.
- **Output.** A clean fibril-probability map that the tracer consumes.

Alternatives worth benchmarking: Cellpose/Omnipose (strong on filamentous
structures), or a pretrained vessel-segmentation backbone fine-tuned.

## Tracing (probability map -> ordered centerlines)

1. Threshold the probability map and skeletonize.
2. Build a graph of the skeleton (nodes = endpoints/junctions, edges = branches).
3. **Resolve crossings by orientation continuity**: at a junction, pair the
   incoming/outgoing branches whose tangent directions are most collinear, so a
   fibril passing under another is followed through the crossing rather than
   truncated.
4. Resample each centerline to uniform arc length and smooth (spline) before
   computing curvature.

## Metric definitions

Let the smoothed centerline be points `p_i`, with segment vectors
`d_i = p_{i+1} - p_i`, segment lengths `Δs_i = |d_i|`, and tangent angles
`θ_i = atan2(d_i.y, d_i.x)`.

- **length** `L = Σ Δs_i`
- **tortuosity** `L / |p_last - p_first|`
- **local direction change** `Δθ_i = wrap(θ_{i+1} - θ_i)` (wrapped to (-π, π])
- **curvature** `κ_i = Δθ_i / Δs_i`
- **max curvature** `max |κ_i|`
- **mean abs curvature** `mean |κ_i|`
- **total absolute turning** `Σ |Δθ_i|` (radians)
- **total absolute turning per length** `Σ |Δθ_i| / L`

Curvature is scale-dependent (units 1/length), so the pixel size from the `.mrc`
header is applied before these are computed. Smoothing is required: raw
pixel-stepped centerlines produce spurious high curvature.

## Statistics per patient

`mean`, `SD`, and a 95% confidence interval per metric, grouped by patient. The
default CI is the t-based interval `mean ± t(0.975, n-1) · SE`.

**Caveat for publication.** Fibrils within one image/patient are not
independent, so the naive CI understates uncertainty. A `--bootstrap` option
resamples at the image level for a more defensible interval; a mixed-effects
model is the fully rigorous route.

## Validation

- Pixel-level Dice/IoU between predicted and manual masks.
- Per-fibril matching (Hungarian on centerline distance) then compare
  length/tortuosity distributions predicted vs manual.
- Always render overlays and eyeball a handful before trusting the numbers.
