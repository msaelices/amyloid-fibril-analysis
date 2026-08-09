# Decision log

Why the code is the way it is, and what would change it.

The source explains *what* each module does; the tests pin *that it keeps doing
it*. Neither records why the alternative was rejected, which is the part you
cannot reconstruct by reading. Each entry below states the decision, what was
rejected, **the measurement that decided it**, and what evidence would overturn
it.

Ordered roughly by how load-bearing they are.

---

## 1. The metrics never depend on the detector

**Decision.** `afa/morphology/metrics.py` takes an ordered centerline and a
pixel size and returns closed-form descriptors. Nothing in it knows how the
centerline was produced.

**Why.** Detection is a hard, permanently imperfect vision problem; the
descriptors are exact arithmetic. Coupling them would make every reported number
only as trustworthy as the detector. Kept apart, `afa metrics-from-rois`
produces publication-grade numbers from manual traces today, whatever the state
of the model.

**Would change if.** Nothing. This is the invariant the project rests on. Any
change to `metrics.py` must keep `tests/test_metrics.py` green, which pins the
values against analytic ground truth (straight line, right-angle L, circle of
radius r).

## 2. The manual traces are snapped onto the fibril before use

**Decision.** `afa/trace/snap.py` moves each drawn polyline sideways onto the
ridge before it becomes either a training label or ground truth.

**Why.** The traces were drawn deliberately *beside* each fibril so the drawn
line would not hide it. Used as-is they are wrong twice: rasterizing them labels
empty background, and an offset curve is not metrically equal to the curve it
follows — for an offset `d`, curvature becomes `k / (1 - d*k)`, and curvature is
one of the requested descriptors.

**Measured.** Median offset magnitude ~10 px across the annotation set. Snapped
traces gain 5.2x mean ridge response.

**Would change if.** A future batch is traced *on* the fibrils, e.g. with ImageJ
ROIs. Then `snap=False` and this whole stage is skipped. It exists for the
quirks of one annotation batch, not as a permanent part of the design.

## 3. Snapping optimizes the whole trace, not each point

**Decision.** Dynamic programming over per-vertex offsets, with a penalty on
offset change, rather than an independent argmax per vertex.

**Why.** A per-point argmax jumps to whichever ridge is locally strongest, so a
trace crosses onto a neighbouring fibril wherever the response is ambiguous.
The DP keeps the offset sequence coherent and rides through faint stretches.

**Measured.** With `jump_penalty=0.02`, 55% of traces ranged over more than the
entire search band and 40% jumped 20 px sideways within 20 px of arc length. At
0.3 with the two-pass anchor, both are **0%**.

## 4. The anchor is estimated per trace, in two passes

**Decision.** Pass 1 uses a tie-break weight only and estimates the offset that
trace was drawn at; pass 2 anchors on that median with the full weight.

**Why.** Some anchor is needed, or a stretch with no ridge signal has no
preferred offset and slides to the edge of the search band — 25 px on no
evidence. But anchoring on *zero* hard enough to stop that also cancels the real
~10 px offset, which is the entire thing being corrected for.

**Measured.** Single-pass at `anchor_penalty=0.01`: wandering eliminated, but
median |offset| collapses to **0.0**. Two-pass at the same weight: wandering
still 0%, median |offset| preserved at **8.0 px**.

**Rejected.** Simply raising `anchor_penalty`. It trades one failure for another.

## 5. Unannotated ridge-like pixels are *ignored*, not called background

**Decision.** `build_labels` marks them ignore and excludes them from the loss.

**Why.** Only a few fibrils per image were traced. Calling every unlabelled
pixel background trains the model to suppress exactly what it should find.

**Measured.** 21% of pixels ignored against 0.6% positive.

**Cost, and it is real.** The ignore region is where a detector's excess
predictions land, so any metric restricted to its complement flatters the model:
Dice 0.452 restricted, **0.094** unrestricted, with 92% of false positives
falling inside the excluded region. See `docs/traps.md` §1.

**Would change if.** Positive-unlabeled learning (issue #5) lands. Topaz's
GE-binomial objective is the principled version of what this fakes: it keeps the
unlabelled pixels and constrains the *expected* positive rate instead of
discarding a fifth of every image. This decision is a placeholder, not a
destination.

## 6. Both Dice values are always reported

**Decision.** `dice_labelled` and `dice_all_pixels` together, plus the excluded
fraction, in every output.

**Why.** Neither alone is honest. The restricted one flatters over-prediction;
the unrestricted one counts untraced real fibrils as errors. The truth is
between them, and a reader given only one number cannot know which way it leans.

## 7. Coverage is reported next to recall

**Decision.** `centerline_coverage` measures the fraction of a manual fibril
lying near *any* detection, regardless of how many objects that took.

**Why.** One-to-one recall alone cannot distinguish "missed the fibril" from
"found it in twenty pieces", and those call for completely different work.

**Measured.** The pairing that motivated it: coverage 0.77 with recall 3/20 —
the detector was fine, the tracer was the problem. Without coverage that reads
as a detection failure and the effort goes to the wrong place.

## 8. Junction ends are paired per node, not chained greedily

**Decision.** At each junction, pair the incident branch ends to maximize total
collinearity; solve exhaustively for the degrees that occur (3 to 6).

**Why.** A fibril crossing a dense field must win a decision at every junction it
passes. Under a greedy walk the winner depended on which branch the iteration
happened to start from, so a fibril could lose a crossing to a branch that
merely got there first.

**Measured.** Recall 0.12 to 0.16 on validation from the rewrite alone.

**Honest note.** The old greedy walk with the retuned angle and gap bridging
also reaches the same headline number on test. The rewrite is correct and
order-independent; it is not what produced the improvement.

## 9. Pair weights are normalized, not raw cosines

**Decision.** An admissible pair scores `(cos(turn) - cos(limit)) / (1 -
cos(limit))`, in (0, 1].

**Why.** Two consequences of summing raw cosines, both real. Two links at the
angle limit (0.515 each) outscored one straight continuation (0.9998), breaking
a fibril that passes cleanly through — at 5.2% of junctions under the 60 degree
default. And past 90 degrees the cosine goes negative, so every admissible
matching scored below the empty one and linking **silently switched off**.

## 10. Short junction-to-junction branches collapse into one node

**Decision.** Union-find over bridges shorter than `merge_junction_px`; length
filtering applies only to free-ended branches.

**Why.** Skeletonizing an X does not give one junction, it gives two Y-junctions
joined by a short bridge. `min_branch_px` then deleted that bridge as a "short
branch", leaving the four fibril halves on two different nodes with no two
halves of the same fibril sharing one. Crossings could never be linked, ever.

**Measured.** A synthetic X of 400 and 284 px returned two ~400 px hybrids
(half of one spliced to half of the other); it now returns 284 and 400.

## 11. Gap bridging requires image evidence and is off by default

**Decision.** `bridge_gaps` takes the detector's probability map and requires
support along the segment it is about to draw. A bare `trace_centerlines(mask)`
has bridging off; the pipeline enables it and supplies the map.

**Why.** Geometry alone welds two different fibrils that happen to be collinear.
See `docs/traps.md` §3.

**Measured.** 38% of purely geometric joins at a 60 px gap were between distinct
fibrils. The available ground truth measured that as 0%.

## 12. A self-contained U-Net, not a pretrained backbone

**Decision.** ~70 lines of torch in `afa/segment/torch_unet.py` instead of
`segmentation_models_pytorch` with a resnet34 encoder, which is what the
original handoff recommended. Augmentation is NumPy flips and 90-degree
rotations rather than albumentations.

**Why.** A few dozen grayscale images whose texture statistics are nothing like
ImageNet's, trained on CPU. A large pretrained encoder buys little, needs a
weights download at train time, and dragged in a timm/torchvision chain that
conflicted with the CPU torch build.

**Would change if.** The dataset grows substantially, or a GPU becomes usable.
The interface does not change if the backbone is swapped.

## 13. `pos_weight` stays at 10 and the LR schedule stays off

**Decision.** Both left as they were, after measuring.

**Why.** A four-run sweep produced a **negative result**, recorded rather than
buried. `pos_weight` behaves as a precision/recall knob exactly as expected but
no value was clearly better across 22 validation fibrils. The cosine schedule
buys one extra matched fibril and 0.06 coverage while **doubling the median
tortuosity error** (7.4% to 17.3%) — and tortuosity is currently the only
descriptor valid without a pixel size.

**Would change if.** Real units arrive (issue #4), which changes which metric
the trade should be judged on.

## 14. Three-way split, and reporting defaults to test

**Decision.** train / val / test. The checkpoint is selected on val; scores are
reported on test.

**Why.** Selection and reporting were happening on the same 8 images: the
shipped checkpoint was the best of 120 evaluations on them, 22% better than the
median of late epochs. Every number was optimistically biased.

**Cost.** Only 25 images left for training, and 20 fibrils for reporting, which
is why almost nothing reaches significance (`docs/traps.md` §5).

## 15. Run reports are tracked in git, weights are not

**Decision.** `reports/` holds per-run JSON plus an append-only
`training_runs.jsonl`; `models/` stays ignored.

**Why.** Reports previously lived beside the weights under `models/`, so the
only record of what had been tried lived on one laptop. They are small text with
no patient data, and each carries the commit that produced it.

## 16. `.gitignore` denies by default

**Decision.** Whole directories ignored, the few tracked placeholders
re-included explicitly, plus extension rules as a safety net.

**Why.** Listing subfolders individually meant any new one defaulted to tracked,
and a derived dataset of patient micrographs was staged for commit as a result.
A 58 MB deck of micrographs also sat untracked in the repo root, one `git add
-A` from being published.

**Note.** The `!data/**/` lines are load-bearing: git will not descend into an
excluded directory, so without them the README and `.gitkeep` exceptions can
never match.
