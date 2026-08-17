# Decision log

Why each alternative was rejected, with the measurement that decided it. The
source says what the code does; the tests pin that it keeps doing it. This is
the part you cannot reconstruct by reading.

Each entry ends with what would overturn it. Several are placeholders, not
destinations.

---

**1. Metrics never depend on the detector.** `morphology/metrics.py` takes a
centerline and a pixel size; nothing in it knows how the centerline was made.
Detection is permanently imperfect, the descriptors are exact arithmetic, and
coupling them would make every number only as good as the model. *Invariant, not
a trade-off.*

**2. Manual traces are snapped onto the ridge before use** (`trace/snap.py`).
They were drawn *beside* each fibril, median offset ~10 px, so rasterizing them
labels background and their curvature is that of an offset curve,
`k / (1 - d*k)`. Snapping gains 5.2x mean ridge response. *Drop it if a future
batch is traced on the fibrils (ImageJ ROIs): `snap=False`.*

**3. Snapping optimizes the whole trace, not each vertex.** A per-point argmax
jumps to whichever ridge is locally strongest, so traces cross onto neighbours.
With `jump_penalty=0.02`, 55% of traces ranged over the entire search band and
40% jumped 20 px within 20 px of arc length; at 0.3 with the two-pass anchor,
both are 0%.

**4. The anchor is estimated per trace, in two passes.** Without an anchor a
signal-free stretch slides 25 px to the edge of the band on no evidence. But
anchoring on *zero* hard enough to stop that also cancels the real offset:
single-pass at `anchor_penalty=0.01` leaves median |offset| at **0.0**;
two-pass at the same weight preserves **8.0 px** with wandering still at 0%.
*Rejected: simply raising the penalty. It trades one failure for another.*

**5. Unannotated ridge-like pixels are ignored, not called background.** Only a
few fibrils per image were traced; calling the rest background trains the model
to suppress what it should find. 21% of pixels ignored against 0.6% positive.
The cost is real — Dice 0.452 restricted vs **0.094** unrestricted, with 92% of
false positives inside the excluded region (`traps.md` §1). *Replace with
positive-unlabeled learning, issue #5. This is a placeholder.*

**6. Both Dice values are always reported**, with the excluded fraction. The
restricted one flatters over-prediction, the unrestricted one counts untraced
real fibrils as errors. A reader given one number cannot tell which way it
leans.

**7. Coverage is reported next to recall.** Recall alone cannot separate "missed
the fibril" from "found it in twenty pieces", and those need different work. The
pair that motivated it: coverage 0.77 with recall 3/20 — the detector was fine,
the tracer was not.

**8. Junction ends are paired per node, not chained greedily.** Under a greedy
walk the winner depended on which branch the loop reached first, so a fibril
could lose a crossing to whichever branch got there first. Recall 0.12 → 0.16 on
validation. *Honest note: the old greedy walk with the retuned angle and
bridging reaches the same headline number. This is correct, not decisive.*

**9. Pair weights are normalized to (0, 1], not raw cosines.** Summing raw
cosines let two links at the angle limit (0.515 each) outscore one straight
continuation (0.9998), at 5.2% of junctions. And past 90 degrees the cosine goes
negative, so every admissible matching lost to the empty one and linking
**silently switched off**.

**10. Short junction-to-junction branches collapse into one node.**
Skeletonizing an X gives two Y-junctions joined by a bridge, not one node;
`min_branch_px` deleted that bridge as noise, leaving the four fibril halves on
two nodes so crossings could never link. A synthetic X of 400 and 284 px
returned two ~400 px hybrids; it now returns 284 and 400.

**11. Gap bridging requires image evidence, and is off without it.** Geometry
alone welds two collinear fibrils: 38% of purely geometric joins at 60 px were
between distinct fibrils, and the ground truth measured that as 0%
(`traps.md` §3). The pipeline supplies the detector's map; a bare
`trace_centerlines(mask)` has bridging off.

**11b. Morphological opening of the mask was tried and rejected.** Skeletonizing
the thresholded mask yields a hairball (655-1122 branches per image, median
length 5-12 px, against a median manual fibril of 430 px), so cleaning the mask
looked like the fix. Opening does raise recall, but it takes the **mean**
absolute tortuosity error from 0.115 to 1.15 while leaving the median untouched
at 0.064 -- and tortuosity is one of only two descriptors currently usable.
Reporting the median hid that completely. It also sits on a cliff: radius 3
loses 12 fibrils and is worse than no opening at all, and the radius is in
pixels, so a batch imaged 30% smaller would land there. *Revisit only with a
radius derived from measured fibril width, and judged on the mean as well as the
median.*

**12. A self-contained U-Net, not a pretrained backbone.** A few dozen grayscale
images with nothing like ImageNet statistics, on CPU. A resnet34 encoder buys
little, needs a weights download, and dragged in a timm/torchvision chain that
conflicted with the CPU torch build. *Revisit if the dataset grows or a GPU
becomes usable; the interface does not change.*

**13. `pos_weight` stays at 10 and the LR schedule stays off.** A four-run sweep
gave a negative result. `pos_weight` behaves as a precision/recall knob but no
value won across 22 validation fibrils. The cosine schedule buys one extra
matched fibril and 0.06 coverage while **doubling the median tortuosity error**
(7.4% → 17.3%) — and tortuosity is the only descriptor currently valid without a
pixel size. *Revisit when real units arrive (#4), which changes the metric the
trade is judged on.*

**14. Three-way split; reporting defaults to test.** Selection and reporting had
been on the same 8 images: the shipped checkpoint was the best of 120
evaluations on them, 22% better than the median of late epochs. The cost is only
25 training images and 20 reporting fibrils, which is why little reaches
significance (`traps.md` §5).

**15. Run reports are tracked, weights are not.** Reports used to live beside
the weights under the ignored `models/`, so the record of what had been tried
existed on one laptop. They are small text with no patient data, each carrying
the commit that produced it.

**16. `.gitignore` denies by default.** Listing subfolders individually meant
any new one defaulted to tracked, and a derived dataset of patient micrographs
was staged for commit as a result. The `!data/**/` lines are load-bearing: git
does not descend into an excluded directory, so without them the README and
`.gitkeep` exceptions never match.
