# Traps

Six mistakes already made here. Each looked correct while being written, and
four were caught by adversarial review rather than by the test suite. Read this
first.

---

## 1. Partial annotation blinds the validation

Only 2 to 8 fibrils were traced per micrograph, out of many visible. A metric
restricted to the annotated regions can score perfectly while the detector is
wrong everywhere else, because everywhere else is not being looked at.

The `ignore` mask is exactly such a restriction, and it excludes the ridge-like
pixels — precisely where a detector's excess predictions land. It covers **21%**
of each image and swallows **92%** of the U-Net's false positives: Dice is 0.452
over labelled pixels and **0.094** over all of them. Both are defensible, which
is why `scripts/validate.py` prints both plus the excluded fraction.

The sharpest case: gap bridging welded *different* fibrils 38% of the time and
scored **0% error against the ground truth**, because with 2 to 8 annotated
fibrils per image two annotated ones never land collinear. The test had no power
to detect it.

> Before trusting a new metric, ask what fraction of the image it looks at, and
> whether the failure you care about can occur inside that fraction.

## 2. Resampling and smoothing are not free to repeat

`_finish` applied twice leaves **length essentially unchanged** while roughly
**halving curvature**. That asymmetry is what makes it dangerous: the metric you
would spot-check looks fine.

With an identical set of 16 centerlines and no join actually made, median
`max_curvature` went 0.32 → 0.17, `mean_abs_curvature` 0.056 → 0.021,
`total_abs_turning` 13.1 → 6.8, and `length` moved 0.8%.

Hence `bridge_gaps` reporting which chains it merged, so only those are finished
again. Any future post-processing stage must do the same.

## 3. Geometry cannot tell a gap in one fibril from the space between two

Two fibrils lying end to end on one line satisfy *every* collinearity test:
both tangents point at each other and at the connecting segment. Two straight
400 px fibrils 59 px apart come back as a single 859 px object.

There is no geometric fix, only tighter thresholds that also block legitimate
joins. Only the image separates the cases — a real fibril continues faintly
across its own gap, two different ones have nothing between them. Hence
`bridge_gaps(evidence=...)`.

> When a decision depends on whether two things are the same object, go back to
> the pixels.

## 4. Numbers stop being comparable when you change what produced them

- **Different loss function.** Changing `pos_weight` changes the objective, so
  the losses sit on different scales. `scripts/compare_runs.py` exists only
  because of this, and scores checkpoints on fixed downstream metrics instead.
- **Different validation split.** `unet_long` scored 0.630 and `unet_v2` 0.862;
  the second is not worse, they were scored on different images.
- **Different split definition.** When the test split was introduced, the old
  validation images *became* the test set, so `unet` and `unet_long` had their
  checkpoints selected on exactly the images now reported as unseen.
  `scripts/validate.py` cannot warn about this — to it, those are just the test
  set.

## 5. Tuning on the evaluation set invents its own significance

Sweeping ~50 tracer configurations against the same 111 fibrils and then
reporting the best cell's p-value treats a search as if it had been a single
comparison. Measured on the real search:

| | p |
| --- | --- |
| raw, as first reported | 0.0074 |
| Bonferroni over the 12 documented mask variants | 0.089 |
| Bonferroni over the full 51-variant family | 0.377 |
| Westfall-Young max-T (multiplicity *and* clustering) | 0.306 |

A family of **seven** is enough to destroy a p of 0.0074. Twelve of the 51
variants reached raw p < 0.05, which is what selection looks like, not twelve
real effects.

Nested selection put the honest gain at **+0.03**, against +0.10 as reported.

> Count the configurations you tried before you quote a p-value. If you cannot
> reconstruct the count, you cannot quote the p-value.

**And note what cross-validation does not fix.** The k-fold in `cv5_*` makes the
*detector* out-of-fold. It does nothing for the *tracer*, whose parameters were
chosen on the pooled out-of-fold predictions of all 41 images. There is at
present **no held-out estimate anywhere in this project**: the detection
threshold, match distance, coverage tolerance, `width_px` and both tracer
parameters were all fixed by looking at these same images.

## 6. Small samples do not support the claims they seem to

The test split holds **20 fibrils**. Recall going 3/20 → 7/20 looks large and is
**not significant**: exact McNemar p = 0.22, Clopper-Pearson intervals
[0.03, 0.38] and [0.15, 0.59] overlapping heavily. The first version of that PR
presented it as a result. It is an indication.

---

Tests catch regressions in behaviour that was once correct. They are
structurally blind to a metric that cannot see the failure it is meant to
measure — that needs someone actively trying to refute the claim.
