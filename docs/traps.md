# Traps

Five mistakes that have already been made in this repository. Each looked
correct while being written, and four of the five were caught only by an
adversarial review, not by the test suite. They are collected here because they
are the ones a newcomer will make again.

If you read nothing else in `docs/`, read this.

---

## 1. Partial annotation blinds the validation

**Only 2 to 8 fibrils were traced per micrograph, out of many that are visible.**
Every consequence of that is counter-intuitive.

A metric restricted to the annotated regions can score perfectly while the
detector is badly wrong everywhere else, because "everywhere else" is not being
looked at. The `ignore` mask (`afa/segment/dataset.py`) is exactly such a
restriction: it excludes the ridge-like pixels nobody annotated, which is
precisely where a fibril detector's excess predictions land.

Measured: the ignore mask covers **21%** of each image and swallows **92%** of
the U-Net's false positives. Dice over the labelled pixels is 0.452; over all
pixels it is **0.094**. Both numbers are defensible and neither is "the" answer,
which is why `scripts/validate.py` prints both and the excluded fraction.

The sharpest instance: gap bridging was welding *different* fibrils together
38% of the time, and scored **0% error against the ground truth**, because with
2 to 8 annotated fibrils per image, two annotated ones never land collinear. The
test had no power to detect the defect at all.

> Before trusting any new metric, ask what fraction of the image it is looking
> at, and whether the failure you care about can even occur inside that fraction.

## 2. Idempotence: resampling and smoothing are not free to repeat

`_finish` (resample + smooth) applied twice leaves **length essentially
unchanged** while roughly **halving curvature**. That asymmetry is what makes it
dangerous: the metric you would spot-check looks fine.

Measured: with an identical set of 16 centerlines and no join actually made,
median `max_curvature` went 0.32 to 0.17, `mean_abs_curvature` 0.056 to 0.021,
`total_abs_turning` 13.1 to 6.8, while `length` moved 0.8%.

This is why `bridge_gaps` returns which chains it merged, so only those are
finished again. Any future post-processing stage must do the same.

## 3. Geometry cannot tell a gap in one fibril from the space between two

Two fibrils lying end to end on the same line satisfy **every** collinearity
test you can write: both tangents point at each other and at the connecting
segment. Two straight 400 px fibrils 59 px apart come back as a single 859 px
object.

There is no geometric fix, only tighter thresholds that also stop legitimate
joins. The only thing that distinguishes the two cases is **the image**: a real
fibril continues faintly across its own gap, two different fibrils have nothing
in between. Hence `bridge_gaps(evidence=...)`, and hence bridging being off by
default when no detector map is supplied.

> When a decision depends on whether two things are the same object, geometry is
> usually not enough. Go back to the pixels.

## 4. Numbers stop being comparable when you change what produced them

Three instances, all real:

- **Different loss function.** Changing `pos_weight` changes the objective, so
  the resulting loss values sit on different scales. Ranking a `pos_weight`
  sweep by `best_val_loss` is meaningless. `scripts/compare_runs.py` exists only
  because of this, and scores every checkpoint on metrics held fixed instead.
- **Different validation split.** `unet_long` scored 0.630 and `unet_v2` 0.862;
  the second is not worse, they were scored on different images.
- **Different split definition.** When the test split was introduced, the images
  that had been the validation set *became* the test set. `unet` and
  `unet_long` therefore had their checkpoints selected on exactly the images
  that are now reported as unseen. `scripts/validate.py` cannot warn about this,
  because to it those images are simply the test set.

> A loss value is only comparable to another produced by the same objective on
> the same data. Almost every interesting change violates one of the two.

## 5. Small samples do not support the claims they seem to

The test split holds **20 fibrils**. Recall going from 3/20 to 7/20 looks like a
large improvement and is **not statistically significant**: exact McNemar
p = 0.22, with Clopper-Pearson intervals [0.03, 0.38] and [0.15, 0.59] that
overlap heavily.

The first version of that PR presented it as a result. It is an indication.

> With 20 fibrils, almost nothing you can measure will reach significance.
> Report intervals, and say plainly when a difference is not distinguishable
> from noise.

---

## How these were found

Four of the five came from adversarial review: an independent pass whose brief
is to *refute* the work rather than confirm it, writing scripts to break specific
claims. Both such reviews found real defects, and in both cases the most serious
one was invisible to the test suite by construction.

The tests are good at catching regressions in behaviour that was once correct.
They are structurally poor at catching a metric that cannot see the failure it
is supposed to measure. Those need someone actively trying to break the claim.
