# reports/

Training run records. **Tracked in git on purpose**, unlike `models/`.

Weights are large binaries and stay out of the repository. The record of *what
was tried and how it went* is small text and belongs in it: without that, the
only evidence a run ever happened lives on one laptop, and there is no way to
compare a new result against an old one.

Nothing here contains patient data. The ids (`slide01`, ...) are the derived
dataset's filenames, not clinical identifiers.

## Files

- `<run>.json` — one per run: per-epoch train and validation loss, the
  train/val/test split, every hyperparameter, and the commit the run was made
  from. Overwritten if a run of the same name is repeated.
- `training_runs.jsonl` — append-only, one summary line per run. This is the
  history: it survives re-runs and rewrites, so a run is never silently
  replaced by a later one with the same name.

## Reading a run

`best_val_loss` is the checkpoint that was kept; training does not stop early,
it runs every epoch and keeps the best. A large gap between `final_train_loss`
and `final_val_loss` means the later epochs overfit and were discarded.

**Validation losses are not comparable across runs whose `val_ids` differ.**
The split changed when the test split was introduced, so `unet_long` (0.630) and
`unet_v2` (0.862) were scored on different images and the second is not worse
for being higher. Compare the shape of a curve across runs, and the value only
within a fixed split.

## Runs so far

| run | commit | epochs | train/val/test | best val | note |
| --- | --- | --- | --- | --- | --- |
| `unet` | `dbcbb64` | 30 | 33/8/0 | 0.708 | stopped while still improving |
| `unet_long` | `dbcbb64` | 120 | 33/8/0 | 0.630 @ 44 | overfits after ~45 |
| `unet_v2` | `25bea36` | 120 | 25/8/8 | 0.862 @ 49 | honest 3-way split, fewer training images |

The first three were recovered from `models/` after the fact and carry
`"backfilled"`; their timestamps come from file mtime and the commits were
identified from the branch history, so treat them as approximate.

> **`unet` and `unet_long` must never be evaluated on the test split.** Both
> predate the three-way split, and their 8 validation images are, image for
> image, the 8 that are now the test set: `slide03, 04, 05, 22, 25, 27, 28, 35`.
> Their checkpoints were selected on exactly those images, so any score they
> get there is meaningless. `scripts/validate.py` cannot warn about this — it
> only knows the split it computes itself, and to it those images are the test
> set. `unet_v2` and the four sweep runs are clean.

## The pos_weight / schedule sweep

Four 50-epoch runs, scored by `scripts/compare_runs.py` on the validation split.
Ranking them by `best_val_loss` would have been meaningless: changing
`pos_weight` changes the objective, so those values sit on different scales.

| run | pos_weight | schedule | Dice | coverage | recall |
| --- | --- | --- | --- | --- | --- |
| `sweep_pw10` | 10 | cosine | 0.404 | 0.79 | 0.32 |
| `sweep_pw30` | 30 | cosine | 0.341 | 0.77 | 0.18 |
| `sweep_pw10_nosched` | 10 | none | 0.459 | 0.75 | 0.18 |
| `sweep_pw3` | 3 | cosine | 0.469 | 0.69 | 0.18 |

**Outcome: no change to the defaults was justified.**

`pos_weight` behaves exactly as a precision/recall knob — lower gives a cleaner
mask and misses more fibrils, higher the reverse — but no value was clearly
better, and with 20 fibrils in the split the spread is within noise. The 10.0
that had been sitting there unjustified is fine.

The cosine schedule was added expecting a win and did not get one. On the test
split, against the identical configuration without it:

| | no schedule | cosine |
| --- | --- | --- |
| one-to-one recall | 7/20 | 8/20 |
| coverage | 0.78 | 0.84 |
| length error | 19% | 24% |
| **tortuosity error, median** | **7.4%** | **17.3%** |
| tortuosity error, mean | 6.9% | 47.0% |

It buys one extra fibril and 0.06 coverage, and costs a doubled median
tortuosity error plus two traces that wander between fibrils (predicted
tortuosity 2.5 against a true 1.0). Tortuosity is the only descriptor valid
without a pixel size, so that trade is bad here. The schedule stays available
behind `--cosine-schedule` and is off by default.

## The cross-validated baseline (`cv5_*`)

Five folds over all 41 images, so every image is scored by a model that never saw
it and the pooled sample is **111 fibrils instead of 20**.

| | pooled |
| --- | --- |
| one-to-one recall | **0.252** (28/111) |
| coverage | 0.735 |
| Dice, labelled / all px | 0.443 / 0.094 |
| objects per image | 97 |
| recall per fold | 0.33, 0.29, 0.24, 0.15, 0.21 (sd 0.063) |

**This supersedes the single-split figure, and is lower than it.** That split
reported 7/20 = 0.35, which sits outside both the pooled 95% interval and the
per-fold range: it was a favourable draw, not a result. Treat 0.25 as the
baseline.

The point of the exercise is the interval, not the point estimate:

| | recall | 95% CI | width |
| --- | --- | --- | --- |
| single split | 0.350 | [0.154, 0.592] | 0.438 |
| 5 folds | 0.252 | [0.175, 0.344] | **0.169** |

2.6x tighter. A paired comparison at n=20 needed roughly a 20-point swing to
reach significance; at n=111 a few points suffice. Experiments on this
repository are measurable now, which they were not before.

Morphology on the 28 matched fibrils, in nm (0.3299 nm/px, calibrated from the
matched `.mrc` files):

| metric | manual mean | mean abs error | relative |
| --- | --- | --- | --- |
| length | 112.4 nm | 18.6 nm | 17% |
| tortuosity | 1.023 | 0.115 | 11% |
| max curvature | 1.19 /nm | 0.76 | 95% |
| mean abs curvature | 0.046 /nm | 0.149 | 500% |

Length and tortuosity are usable. **Every curvature column is not**, and that is
fragmentation (issue #7), not units.

## Reproducibility

`sweep_pw10_nosched` reproduced `unet_v2` exactly: same best validation loss at
the same epoch, and the saved tensors are bit-identical. (The checkpoint files
have different md5s, but only because the container format carries metadata; the
weights match to 0.0.) Same seed and same configuration give the same model.
