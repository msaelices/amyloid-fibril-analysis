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
