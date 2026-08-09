# Contributing

For someone new, and for someone who owns this code but did not write all of it.
Same path either way.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # deterministic core + test tooling
```

Enough for everything except training. The learned detector needs torch, kept
behind an optional extra so the core installs light:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the CPU index; plain `torch` pulls a CUDA build that conflicts with
everything else resolved from PyPI.

```bash
pytest                                  # 78 tests
ruff check src tests scripts
python scripts/make_synthetic_demo.py --out outputs_demo
```

The demo builds a synthetic micrograph with known fibrils and runs the pipeline
over it. If it writes `overlay_demo.png`, `per_image.csv` and `per_patient.csv`,
the install works.

## Read before you write

Under an hour, in this order:

1. **[`docs/traps.md`](docs/traps.md)** — five mistakes already made here, four
   of which the test suite could not catch.
2. **[`docs/approach.md`](docs/approach.md)** — what the pipeline does.
3. **[`docs/decisions.md`](docs/decisions.md)** — why the alternatives lost.
4. **The tests, before the source.** `tests/test_snap.py`,
   `test_tracer_linking.py` and `test_validate.py` state one domain fact per
   test, in a sentence.

## Taking ownership: reproduce a number

Reading does not give you ownership. Reproducing and breaking claims does. Pick
one published number first:

```bash
python scripts/validate.py --data data/dataset --weights models/unet_v2.pt \
    --detector unet --split test --out /tmp/check
```

From the output alone:

- Why are **two** Dice values printed, and why is neither "the" answer?
- Coverage 0.78 with one-to-one recall 7/20 — what does the pair tell you that
  neither number alone does?
- Is 7/20 better than the 3/20 the previous version scored? (`traps.md` §5. The
  answer is not "yes".)

Answer those and you understand the validation. Then change a tracer parameter
and **predict the direction of the effect before measuring**; that is owning the
tracer.

## Making a change

**Measure before you fix.** Twice the obvious cause was the wrong one: raising
the detection threshold made coverage worse, and opening the junction angle
contributed nothing. Write the diagnostic first.

**Tune on validation, report on test.** The checkpoint is already selected on
val, so numbers there are optimistic. `--split test` is the honest one and warns
if pointed at val.

**A claim needs a measurement.** Put the number and its uncertainty in the PR,
and say plainly when a difference is not distinguishable from noise. With 20
test fibrils, most are not.

**Add a test that would have failed before.** Required for deterministic logic.
For a bug fix, encode the failure rather than the fix — see
`test_bridging_does_not_resmooth_untouched_centerlines`, which pins a property.

## Committing

- One topic per commit; do not mix a refactor with a behaviour change.
- Explain *why*, with the measurement. `git log` is the densest record of
  reasoning here — read a few before writing your first.
- Do not amend; add a commit on top.
- Branch and open a PR. Never commit to `main`.
- `pytest` green and `ruff check src tests scripts` clean, every time.

## Never commit patient data

`.gitignore` denies by default: `data/`, `models/` and `outputs/` wholesale,
plus extension rules for micrographs, decks, spreadsheets and weights. This has
already caught a real near-miss — a 58 MB deck of micrographs in the repo root,
one `git add -A` from being published.

```bash
git add -A --dry-run    # nothing under data/, models/, outputs/ may appear
```

To commit something the rules catch, such as a documentation figure, be explicit:
`git add -f docs/img/whatever.png`.

## Training runs

Weights are not tracked; **the record is**. `scripts/train_unet.py` writes a
report to `reports/` plus an append-only line in `training_runs.jsonl` with the
losses, the split, every hyperparameter and the commit. Commit it: a run whose
record lives only on your laptop cannot be compared against later.

Losses from **different splits** or **different loss functions** are not
comparable (`traps.md` §4); use `scripts/compare_runs.py`, which scores on fixed
downstream metrics.

Label building costs tens of seconds per image and is cached under
`data/dataset/cache/`, keyed on the labelling parameters *and* the trace file
contents. If a run seems to ignore your changes, check the key before suspecting
the cache.

## Review

**Every PR gets an adversarial review**: an independent pass whose brief is to
*refute* the work, writing scripts to break specific claims rather than reading
for style. It has run twice and found real defects both times, including one the
test suite scored as 0% error while it was wrong 38% of the time.

Attack the claims, not the formatting. Ask what the number would look like if
the author's assumption were wrong.

## Your first contribution

- **#12**, the fibril numbering — small, self-contained, needs judgement rather
  than machinery.
- **#9**, threshold selection — takes you through detector, tracer and
  validation in one go, and ends in a measurement.
- **#11**, the content-independent ignore threshold — a confirmed defect with a
  known reproduction, so the test writes itself first.

Avoid **#5** (positive-unlabeled learning) as a first task: the most valuable
item on the list, and the least settled part of the design.
