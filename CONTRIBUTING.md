# Contributing

This document is for two readers: someone new who wants to contribute, and
someone who owns this code but did not write all of it and wants to take control
of it. The path is the same either way.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"              # deterministic core + test tooling
```

That is enough for everything except training. The learned detector needs torch,
kept behind an optional extra so the deterministic pipeline installs light:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the CPU index. Installing plain `torch` pulls a CUDA build that then
conflicts with anything else resolved from PyPI.

Check it worked:

```bash
pytest                                  # 78 tests, all green
ruff check src tests scripts            # clean
python scripts/make_synthetic_demo.py --out outputs_demo
```

The demo generates a synthetic micrograph with known fibrils and runs the whole
pipeline over it. If that produces `overlay_demo.png`, `per_image.csv` and
`per_patient.csv`, you have a working install.

## Read before you write

In this order, and it will take under an hour:

1. **[`docs/traps.md`](docs/traps.md)**. Five mistakes already made here. Four
   of them the test suite could not catch. You will otherwise make them again.
2. **[`docs/approach.md`](docs/approach.md)**. What the pipeline does.
3. **[`docs/decisions.md`](docs/decisions.md)**. Why the alternatives were
   rejected, with the measurement behind each.
4. **The tests, before the source.** `tests/test_snap.py`,
   `tests/test_tracer_linking.py` and `tests/test_validate.py` each state a
   domain fact per test, in a sentence. They teach faster than the modules.

## Taking ownership: reproduce a number

Reading does not give you ownership of code. Being able to reproduce and break
its claims does. Before changing anything, pick one published number and
regenerate it:

```bash
python scripts/validate.py --data data/dataset --weights models/unet_v2.pt \
    --detector unet --split test --out /tmp/check
```

Now answer, from the output alone:

- Why are **two** Dice values printed, and why is neither of them "the" answer?
- Coverage is 0.78 and one-to-one recall is 7/20. What does that pair tell you
  that either number alone does not?
- Is 7/20 better than the 3/20 the previous version scored? (The honest answer
  is in `docs/traps.md` §5, and it is not "yes".)

If you can answer those three, you understand the validation. If you can then
change a tracer parameter and **predict the direction of the effect before
measuring it**, you own the tracer.

## Making a change

**Measure before you fix.** Twice now the obvious cause has been the wrong one.
Raising the detection threshold looked like the fix for a noisy mask and made
coverage worse. Opening the junction angle looked like the fix for fragmented
fibrils and contributed nothing. Write the diagnostic first.

**Tune on validation, report on test.** `split_images` gives three splits. The
checkpoint is already selected on val, so any number measured there is
optimistic. `scripts/validate.py --split test` is the honest one, and it warns
if you point it at val.

**A claim needs a measurement.** "This should be better" is not a reason to
merge. Put the number and its uncertainty in the PR description, and say
plainly when a difference is not distinguishable from noise. The test split
holds 20 fibrils; most differences are not.

**Add a test that would have failed before.** For deterministic logic that is
required, not optional. If the change is a bug fix, the test should encode the
failure, not the fix — see `test_bridging_does_not_resmooth_untouched_centerlines`,
which pins a property rather than an implementation.

## Committing

- **One topic per commit.** Do not mix a refactor with a behaviour change. This
  is enforced by review, not by tooling.
- **Explain *why* in the message**, and include the measurement. `git log` is
  the densest record of reasoning in this repository — read a few before writing
  your first.
- **Do not amend.** Add a new commit on top.
- **Branch, then PR.** Never commit to `main` directly.
- `pytest` green and `ruff check src tests scripts` clean before every commit.

## Never commit patient data

`.gitignore` denies by default: `data/`, `models/` and `outputs/` are ignored
wholesale, plus extension rules for micrographs, decks, spreadsheets and
weights. This is deliberate and has already caught a real near-miss — a 58 MB
deck of micrographs sitting in the repo root, one `git add -A` from being
published.

Before any commit that touches paths you are unsure of:

```bash
git add -A --dry-run       # nothing under data/, models/, outputs/ may appear
```

To commit something the rules catch, such as a documentation figure, be explicit
about it: `git add -f docs/img/whatever.png`.

## Training runs

Weights are not tracked; **the record of the run is**. `scripts/train_unet.py`
writes a per-run report to `reports/` plus an append-only line in
`reports/training_runs.jsonl`, carrying the losses, the split, every
hyperparameter and the commit that produced them.

Commit that report. A run whose record lives only on your laptop may as well not
have happened, and comparing against it later is impossible.

Two things to know before comparing runs:

- Validation losses from **different splits** are not comparable.
- Validation losses from **different loss functions** are not comparable either,
  which is why `scripts/compare_runs.py` exists and scores checkpoints on fixed
  downstream metrics instead.

Label building runs a ridge filter and a dynamic program per trace, tens of
seconds per image. It is cached under `data/dataset/cache/`, keyed on the
labelling parameters *and* the content of the trace files, so re-importing
annotations invalidates it correctly. If a run seems to ignore your changes,
check the key before suspecting the cache.

## Review

**Every PR gets an adversarial review**: an independent pass whose brief is to
*refute* the work, writing scripts to break specific claims rather than reading
for style. It has run twice and found real defects both times, including one the
test suite scored as 0% error while it was wrong 38% of the time.

This is not ceremony. The test suite is good at catching regressions in
behaviour that was once correct, and structurally blind to a metric that cannot
see the failure it is supposed to measure. Only someone actively trying to break
the claim finds those.

When you review, attack the claims, not the formatting. Ask what the number
would look like if the author's assumption were wrong.

## Your first contribution

The open issues are labelled. Good places to start, in rough order of how much
of the codebase they force you to touch:

- **#12**, resolving the fibril numbering — small, self-contained, needs
  judgement rather than machinery.
- **#9**, choosing the detection threshold from a curve instead of the hardcoded
  0.5 — takes you through the detector, the tracer and the validation harness in
  one go, and ends in a measurement.
- **#11**, the content-independent ignore threshold — a confirmed defect with a
  known reproduction, so the test is easy to write first.

Avoid **#5** (positive-unlabeled learning) as a first task. It is the most
valuable thing on the list and it touches the least settled part of the design.
