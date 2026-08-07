# notebooks/

Exploratory notebooks. Suggested first notebook:

1. Load a sample `.mrc` (`afa.io.load_mrc`) and inspect pixel size + intensity.
2. Overlay the manual traces (`afa.io.load_traces` + `afa.viz.save_overlay`) to
   confirm the annotation format is read correctly.
3. Run the classical detector (`afa.segment.vesselness_probability`) and tune
   `sigmas` / `prob_threshold` for your fibril widths.
4. Compare automatic traces vs manual ground truth.

Keep notebooks out of the critical path — the reusable logic lives in `src/afa`.
