"""Command-line interface for amyloid-fibril-analysis.

Examples
--------
    afa metrics-from-rois img.mrc img_RoiSet.zip --patient P1 --out outputs/
    afa trace img.mrc --patient P1 --out outputs/
    afa summarize outputs/per_image.csv --out outputs/per_patient.csv
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Amyloid fibril tracing & morphology.")


def _load_config(config: Path | None, pixel_size_a: float | None):
    from afa.config import Config

    cfg = Config.from_yaml(config) if config else Config()
    if pixel_size_a is not None:
        cfg.pixel_size_a = pixel_size_a
    return cfg


def _write_outputs(df, img, centerlines, out: Path, image_id: str, overlay: bool):
    out.mkdir(parents=True, exist_ok=True)
    per_image = out / "per_image.csv"
    if per_image.exists():
        import pandas as pd

        prev = pd.read_csv(per_image)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(per_image, index=False)
    typer.echo(f"Wrote {per_image} ({len(df)} rows total)")
    if overlay:
        from afa.viz import save_overlay

        labels = list(df["filament_id"].astype(str))[-len(centerlines):]
        p = save_overlay(img.data, centerlines, out / f"overlay_{image_id}.png", labels=labels)
        typer.echo(f"Wrote {p}")


@app.command("metrics-from-rois")
def metrics_from_rois(
    mrc: Path = typer.Argument(..., exists=True, help="Path to the .mrc micrograph"),
    traces: Path = typer.Argument(..., exists=True, help="ImageJ ROI(.zip/.roi) or CSV"),
    patient: str = typer.Option(..., "--patient", help="Patient id"),
    out: Path = typer.Option(Path("outputs"), "--out", help="Output directory"),
    pixel_size_a: float = typer.Option(None, "--pixel-size-a", help="Override A/pixel"),
    config: Path = typer.Option(None, "--config", exists=True),
    overlay: bool = typer.Option(True, "--overlay/--no-overlay"),
) -> None:
    """Compute metrics from EXISTING manual traces (no automatic detection)."""
    from afa.pipeline import metrics_from_traces

    cfg = _load_config(config, pixel_size_a)
    df, img, cls = metrics_from_traces(mrc, traces, patient_id=patient, config=cfg)
    _write_outputs(df, img, cls, out, Path(mrc).stem, overlay)


@app.command("trace")
def trace(
    mrc: Path = typer.Argument(..., exists=True, help="Path to the .mrc micrograph"),
    patient: str = typer.Option(..., "--patient", help="Patient id"),
    out: Path = typer.Option(Path("outputs"), "--out", help="Output directory"),
    pixel_size_a: float = typer.Option(None, "--pixel-size-a", help="Override A/pixel"),
    config: Path = typer.Option(None, "--config", exists=True),
    overlay: bool = typer.Option(True, "--overlay/--no-overlay"),
) -> None:
    """Automatically detect + trace fibrils, then compute metrics."""
    from afa.pipeline import trace_and_measure

    cfg = _load_config(config, pixel_size_a)
    df, img, cls = trace_and_measure(mrc, patient_id=patient, config=cfg)
    _write_outputs(df, img, cls, out, Path(mrc).stem, overlay)


@app.command("summarize")
def summarize(
    per_image_csv: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("outputs/per_patient.csv"), "--out"),
    bootstrap: bool = typer.Option(False, "--bootstrap", help="Image-level bootstrap CI"),
) -> None:
    """Aggregate a per_image.csv into per-patient means/SD/95% CI."""
    import pandas as pd

    from afa.stats import summarize_per_patient

    df = pd.read_csv(per_image_csv)
    summary = summarize_per_patient(df, bootstrap=bootstrap)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    typer.echo(f"Wrote {out} ({len(summary)} rows)")


if __name__ == "__main__":
    app()
