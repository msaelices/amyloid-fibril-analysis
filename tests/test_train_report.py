"""Tests for the training run log.

The point of the log is that history survives. A re-run overwrites its own
detailed report, but must never erase the record that the earlier run happened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_unet import write_report  # noqa: E402


def _report(run: str, best: float) -> dict:
    return {
        "run": run,
        "finished_at": "2026-08-08T10:00:00+00:00",
        "commit": "abc1234",
        "best_val_loss": best,
        "best_epoch": 3,
        "final_train_loss": 0.2,
        "final_val_loss": 0.9,
        "n_train_patches": 400,
        "n_val_patches": 100,
        "n_fibrils": 111,
        "train_loss": [1.0, 0.5, 0.3, 0.2],
        "val_loss": [1.1, 0.9, best, 0.9],
        "args": {"epochs": "4"},
    }


def test_writes_a_detailed_report_and_a_summary_line(tmp_path):
    write_report(_report("run_a", 0.42), tmp_path)

    detail = json.loads((tmp_path / "run_a.json").read_text())
    assert detail["train_loss"] == [1.0, 0.5, 0.3, 0.2]

    lines = (tmp_path / "training_runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["run"] == "run_a"
    assert summary["best_val_loss"] == pytest.approx(0.42)
    # The curves stay in the detailed file; the log is a summary.
    assert "train_loss" not in summary


def test_rerunning_the_same_name_keeps_the_earlier_line(tmp_path):
    write_report(_report("run_a", 0.42), tmp_path)
    write_report(_report("run_a", 0.31), tmp_path)

    lines = (tmp_path / "training_runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(x)["best_val_loss"] for x in lines] == [0.42, 0.31]

    # The detailed report is the latest one.
    assert json.loads((tmp_path / "run_a.json").read_text())["best_val_loss"] == 0.31


def test_creates_the_directory_if_missing(tmp_path):
    target = tmp_path / "nested" / "reports"
    write_report(_report("run_a", 0.5), target)
    assert (target / "run_a.json").exists()


def test_the_checked_in_history_is_readable_and_consistent():
    """Guards the real reports/ directory, not a fixture."""
    log = Path(__file__).resolve().parents[1] / "reports" / "training_runs.jsonl"
    runs = [json.loads(line) for line in log.read_text().strip().splitlines()]

    assert runs, "the run log should not be empty"
    for run in runs:
        assert run["run"] and run["finished_at"]
        assert run["best_val_loss"] > 0
        detail = log.parent / f"{run['run']}.json"
        assert detail.exists(), f"summary for {run['run']} has no detailed report"
