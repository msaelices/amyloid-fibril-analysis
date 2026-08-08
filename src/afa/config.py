"""Pipeline configuration loaded from YAML (with sensible defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class DetectConfig:
    method: str = "classical"          # "classical" | "unet"
    invert: bool = True
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    prob_threshold: float = 0.2
    unet_weights: str | None = None


@dataclass
class TraceConfig:
    resample_step: float = 1.0
    smooth_window: int = 5
    min_branch_px: int = 15
    link_crossings: bool = True
    max_link_angle_deg: float = 60.0
    merge_junction_px: float = 15.0
    bridge_gap_px: float = 60.0
    bridge_angle_deg: float = 30.0
    min_bridge_evidence: float = 0.25


@dataclass
class Config:
    pixel_size_a: float | None = None   # override; else read from MRC header
    detect: DetectConfig = field(default_factory=DetectConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    bootstrap_ci: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        detect = DetectConfig(**raw.pop("detect", {}))
        trace = TraceConfig(**raw.pop("trace", {}))
        known = {f.name for f in fields(cls)} - {"detect", "trace"}
        rest = {k: v for k, v in raw.items() if k in known}
        return cls(detect=detect, trace=trace, **rest)
