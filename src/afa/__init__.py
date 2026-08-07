"""amyloid-fibril-analysis: tracing and morphology of amyloid fibrils in cryo-EM."""

from __future__ import annotations

__version__ = "0.1.0"

from afa.morphology.metrics import FibrilMetrics, compute_metrics

__all__ = ["FibrilMetrics", "compute_metrics", "__version__"]
