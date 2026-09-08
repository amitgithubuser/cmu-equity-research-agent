"""Observability & human-in-the-loop (architecture §10, Lab 6.2).

Three pieces, all offline-testable:
  * `trajectory`   — the append-only action log with a monitor view (scratchpad excluded).
  * `monitor`      — the weighted-signal suspicion score → `needs_review`.
  * `review_queue` — the ReviewQueue record + the `human_review_node` graph interrupt.

Threshold calibration (`calibration.py`) is added in Phase 10; it consumes labeled runs to set the
`escalation_threshold` from a ROC curve rather than by hand (closes OQ-7).
"""

from .langsmith import graph_run_config, run_tracing_context, tracing_enabled
from .monitor import monitor_node, score_suspicion
from .review_queue import (
    ReviewItem,
    ReviewQueue,
    apply_decision,
    human_review_node,
    review_payload,
)
from .trajectory import TrajectoryLogger, TrajectoryStep

__all__ = [
    "ReviewItem",
    "ReviewQueue",
    "TrajectoryLogger",
    "TrajectoryStep",
    "apply_decision",
    "graph_run_config",
    "human_review_node",
    "monitor_node",
    "review_payload",
    "run_tracing_context",
    "score_suspicion",
    "tracing_enabled",
]
