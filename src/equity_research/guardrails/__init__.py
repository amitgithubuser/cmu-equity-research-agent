"""Three-stage guardrail pipeline (architecture §9, Lab 6.1): pre-gen input check, during-gen
overconfidence scan, post-gen claim-source verification. Each is a plain-Python function wired as a
LangGraph node — deterministic, unit-testable, and independently ablatable (Phase 10).
"""

from .confidence_guard import confidence_guard_node, scan_overconfidence
from .input_guard import check_ticker, detect_advice, input_guard_node
from .quality_gate import quality_gate_node
from .source_guard import source_guard_node, verify_claims

__all__ = [
    "check_ticker",
    "confidence_guard_node",
    "detect_advice",
    "input_guard_node",
    "quality_gate_node",
    "scan_overconfidence",
    "source_guard_node",
    "verify_claims",
]
