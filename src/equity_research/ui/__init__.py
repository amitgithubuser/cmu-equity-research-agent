"""Streamlit UI package (Phase 12). `presenter` holds framework-free, unit-tested render helpers;
`app` is the Streamlit shell (run with `streamlit run src/equity_research/ui/app.py`).
"""

from .presenter import (
    brief_markdown,
    case_evaluation_sections,
    evidence_reference,
    evidence_source_rows,
    evidence_strength_label,
    financial_snapshot_rows,
    grouped_case_content,
    management_commentary,
    monitor_summary,
    progress_label,
    runtime_profile_rows,
    trajectory_lines,
    trend_change_rows,
    trends_chart_data,
    trends_table_markdown,
)

__all__ = [
    "brief_markdown", "case_evaluation_sections", "evidence_reference", "evidence_source_rows", "evidence_strength_label",
    "financial_snapshot_rows", "grouped_case_content", "management_commentary",
    "monitor_summary", "progress_label", "runtime_profile_rows", "trajectory_lines",
    "trend_change_rows",
    "trends_chart_data", "trends_table_markdown",
]
