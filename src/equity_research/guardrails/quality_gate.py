"""Consolidated release gate for the final research report.

The checks stay independently testable, but the top-level graph needs only one decision point. This
keeps the workflow readable while preserving confidence-language checks, source verification,
behavior monitoring, and report completeness.
"""

from __future__ import annotations

from ..observability.monitor import monitor_node
from .confidence_guard import confidence_guard_node
from .source_guard import source_guard_node


def _missing_sections(state: dict, brief: dict) -> list[str]:
    required = {
        "bull_case": brief.get("bull_case"),
        "bear_case": brief.get("bear_case"),
        "evidence_appendix": brief.get("evidence_appendix"),
    }
    question = str(state.get("question", "") or "").strip().lower()
    broad = not question or any(term in question for term in (
        "research", "overview", "full report", "company analysis", "investment thesis"
    ))
    if broad:
        required.update({
            "executive_summary": brief.get("executive_summary"),
            "company_overview": brief.get("company_overview"),
            "recent_developments": brief.get("recent_developments"),
            "financial_analysis": brief.get("financial_analysis"),
            "management_commentary": brief.get("management_commentary"),
        })
    return [name for name, value in required.items() if not value]


def _report_repair_task_ids(state: dict, missing: list[str]) -> list[str]:
    """Map missing report sections back to the smallest useful evidence-task set."""
    roles_by_section = {
        "bull_case": {"management_upside"},
        "bear_case": {"management_downside", "filing_risk"},
        "company_overview": {"company_profile"},
        "recent_developments": {"recent_news"},
        "management_commentary": {"management_upside", "management_downside", "management_context"},
        "financial_analysis": {
            "metric_revenue_growth", "metric_gross_margin", "metric_operating_margin",
            "metric_net_margin", "metric_fcf_margin",
        },
    }
    wanted: set[str] = set()
    for section in missing:
        wanted.update(roles_by_section.get(section, set()))
    if "evidence_appendix" in missing:
        wanted.update(
            task.get("evidence_role", "")
            for task in (state.get("plan", []) or []) if task.get("required")
        )
    return list(dict.fromkeys(
        task.get("id") for task in (state.get("plan", []) or [])
        if task.get("id") and task.get("evidence_role") in wanted
    ))


def quality_gate_node(state: dict, embeddings=None) -> dict:
    """Run all post-generation checks and return one pass, repair, or review decision."""
    working = dict(state)
    update: dict = {}
    logs: list[str] = []

    confidence = confidence_guard_node(working)
    logs.extend(confidence.pop("log", []))
    working.update(confidence)
    update.update(confidence)

    source = source_guard_node(working, embeddings=embeddings)
    logs.extend(source.pop("log", []))
    working.update(source)
    update.update(source)

    monitor = monitor_node(working)
    logs.extend(monitor.pop("log", []))
    working.update(monitor)
    update.update(monitor)

    brief = working.get("draft_brief", {}) or {}
    missing = _missing_sections(working, brief)
    triggers = list(dict.fromkeys(working.get("review_triggers", []) or []))
    report_rounds = int(working.get("report_repair_rounds", 0) or 0)
    report_max = int(working.get("report_repair_max", 1) or 0)
    report_reopen = _report_repair_task_ids(working, missing)
    report_repair = bool(missing and report_reopen and report_rounds < report_max)
    if missing and not report_repair:
        triggers.append("report_incomplete")
        update["needs_review"] = True
        update["review_triggers"] = list(dict.fromkeys(triggers))
        reason = working.get("escalation_reason", "")
        addition = "missing report sections: " + ", ".join(missing)
        update["escalation_reason"] = "; ".join(part for part in (reason, addition) if part)

    repairable = bool(working.get("unsourced") or working.get("unsupported"))
    if report_repair:
        decision = "repair"
        update.update({
            "report_repair_rounds": report_rounds + 1,
            "reopen": list(dict.fromkeys([
                *(working.get("reopen", []) or []), *report_reopen,
            ])),
        })
    elif repairable and bool(update.get("reopen")):
        decision = "repair"
    elif update.get("needs_review", working.get("needs_review", False)):
        decision = "human_review"
    else:
        decision = "pass"

    report = {
        "decision": decision,
        "confidence": update.get("overconfidence", working.get("overconfidence", {})),
        "source": update.get("source_report", working.get("source_report", {})),
        "monitor": update.get("suspicion_report", working.get("suspicion_report", {})),
        "completeness": {"missing_sections": missing, "passed": not missing},
        "repair_tasks": report_reopen if report_repair else [],
    }
    if decision == "repair":
        # Re-evaluate the thesis from scratch after evidence changes. Keeping prior branches would let
        # an unsupported interpretation survive even after its source task was repaired.
        update.update({"depth": 0, "branches": [], "scored_branches": [], "tot_iterations": 0})

    update.update({
        "quality_decision": decision,
        "quality_report": report,
        "trajectory": [{
            "action": "quality_gate",
            "args": {},
            "observation": f"decision={decision}; missing={missing}",
            "status": "ok" if decision == "pass" else "refused",
        }],
        "log": [*logs, f"quality_gate: decision={decision} missing={missing}"],
    })
    return update
