"""Planner node (architecture §4, agent #1) — Haiku, cheap decomposition.

Turns ticker + question into an evidence plan and stopping conditions. `max_retries` is derived from
the plan size so the research loop is bounded relative to the work to do (implementation-plan §6.1).
"""

from __future__ import annotations

from ..schemas import EvidencePlan
from ..window import resolve_analysis_window
from .prompts import PLANNER_PROMPT


def assign_task_ids(tasks: list[dict]) -> list[dict]:
    """Give every plan task a stable `id` (t0, t1, ...) unless it already has one (F-01).

    Completion is tracked by this id, so the Researcher can mark a task done even though the LLM's
    generated evidence claim differs from the task wording. Deterministic and idempotent.
    """
    out = []
    for i, t in enumerate(tasks):
        task = dict(t)
        if not task.get("id"):
            task["id"] = f"t{i}"
        out.append(task)
    return out


def _is_broad_request(question: str) -> bool:
    text = (question or "").strip().lower()
    return not text or any(term in text for term in (
        "research", "overview", "full report", "company analysis", "investment thesis"
    ))


def _ensure_task(tasks: list[dict], task: dict) -> None:
    """Append a task unless its report role (or numeric metric) is already represented."""
    match = None
    if task.get("kind") == "number":
        match = next((
            item for item in tasks
            if item.get("kind") == "number" and item.get("metric") == task.get("metric")
        ), None)
    elif task.get("evidence_role") not in {None, "", "question_specific"}:
        match = next((
            item for item in tasks if item.get("evidence_role") == task.get("evidence_role")
        ), None)
        if match is None and task.get("evidence_role") == "management_context":
            match = next((
                item for item in tasks
                if item.get("kind") == "text" and any(
                    term in (
                        str(item.get("claim", "")) + " " + str(item.get("section_hint", ""))
                    ).lower()
                    for term in ("earnings", "transcript", "conference call")
                )
            ), None)
        if match is None and task.get("kind") in {"profile", "news", "ratings"}:
            match = next((item for item in tasks if item.get("kind") == task.get("kind")), None)
    else:
        match = next((
            item for item in tasks
            if item.get("kind") == task.get("kind") and item.get("claim") == task.get("claim")
        ), None)
    if match is None:
        tasks.append(task)
        return
    # Preserve the Planner's wording while attaching deterministic coverage metadata.
    match.setdefault("evidence_role", task.get("evidence_role", "question_specific"))
    if task.get("evidence_role") == "management_context" and not match.get("section_hint"):
        match["section_hint"] = "earnings call transcript"
    if task.get("required"):
        match["required"] = True


def ensure_report_coverage(tasks: list[dict], question: str) -> list[dict]:
    """Guarantee the evidence needed for management context and a detailed broad report."""
    out = [dict(task) for task in tasks]
    for task in out:
        if task.get("kind") != "text" or task.get("evidence_role") not in {
            None, "", "question_specific",
        }:
            continue
        text = (str(task.get("claim", "")) + " " + str(task.get("section_hint", ""))).lower()
        if "item 1a" in text or ("filing" in text and "risk" in text):
            task.update(evidence_role="filing_risk", required=True)
        elif any(term in text for term in ("earnings", "transcript", "conference call")):
            if any(term in text for term in (
                "risk", "downside", "constraint", "competition", "cost", "uncertainty",
            )):
                task.update(evidence_role="management_downside", required=True)
            elif any(term in text for term in (
                "growth", "demand", "upside", "driver", "outlook", "catalyst",
            )):
                task.update(evidence_role="management_upside", required=True)
            else:
                task["evidence_role"] = "management_context"
    if not _is_broad_request(question):
        _ensure_task(out, {
            "claim": "management commentary relevant to the research question",
            "kind": "text",
            "section_hint": "earnings call transcript",
            "evidence_role": "management_context",
        })
        return out

    baseline = [
        {
            "claim": "management evidence for growth drivers, demand, execution, and upside catalysts",
            "kind": "text", "section_hint": "earnings call transcript",
            "evidence_role": "management_upside", "required": True,
        },
        {
            "claim": "management evidence for constraints, uncertainty, competition, costs, and downside risks",
            "kind": "text", "section_hint": "earnings call transcript",
            "evidence_role": "management_downside", "required": True,
        },
        {
            "claim": "material risk factors and downside evidence from the latest SEC filing",
            "kind": "text", "section_hint": "Item 1A Risk Factors and Item 7 MD&A",
            "evidence_role": "filing_risk", "required": True,
        },
        {"claim": "company profile and business model", "kind": "profile",
         "evidence_role": "company_profile", "required": True},
        {"claim": "recent material company developments", "kind": "news",
         "evidence_role": "recent_news", "required": True},
        {"claim": "current analyst recommendation distribution", "kind": "ratings",
         "evidence_role": "analyst_ratings"},
        {"claim": "revenue growth trend", "kind": "number", "metric": "revenue_growth",
         "evidence_role": "metric_revenue_growth", "required": True},
        {"claim": "gross margin trend", "kind": "number", "metric": "gross_margin",
         "evidence_role": "metric_gross_margin", "required": True},
        {"claim": "operating margin", "kind": "number", "metric": "operating_margin",
         "evidence_role": "metric_operating_margin", "required": True},
        {"claim": "net margin", "kind": "number", "metric": "net_margin",
         "evidence_role": "metric_net_margin", "required": True},
        {"claim": "free cash flow margin", "kind": "number", "metric": "fcf_margin",
         "evidence_role": "metric_fcf_margin", "required": True},
    ]
    for task in baseline:
        _ensure_task(out, task)
    # The two directional transcript tasks supersede an extra generic transcript request.
    out = [task for task in out if task.get("evidence_role") != "management_context"]
    return out


def planner_node(state: dict, llm) -> dict:
    """Decompose ticker+question into an evidence plan + stopping conditions."""
    window = state.get("analysis_window") or resolve_analysis_window(
        state.get("question", ""), state.get("as_of", "")
    ).to_dict()
    prompt = PLANNER_PROMPT.format(
        ticker=state.get("ticker", ""),
        question=state.get("question", "") or "general research",
        analysis_window=window,
        transcript_status=state.get("transcript_status", "not checked"),
        review_feedback=(
            state.get("human_note", "") if state.get("human_decision") == "revise" else "none"
        ) or "none",
    )
    plan: EvidencePlan = llm.with_structured_output(EvidencePlan).invoke(prompt)
    tasks = ensure_report_coverage([t.model_dump() for t in plan.tasks], state.get("question", ""))
    tasks = assign_task_ids(tasks)
    # One pass executes every task once. Each allowed source-repair round can reopen as many as every
    # task, so reserve enough bounded executions for the configured repair ceiling. This avoids
    # cutting off the last task merely because an earlier quality check requested several repairs.
    repair_rounds = sum(max(0, int(state.get(name, default) or 0)) for name, default in (
        ("editor_reretrieve_max", 1),
        ("balance_repair_max", 1),
        ("report_repair_max", 1),
    ))
    research_budget = len(tasks) * (1 + repair_rounds)
    return {
        "plan": tasks,
        "analysis_window": window,
        # bounded worst case: initial pass plus every configured targeted repair round
        "max_retries": research_budget,
        "retries": 0,
        "editor_rounds": 0,
        "balance_gaps": [],
        "balance_repair_rounds": 0,
        "report_repair_rounds": 0,
        "depth": 0,
        "need_more": bool(tasks),
        # A human revision reuses the same source window but reopens every newly planned task. The
        # covered channel is append-only, so reopen is the explicit, bounded way to run them again.
        "reopen": [task["id"] for task in tasks] if state.get("human_decision") == "revise" else [],
        "human_decision": "",
        "revision_requested": False,
        "trajectory": [{
            "action": "planner",
            "args": {"ticker": state.get("ticker", ""), "task_count": len(tasks)},
            "observation": "evidence plan created",
            "status": "ok",
        }],
        "log": [f"planner: {len(tasks)} tasks; stop when {plan.stopping_note or 'plan exhausted'}"],
    }
