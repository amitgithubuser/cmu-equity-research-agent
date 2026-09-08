"""The blackboard — LangGraph shared state (architecture §5.4, Lab 5.1 `ControlState`).

Agents do not message point-to-point; they read/write this single object (architecture §5.2).
The one subtlety is the `Annotated[list, add]` **reducers**: LangGraph merges each node's returned
partial state through the reducer, so `evidence`/`log`/`open_uncertainties` are *appended* across
rounds instead of clobbered. Fields without a reducer are last-write-wins.

Note: `TypedDict` gives us the shape for LangGraph without importing langgraph here, so this module
(and the tests that use it) stay dependency-light and offline.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages


def merge_evidence(current: list[dict] | None, incoming: list[dict] | None) -> list[dict]:
    """Append new task evidence while replacing an older attempt of the same task.

    Most evidence accumulates across different plan tasks. A quality repair is different: its new
    evidence must supersede the unsupported prior attempt, otherwise the old claim remains in the
    report forever and no repair can succeed.
    """
    current = list(current or [])
    incoming = list(incoming or [])
    replaced = {item.get("source_task") for item in incoming if item.get("source_task")}
    if replaced:
        current = [item for item in current if item.get("source_task") not in replaced]
    return [*current, *incoming]


class AgentState(TypedDict, total=False):
    # --- inputs ---
    ticker: str
    question: str
    as_of: str                       # reporting period / date anchor (prevents stale-data leaks)
    use_source_cache: bool           # reuse complete cached transcripts/filings unless refreshed
    analysis_window: dict            # one scope shared by transcripts, market data, and RAG
    transcript_status: str           # "ingested" | "cached" | "partial" | "unavailable"
    transcript_periods: list[str]
    transcript_chunks: int
    transcript_error: str
    filing_status: str               # "ingested" | "partial" | "unavailable"
    filing_forms: list[str]
    filing_periods: list[str]
    filing_chunks: int
    filing_error: str

    # --- agent/tool conversation ---
    # LangGraph's message reducer keeps the Researcher <-> ToolNode loop auditable and lets the
    # compiled graph use the same model/tool/observation pattern shown in the official docs.
    messages: Annotated[list, add_messages]
    research_action: dict             # latest Researcher decision, for UI/trace explanations

    # --- planning ---
    plan: list[dict]                 # list[EvidenceTask]
    active_task: dict                # task selected by Researcher for the next ToolNode action
    covered: Annotated[list[str], add]   # ids of plan tasks already attempted (F-01: completion by id)
    reopen: list[str]                # ids forced back to "open" for targeted repair (F-07b source-guard)
    open_uncertainties: Annotated[list[str], add]

    # --- evidence (blackboard core) ---
    evidence: Annotated[list[dict], merge_evidence]  # append tasks; replace a repaired task attempt
    metrics: dict                          # computed ratios keyed by name, each with provenance
    history: dict                          # multi-period line items (market_history) for trend charts
    qoq: dict                              # quarter-over-quarter diff vs. long-term memory (§6.1)
    ratings: dict                          # analyst consensus recommendation distribution (F-04, cited)
    company_profile: dict                  # sourced company identity and business description
    recent_developments: list[dict]        # cited current-news evidence retained for the report
    news_sentiment: float                  # attributed headline tone signal in [-1,1] (F-04)

    # --- thesis / ToT ---
    branches: list[dict]             # list[ThesisBranch]
    scored_branches: list[dict]
    depth: int
    tot_iterations: int

    # --- synthesis ---
    draft_brief: dict                # pre-guardrail draft
    brief: dict                      # final ResearchBrief (Pydantic-validated)
    confidence: float

    # --- control (Lab 5.1 pattern) ---
    retries: int                     # research-loop counter
    max_retries: int
    max_depth: int                   # ToT depth cap (mirrors settings.tot_max_depth into state)
    editor_rounds: int               # Editor->Researcher re-retrieval counter (bounds the back-edge)
    editor_reretrieve_max: int
    balance_gaps: list[str]          # required evidence roles that produced no usable evidence
    balance_repair_rounds: int       # bounded pre-thesis evidence repair counter
    balance_repair_max: int
    report_repair_rounds: int        # bounded post-report completeness repair counter
    report_repair_max: int
    route: Literal["numbers", "text"]
    need_more: bool                  # research loop: Planner stopping condition not yet met
    unsourced: bool                  # Editor: a claim lacked a citation -> back-edge to Researcher
    unsupported: bool                # source_guard: a claim's support fell below floor -> back-edge
    proceed: bool                    # input_guard verdict (False => fail-safe reject)
    needs_review: bool               # monitor verdict -> HITL
    escalation_reason: str
    human_decision: str              # "allow" | "revise" | "block" (from interrupt resume)
    human_note: str                  # reviewer's free-text note (from interrupt resume)
    human_revision_rounds: int       # completed reviewer-requested research passes
    max_human_revisions: int         # cap for reviewer-requested graph loops
    blocked: bool                    # finalize: brief withheld after a human BLOCK (§10.4 fail-safe)
    revision_requested: bool         # finalize: human asked for another pass
    error: str
    log: Annotated[list[str], add]   # human-readable trace lines

    # NOTE: every key a node returns MUST be declared here — real LangGraph only persists declared
    # channels across super-steps (undeclared keys are silently dropped between nodes). There is no
    # home-grown fallback engine, so this declaration list is the single source of truth for state.

    # --- guardrail verdicts (Lab 6.1) ---
    issues: list[str]                # input_guard: why an input was rejected/flagged
    risk_level: str                  # input_guard: classified risk of the query
    advice_rerouted: bool            # input_guard: an advice question was reframed to a research one
    overconfidence: dict             # confidence_guard: overconfidence scan {score,verdict,flagged,...}
    overconfident: bool              # confidence_guard: FAIL survived softening -> monitor HITL (F-07)
    groundedness: float              # source_guard: fraction of claims supported by evidence
    source_report: dict              # source_guard: per-claim support detail
    quality_report: dict             # consolidated confidence/source/monitor/completeness results
    quality_decision: Literal["pass", "repair", "human_review"]

    # --- observability (Lab 6.2) ---
    trajectory_id: str
    trajectory: Annotated[list[dict], add]   # TrajectoryLogger steps (monitor reads the public view)
    suspicion: float
    suspicion_report: dict           # monitor: {suspicion, fired, detail}
    review_triggers: list[str]        # hard HITL reasons, including missing transcript coverage
    insufficient: dict[str, bool]    # critic: per-side evidence starvation, e.g. {"bear": True}
