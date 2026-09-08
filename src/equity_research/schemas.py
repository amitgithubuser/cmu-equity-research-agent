"""Pydantic contracts (architecture §12) — the structured-output target for the Editor and the
assertion surface for guardrail + regression tests.

Two schema-level guardrails are enforced *by construction* here (architecture §9.4):
  1. `Evidence` cannot exist without a `Citation`  → every factual claim is provenance-carrying.
  2. `ResearchBrief` always carries the "not financial advice" disclaimer (defaulted, non-empty).

Also defined here: the Planner's `EvidencePlan` / `EvidenceTask` (implementation-plan §6.1), which
are structured-output targets too but are internal plumbing rather than the public brief.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- evidence & provenance
class Citation(BaseModel):
    """Where a claim comes from — enough for a human to independently verify it."""

    source: str                        # "10-K 2025 Item 1A" / transcript / news headline
    company: str
    period: str                        # e.g., "FY2025", "Q2-2026"
    snippet: str                       # the short quote that actually supports the claim
    url: str | None = None


class Evidence(BaseModel):
    """A single factual statement with mandatory provenance. No citation => cannot be constructed."""

    claim: str
    kind: Literal["text", "number"]
    citation: Citation                 # provenance is required (schema-level guardrail #1)
    value: str | float | None = None   # set for numeric evidence


class Metric(BaseModel):
    """A ratio computed in code by `calc` (never by an LLM). Carries its raw inputs for provenance."""

    name: str                          # "gross_margin", "revenue_growth", "debt_to_equity"
    value: float
    inputs: dict                       # raw line items used
    vendor_value: float | None = None  # cross-check from vendor_ratio
    divergence_flag: bool = False      # set when |calc - vendor| / |vendor| exceeds the threshold
    period: str = ""                   # exact period represented by the computed value
    source: str = ""                   # provider/document used for the raw inputs


# --------------------------------------------------------------------------- thesis / ToT
class ThesisBranch(BaseModel):
    """One partial thesis in the ToT search (architecture §8.2)."""

    side: Literal["bull", "bear", "risk"]
    thesis: str = ""                    # the grounded interpretation produced by the Analyst
    claims: list[Evidence]
    support_score: float = 0.0         # from the Critic rubric
    survived: bool = False
    mechanism: str = ""
    time_horizon: str = ""
    catalysts: list[str] = Field(default_factory=list)
    watch_items: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    evidence_rationales: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- ToT LLM I/O (F-02)
class AngleClaim(BaseModel):
    """One claim inside a generated angle, tied to evidence ALREADY on the blackboard by its index.

    The Analyst does not mint new facts — it selects and frames existing evidence. `evidence_index`
    points at the gathered-evidence list, so every angle claim inherits a real citation (the evidence
    gate then rejects any angle whose claims don't resolve to cited/numeric evidence).
    """

    evidence_index: int                # index into state["evidence"]; -1 if the model went off-source
    rationale: str = ""                # one line: why this evidence supports the angle's side


class Angle(BaseModel):
    """One competing thesis angle the Analyst proposes (bull / bear / risk)."""

    side: Literal["bull", "bear", "risk"]
    thesis: str                        # the one-sentence angle being argued
    claims: list[AngleClaim] = Field(default_factory=list)
    mechanism: str = ""
    time_horizon: str = ""
    catalysts: list[str] = Field(default_factory=list)
    watch_items: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)


class AnalystAngles(BaseModel):
    """Structured-output target for the Analyst generation call (F-02)."""

    angles: list[Angle] = Field(default_factory=list)


class CriticScore(BaseModel):
    """Structured-output target for one Critic scoring call (F-02).

    The Critic returns a bounded score plus a short justification, so a real run has an auditable
    reason for every keep/cut — not just a bare float.
    """

    support_score: float = Field(ge=0, le=1)
    justification: str = ""


class CriticAssessment(BaseModel):
    """Four explicit rubric scores for one indexed thesis branch."""

    branch_index: int = Field(ge=0)
    evidence_support: float = Field(ge=0, le=1)
    consistency: float = Field(ge=0, le=1)
    materiality: float = Field(ge=0, le=1)
    survival: float = Field(ge=0, le=1)
    justification: str = ""

    @property
    def support_score(self) -> float:
        """Deterministic overall score; the model cannot hide the rubric behind one opaque number."""
        values = (self.evidence_support, self.consistency, self.materiality, self.survival)
        return round(sum(values) / len(values), 4)


class CriticAssessments(BaseModel):
    """Batch output for one independent Critic pass over all branches at a ToT depth."""

    assessments: list[CriticAssessment] = Field(default_factory=list)


# --------------------------------------------------------------------------- the public deliverable
class ResearchBrief(BaseModel):
    """The final research report.

    The three thesis sections remain the decision summary. The additional fields retain the richer
    evidence already present on the graph blackboard so the UI can show financial charts, detailed
    thesis reasoning, management commentary, source provenance, and unresolved questions without
    asking an LLM to recreate or embellish them.
    """

    ticker: str
    as_of: str
    executive_summary: str = ""
    company_overview: Evidence | None = None
    recent_developments: list[Evidence] = Field(default_factory=list)
    financial_analysis: list[str] = Field(default_factory=list)
    management_commentary: list[Evidence] = Field(default_factory=list)
    bull_case: list[Evidence]
    bear_case: list[Evidence]
    key_risks: list[Evidence]
    theses: list[ThesisBranch] = Field(default_factory=list)
    metrics: dict[str, Metric] = Field(default_factory=dict)
    qoq: dict = Field(default_factory=dict)
    ratings: dict = Field(default_factory=dict)
    evidence_appendix: list[Evidence] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)   # set from the bull/bear support gap (architecture §8.4)
    confidence_rationale: str = ""
    # Per-period ratio series for trend charts (revenue + margins + growth). Derived by
    # `analytics.build_trends` from `market_history` line items — same `calc` definitions, so a chart
    # never shows a number that isn't traceable to real line items. Empty when no history was fetched.
    trends: list[dict] = Field(default_factory=list)
    source_coverage: dict = Field(default_factory=dict)
    # Schema-level guardrail #2 — a non-empty, defaulted disclaimer that always ships.
    disclaimer: str = "This is research, not financial advice."


# --------------------------------------------------------------------------- planner plumbing
class EvidenceTask(BaseModel):
    """One item the Researcher must gather. `kind` decides the route: number->lookup, text->RAG.

    `id` is the STABLE identity used to track completion (F-01 fix): the Researcher marks a task done by
    id, never by matching the LLM-generated evidence claim text (which differs from the task wording).
    The Planner may omit it; `assign_task_ids()` fills a deterministic `t{n}` before the loop runs.
    """

    id: str = ""                       # stable task id (t0, t1, ...); assigned if the Planner omits it
    claim: str                         # what we need to establish, e.g. "gross margin trend last 4 qtrs"
    # Routing kind (F-04): number->calc, text->RAG, ratings->analyst_ratings, news->news_search.
    # `ratings`/`news` are third-party cross-check evidence the agent ATTRIBUTES, never adopts (D-13).
    kind: Literal["text", "number", "profile", "ratings", "news"]
    metric: str | None = None          # for numeric tasks: which ratio calc should compute
    section_hint: str | None = None    # for text tasks: e.g. "Item 1A Risk Factors"
    evidence_role: str = "question_specific"  # report purpose, distinct from retrieval kind
    required: bool = False             # required baseline evidence for a broad report


class EvidencePlan(BaseModel):
    """The Planner's decomposition of ticker + question into gatherable tasks + stopping conditions."""

    tasks: list[EvidenceTask]
    stopping_note: str = ""            # human-readable "enough evidence when ..." condition


class ResearchAction(BaseModel):
    """The Researcher's next action in the agent/ToolNode loop.

    Defaults make the model boundary fail safe: if a model returns an incomplete decision, the
    Researcher falls back to the first unfinished plan task and its deterministic task kind.
    """

    task_id: str = ""
    tool: Literal[
        "gather_number", "gather_text", "gather_profile", "gather_news", "gather_ratings"
    ] = "gather_text"
    reason: str = ""
    done: bool = False


class PassageClaim(BaseModel):
    """One grounded text claim tied to exactly one retrieved passage by zero-based index."""

    evidence_index: int = Field(ge=0)
    claim: str


class PassageClaims(BaseModel):
    """Structured Researcher output; citations are attached by code from the selected passages."""

    claims: list[PassageClaim] = Field(default_factory=list)
