"""Phase 1 gate (implementation-plan §2.3): the schema-level guardrails proven in isolation.

An `Evidence` without a `Citation` cannot be constructed; a `ResearchBrief` always carries the
disclaimer; confidence is bounded [0,1].
"""

import pytest
from pydantic import ValidationError

from equity_research.schemas import (
    Citation,
    Evidence,
    EvidencePlan,
    EvidenceTask,
    ResearchBrief,
    ThesisBranch,
)


def _citation() -> Citation:
    return Citation(source="10-K FY2025 Item 7", company="TEST", period="FY2025", snippet="revenue rose 20%")


# --- guardrail #1: provenance is mandatory ---------------------------------------------------------
def test_evidence_requires_citation():
    with pytest.raises(ValidationError):
        Evidence(claim="rev up 20%", kind="number", value=20.0)  # no citation -> must fail


def test_evidence_with_citation_ok():
    ev = Evidence(claim="rev up 20%", kind="number", value=20.0, citation=_citation())
    assert ev.citation.company == "TEST"


# --- guardrail #2: disclaimer always present -------------------------------------------------------
def test_brief_disclaimer_default():
    b = ResearchBrief(
        ticker="X", as_of="FY2025", bull_case=[], bear_case=[], key_risks=[],
        confidence=0.5, confidence_rationale="balanced",
    )
    assert "not financial advice" in b.disclaimer.lower()


# --- confidence bounds -----------------------------------------------------------------------------
def test_confidence_bounded_high():
    with pytest.raises(ValidationError):
        ResearchBrief(ticker="X", as_of="FY2025", bull_case=[], bear_case=[], key_risks=[], confidence=1.4)


def test_confidence_bounded_low():
    with pytest.raises(ValidationError):
        ResearchBrief(ticker="X", as_of="FY2025", bull_case=[], bear_case=[], key_risks=[], confidence=-0.1)


# --- round-trips -----------------------------------------------------------------------------------
def test_brief_roundtrip_preserves_citations():
    ev = Evidence(claim="gross margin 40%", kind="number", value=0.40, citation=_citation())
    b = ResearchBrief(
        ticker="TEST", as_of="FY2025", bull_case=[ev], bear_case=[], key_risks=[],
        confidence=0.7, confidence_rationale="bull better supported",
    )
    reloaded = ResearchBrief.model_validate(b.model_dump())
    assert reloaded.bull_case[0].citation.snippet == "revenue rose 20%"


def test_thesis_branch_defaults():
    branch = ThesisBranch(side="bull", claims=[])
    assert branch.support_score == 0.0 and branch.survived is False


def test_evidence_plan_roundtrip():
    plan = EvidencePlan(
        tasks=[
            EvidenceTask(claim="gross margin trend", kind="number", metric="gross_margin"),
            EvidenceTask(claim="key risk factors", kind="text", section_hint="Item 1A"),
        ],
        stopping_note="enough when both bull and bear have >=2 cited claims",
    )
    reloaded = EvidencePlan.model_validate(plan.model_dump())
    assert len(reloaded.tasks) == 2 and reloaded.tasks[0].metric == "gross_margin"
