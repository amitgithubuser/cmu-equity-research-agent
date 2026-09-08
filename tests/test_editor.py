"""Phase 7 gate (implementation-plan §8): schema-valid brief; EVERY claim carries a Citation; the
disclaimer is always present; a claim lacking a source flips `unsourced` -> back-edge.
"""

from equity_research.agents.editor import editor_node
from equity_research.llm import FakeLLM
from equity_research.schemas import ResearchBrief


def _branch(side, claim, snippet="evidence text"):
    return {
        "side": side, "support_score": 0.8, "survived": True,
        "claims": [{
            "claim": claim, "kind": "text",
            "citation": {"source": "10-K FY2025", "company": "NVDA", "period": "FY2025", "snippet": snippet},
        }],
    }


def test_editor_emits_schema_valid_brief():
    state = {"ticker": "NVDA", "as_of": "FY2025", "confidence": 0.7,
             "scored_branches": [_branch("bull", "margins expanded"), _branch("bear", "concentration risk")]}
    out = editor_node(state, FakeLLM())
    brief = ResearchBrief.model_validate(out["draft_brief"])  # must validate
    assert brief.ticker == "NVDA"


def test_every_claim_has_citation():
    state = {"ticker": "NVDA", "as_of": "FY2025", "confidence": 0.7,
             "scored_branches": [_branch("bull", "a"), _branch("bear", "b"), _branch("risk", "c")]}
    out = editor_node(state, FakeLLM())
    brief = ResearchBrief.model_validate(out["draft_brief"])
    for group in (brief.bull_case, brief.bear_case, brief.key_risks):
        for ev in group:
            assert ev.citation.source and ev.citation.snippet
    assert out["unsourced"] is False


def test_disclaimer_always_present():
    out = editor_node({"ticker": "X", "as_of": "FY", "confidence": 0.5, "scored_branches": []}, FakeLLM())
    assert "not financial advice" in out["draft_brief"]["disclaimer"].lower()


def test_editor_uses_engine_confidence_not_invented():
    state = {"ticker": "NVDA", "as_of": "FY2025", "confidence": 0.42,
             "scored_branches": [_branch("bull", "a")]}
    out = editor_node(state, FakeLLM())
    assert out["confidence"] == 0.42  # carried from the ToT engine, not re-derived by the Editor


def test_unsourced_claim_flips_backedge():
    # A synthesizer that emits a brief with a blank-source citation -> unsourced True -> back-edge.
    def bad_synth(state, llm):
        return ResearchBrief(
            ticker="NVDA", as_of="FY2025",
            bull_case=[{"claim": "x", "kind": "text",
                        "citation": {"source": "", "company": "NVDA", "period": "FY2025", "snippet": ""}}],
            bear_case=[], key_risks=[], confidence=0.5, confidence_rationale="",
        )

    out = editor_node({"ticker": "NVDA", "as_of": "FY2025", "editor_rounds": 0}, FakeLLM(), synthesizer=bad_synth)
    assert out["unsourced"] is True
    assert out["editor_rounds"] == 1  # increments so the back-edge is bounded


def test_insufficient_side_noted_in_rationale():
    state = {"ticker": "NVDA", "as_of": "FY2025", "confidence": 0.3,
             "scored_branches": [_branch("bear", "concentration")], "insufficient": {"bull": True, "bear": False}}
    out = editor_node(state, FakeLLM())
    assert "insufficient" in out["draft_brief"]["confidence_rationale"].lower()
    assert out["draft_brief"]["bull_case"] == []  # starved side left empty, not invented


def test_editor_attaches_trends_from_history():
    # history in state -> the brief carries an oldest-first per-period trend series for charting
    state = {
        "ticker": "NVDA", "as_of": "FY2025", "confidence": 0.7,
        "scored_branches": [_branch("bull", "margins expanded")],
        "history": {"line_items": {
            "2024-01-31": {"revenue": 60922.0, "cogs": 16621.0, "net_income": 29760.0},
            "2023-01-31": {"revenue": 26974.0, "cogs": 11618.0, "net_income": 4368.0},
        }},
    }
    out = editor_node(state, FakeLLM())
    trends = out["draft_brief"]["trends"]
    assert [p["period"] for p in trends] == ["2023-01-31", "2024-01-31"]
    assert trends[0]["gross_margin"] == round((26974.0 - 11618.0) / 26974.0, 4)
    assert "revenue_growth" in trends[1]           # second period has a prior to compare against


def test_editor_no_history_means_empty_trends():
    out = editor_node({"ticker": "X", "as_of": "FY", "confidence": 0.5, "scored_branches": []}, FakeLLM())
    assert out["draft_brief"]["trends"] == []       # no fabricated chart when there's no data


def test_editor_deduplicates_reused_evidence_across_surviving_depths():
    repeated = _branch("bull", "analyst consensus is positive")
    state = {"ticker": "NVDA", "as_of": "FY2025", "confidence": 0.3,
             "scored_branches": [repeated, repeated]}
    out = editor_node(state, FakeLLM())
    assert len(out["draft_brief"]["bull_case"]) == 1


def test_rationale_describes_final_assembled_sides_not_stale_critic_iteration():
    state = {"ticker": "NVDA", "as_of": "FY2025", "confidence": 0.3,
             "scored_branches": [_branch("bull", "grounded bull")],
             "insufficient": {"bull": True, "bear": True}}
    rationale = editor_node(state, FakeLLM())["draft_brief"]["confidence_rationale"]
    assert "bull side had insufficient" not in rationale
    assert "bear side had insufficient" in rationale


def test_editor_retains_rich_report_material_from_blackboard():
    state = {
        "ticker": "NVDA", "as_of": "FY2025", "confidence": 0.7,
        "scored_branches": [{
            **_branch("bull", "demand remained strong"),
            "thesis": "AI demand supports growth",
        }],
        "metrics": {
            "gross_margin": {
                "name": "gross_margin", "value": 0.73,
                "inputs": {"revenue": 100.0, "cogs": 27.0},
            },
        },
        "qoq": {"has_prior": True, "summary": "gross margin increased"},
        "ratings": {"total": 10, "buy": 8},
        "evidence": [_branch("bull", "demand remained strong")["claims"][0]],
        "open_uncertainties": ["customer concentration needs a newer disclosure"],
    }
    brief = editor_node(state, FakeLLM())["draft_brief"]
    assert brief["theses"][0]["thesis"] == "AI demand supports growth"
    assert brief["metrics"]["gross_margin"]["value"] == 0.73
    assert brief["qoq"]["has_prior"] is True and brief["ratings"]["total"] == 10
    assert len(brief["evidence_appendix"]) == 1
    assert brief["open_questions"] == ["customer concentration needs a newer disclosure"]


def test_editor_builds_all_reader_facing_sections_from_verified_state():
    profile = {
        "claim": "NVIDIA designs accelerated-computing platforms for data centers.",
        "kind": "text",
        "citation": {"source": "Company profile", "company": "NVDA", "period": "current",
                     "snippet": "accelerated-computing platforms for data centers"},
    }
    news = {
        "claim": "The company announced a new platform roadmap.", "kind": "text",
        "citation": {"source": "News: company release", "company": "NVDA", "period": "2025-01",
                     "snippet": "announced a new platform roadmap"},
    }
    transcript = {
        "claim": "Management said demand remained strong.", "kind": "text",
        "citation": {"source": "Earnings call transcript Q4-2025", "company": "NVDA",
                     "period": "Q4-2025", "snippet": "demand remained strong"},
    }
    state = {
        "ticker": "NVDA", "as_of": "FY2025", "confidence": 0.75,
        "scored_branches": [
            {**_branch("bull", "demand remained strong"), "thesis": "demand supports growth"},
            {**_branch("bear", "customer concentration remains a risk"),
             "thesis": "concentration can amplify volatility"},
        ],
        "metrics": {"gross_margin": {"name": "gross_margin", "value": 0.73, "inputs": {}}},
        "evidence": [profile, news, transcript],
        "recent_developments": [news],
        "transcript_status": "ingested", "transcript_periods": ["Q4-2025"],
        "transcript_chunks": 6, "filing_status": "ingested", "filing_forms": ["10-K"],
        "filing_periods": ["2025-01-31"], "filing_chunks": 12,
    }
    brief = editor_node(state, FakeLLM())["draft_brief"]
    assert brief["executive_summary"]
    assert brief["company_overview"]["citation"]["source"] == "Company profile"
    assert len(brief["recent_developments"]) == 1
    assert brief["financial_analysis"] == ["Gross margin is 73.0% for FY2025."]
    assert len(brief["management_commentary"]) == 1
    assert brief["source_coverage"]["transcript_status"] == "ingested"
    assert brief["source_coverage"]["filing_chunks"] == 12
