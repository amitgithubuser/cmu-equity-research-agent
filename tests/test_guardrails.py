"""Phase 8 gate (implementation-plan §9.4): each guard tested in isolation on known-bad inputs; an
advice query is caught pre-gen; an unsourced claim provably flags the back-edge.
"""

from equity_research.guardrails import (
    check_ticker,
    confidence_guard_node,
    detect_advice,
    input_guard_node,
    quality_gate_node,
    scan_overconfidence,
    source_guard_node,
    verify_claims,
)
from equity_research.guardrails.source_guard import claim_coverage


# --- Stage 1: input_guard --------------------------------------------------------------------------
def test_input_guard_rejects_bad_ticker():
    out = input_guard_node({"ticker": "not a ticker!", "question": "how are margins?"})
    assert out["proceed"] is False and out["escalation_reason"] == "invalid_ticker"


def test_input_guard_accepts_valid_ticker():
    out = input_guard_node({"ticker": "NVDA", "question": "how are margins trending?"})
    assert out["proceed"] is True and out["risk_level"] == "low"


def test_input_guard_blocks_and_reroutes_advice():
    out = input_guard_node({"ticker": "NVDA", "question": "should I buy NVDA right now?"})
    # advice is re-routed (not hard-refused) to an evidence-only framing, but flagged for escalation
    assert out["proceed"] is True and out.get("advice_rerouted") is True
    assert "buy" not in out["question"].lower()
    assert out["escalation_reason"] == "advice_query"


def test_input_guard_unknown_symbol_when_list_given():
    out = input_guard_node({"ticker": "ZZZZ", "question": "margins?"}, known_symbols={"NVDA", "AAPL"})
    assert out["proceed"] is False and out["escalation_reason"] == "unknown_ticker"


def test_check_ticker_and_detect_advice_units():
    assert check_ticker("NVDA") and check_ticker("BRK.B") and not check_ticker("hello world")
    assert detect_advice("should i sell my shares?") and not detect_advice("what were revenues?")


# --- Stage 2: confidence_guard ---------------------------------------------------------------------
def test_confidence_flags_overconfident_prose():
    text = "This stock is guaranteed to rise. There is no risk. It will always outperform."
    result = scan_overconfidence(text)
    assert result["verdict"] in ("WARN", "FAIL") and result["score"] > 0
    assert "guaranteed" in result["suggestions"]


def test_confidence_passes_calibrated_prose():
    text = "Margins expanded last quarter. Growth is likely to continue given demand. Risks remain."
    assert scan_overconfidence(text)["verdict"] == "PASS"


def test_confidence_guard_node_scans_brief():
    brief = {"confidence_rationale": "This is guaranteed to work with no risk.",
             "bull_case": [], "bear_case": [], "key_risks": []}
    out = confidence_guard_node({"draft_brief": brief})
    assert out["overconfidence"]["verdict"] in ("WARN", "FAIL")


def test_confidence_guard_softens_flagged_prose_in_place():
    # F-07: a flagged brief is rewritten with calmer phrasing, and the softened brief replaces
    # draft_brief so downstream nodes (source_guard, monitor, finalize) see the calm text.
    brief = {"confidence_rationale": "Growth is guaranteed with no risk.",
             "bull_case": [{"claim": "This stock will skyrocket", "kind": "text",
                            "citation": {"source": "x", "company": "X", "period": "FY", "snippet": "s"}}],
             "bear_case": [], "key_risks": []}
    out = confidence_guard_node({"draft_brief": brief})
    softened = out["draft_brief"]
    assert "guaranteed" not in softened["confidence_rationale"]
    assert "no risk" not in softened["confidence_rationale"]
    assert "skyrocket" not in softened["bull_case"][0]["claim"]
    # citation is untouched by softening (provenance preserved)
    assert softened["bull_case"][0]["citation"]["snippet"] == "s"


def test_confidence_guard_fail_closed_on_pervasive_overconfidence():
    # verdict == FAIL (pervasive) -> raise `overconfident` so the monitor escalates to a human, even
    # though the shown prose is also softened. A cosmetic word-swap must not silently ship.
    brief = {"confidence_rationale": "Guaranteed to rise. No risk. It will always outperform.",
             "bull_case": [], "bear_case": [], "key_risks": []}
    out = confidence_guard_node({"draft_brief": brief})
    assert out["overconfidence"]["verdict"] == "FAIL"
    assert out["overconfident"] is True


def test_confidence_guard_passthrough_when_calibrated():
    # PASS prose -> no rewrite, no fail-closed flag (behaves exactly as before).
    brief = {"confidence_rationale": "Margins expanded; growth is likely to continue. Risks remain.",
             "bull_case": [], "bear_case": [], "key_risks": []}
    out = confidence_guard_node({"draft_brief": brief})
    assert out["overconfidence"]["verdict"] == "PASS"
    assert "draft_brief" not in out          # untouched
    assert "overconfident" not in out


def test_source_guard_reopens_originating_task_on_unsupported():
    # F-07b: an unsupported claim populates `reopen` with the id of the task that produced it, so the
    # back-edge does a TARGETED repair instead of the old no-op ("all covered -> plan exhausted").
    brief = _brief_with("The company will acquire a competitor next year",
                        "Revenue and gross margin discussion for the fiscal period")
    evidence = [{"claim": "The company will acquire a competitor next year", "kind": "text",
                 "source_task": "t2",
                 "citation": {"source": "10-K", "company": "NVDA", "period": "FY2025",
                              "snippet": "Revenue and gross margin discussion for the fiscal period"}}]
    out = source_guard_node({"draft_brief": brief, "evidence": evidence})
    assert out["unsupported"] is True
    assert out["reopen"] == ["t2"]


# --- Stage 3: source_guard -------------------------------------------------------------------------
def _brief_with(claim, snippet):
    return {"bull_case": [{"claim": claim, "kind": "text",
                           "citation": {"source": "10-K", "company": "NVDA", "period": "FY2025", "snippet": snippet}}],
            "bear_case": [], "key_risks": []}


def test_source_guard_passes_supported_claim():
    brief = _brief_with("Data center revenue grew rapidly on strong GPU demand",
                        "Data center revenue grew rapidly as GPU demand increased strongly")
    out = source_guard_node({"draft_brief": brief})
    assert out["unsupported"] is False and out["groundedness"] >= 0.9


def test_claim_coverage_does_not_penalize_a_short_claim_cited_to_a_long_profile():
    claim = "NVIDIA Corporation operates in Technology and is classified as Semiconductors."
    snippet = (
        "Company: NVIDIA Corporation. Sector: Technology. Industry: Semiconductors. "
        "NVIDIA Corporation operates as a data center scale AI infrastructure company across many "
        "markets, products, customer segments, and geographic regions."
    )
    assert claim_coverage(claim, snippet) >= 0.5


def test_source_guard_routes_back_on_unsupported():
    brief = _brief_with("The company will acquire a competitor next year",
                        "Revenue and gross margin discussion for the fiscal period")
    out = source_guard_node({"draft_brief": brief})
    assert out["unsupported"] is True  # claim not supported by its snippet -> back-edge


def test_quality_repair_discards_theses_built_from_superseded_evidence():
    brief = _brief_with(
        "The company will acquire a competitor next year",
        "Revenue and gross margin discussion for the fiscal period",
    )
    state = {
        "question": "What are the risks?",
        "draft_brief": brief,
        "evidence": [{**brief["bull_case"][0], "source_task": "t2"}],
        "branches": [{"side": "bull", "thesis": "old interpretation"}],
        "scored_branches": [{"side": "bull", "thesis": "old interpretation"}],
        "depth": 3,
        "editor_rounds": 0,
        "editor_reretrieve_max": 2,
        "transcript_status": "ingested",
        "confidence": 0.7,
    }
    out = quality_gate_node(state)
    assert out["quality_decision"] == "repair"
    assert out["depth"] == 0 and out["branches"] == [] and out["scored_branches"] == []


def test_quality_gate_repairs_missing_bear_with_targeted_downside_tasks_once():
    numeric = {
        "claim": "gross margin", "kind": "number", "value": 0.75,
        "citation": {"source": "statement", "company": "NVDA", "period": "2026-07-31",
                     "snippet": "gross_margin=0.75"},
    }
    state = {
        "question": "How balanced is the investment case?",
        "draft_brief": {
            "bull_case": [numeric], "bear_case": [], "key_risks": [],
            "evidence_appendix": [numeric], "confidence": 0.6,
        },
        "evidence": [{**numeric, "source_task": "t1"}],
        "plan": [
            {"id": "t1", "kind": "text", "claim": "upside", "evidence_role": "management_upside"},
            {"id": "t2", "kind": "text", "claim": "downside", "evidence_role": "management_downside"},
            {"id": "t3", "kind": "text", "claim": "filing risks", "evidence_role": "filing_risk"},
        ],
        "confidence": 0.6,
        "transcript_status": "cached",
        "transcript_periods": ["Q2-2027", "Q1-2027", "Q4-2026", "Q3-2026"],
        "analysis_window": {"n_quarters": 4},
        "report_repair_rounds": 0,
        "report_repair_max": 1,
    }
    out = quality_gate_node(state)
    assert out["quality_decision"] == "repair"
    assert set(out["reopen"]) == {"t2", "t3"}
    assert out["report_repair_rounds"] == 1
    assert out["quality_report"]["repair_tasks"]


def test_verify_claims_counts_numbers_as_supported():
    brief = {"bull_case": [{"claim": "gross margin 40%", "kind": "number", "value": 0.4,
                            "citation": {"source": "line items", "company": "X", "period": "FY", "snippet": "gm"}}],
             "bear_case": [], "key_risks": []}
    result = verify_claims(brief)
    assert result["groundedness"] == 1.0
