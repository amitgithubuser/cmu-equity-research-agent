"""Phase 6 gate (implementation-plan §7.3): BFS bounded; gate rejects uncited; a starved side reports
'insufficient evidence' (never invented); confidence derives from the bull/bear support gap.
"""

from equity_research.agents.thesis import analyst_node
from equity_research.config import get_settings
from equity_research.llm import FakeLLM
from equity_research.nodes_real import critic_cycle_node
from equity_research.schemas import (
    AnalystAngles,
    Angle,
    AngleClaim,
    CriticAssessment,
    CriticAssessments,
    CriticScore,
)
from equity_research.tot import (
    confidence_from_gap,
    evidence_gate,
    llm_batch_scorer,
    llm_generator,
    llm_scorer,
    prune,
    score_branches,
)


def _cited_claim(text="rev grew", side_word="grew"):
    return {"claim": f"{text} {side_word}", "kind": "text",
            "citation": {"source": "10-K", "company": "X", "period": "FY2025", "snippet": text}}


def _uncited_claim(text="rev grew"):
    return {"claim": text, "kind": "text"}  # no citation, no value


# --- deterministic gate ----------------------------------------------------------------------------
def test_gate_rejects_uncited_branch():
    assert evidence_gate({"side": "bull", "claims": [_uncited_claim()]}) is False


def test_gate_accepts_cited_branch():
    assert evidence_gate({"side": "bull", "claims": [_cited_claim()]}) is True


def test_gate_accepts_numeric_claim_without_citation_text():
    numeric = {"claim": "gm 40%", "kind": "number", "value": 0.4,
               "citation": {"source": "line items", "company": "X", "period": "FY", "snippet": "gm"}}
    assert evidence_gate({"side": "bull", "claims": [numeric]}) is True


def test_gate_empty_branch_fails():
    assert evidence_gate({"side": "bull", "claims": []}) is False


# --- scoring + pruning -----------------------------------------------------------------------------
def test_prune_keeps_topk_per_side():
    s = get_settings()
    branches = [{"side": "bull", "claims": [_cited_claim(f"c{i}", "grew")]} for i in range(5)]
    scored = score_branches(branches, FakeLLM(), s)  # all cited -> survive
    kept, _ = prune(scored, s)
    assert len([b for b in kept if b["side"] == "bull"]) <= s.tot_keep_per_side


def test_prune_drops_lower_scored_same_side_topic_overlap():
    s = get_settings()
    branches = [
        {"side": "bull", "thesis": "Gross margin expansion lifts profitability",
         "claims": [_cited_claim("gross margin expanded", "strong")]},
        {"side": "bull", "thesis": "Operating margin expansion supports earnings",
         "claims": [_cited_claim("operating margin expanded", "strong")]},
        {"side": "bull", "thesis": "Revenue growth reflects stronger demand",
         "claims": [_cited_claim("revenue growth accelerated", "strong")]},
    ]
    scores = iter([0.91, 0.82, 0.79])
    scored = score_branches(branches, FakeLLM(), s, scorer=lambda _branch, _llm: next(scores))
    kept, _ = prune(scored, s)
    theses = [branch["thesis"] for branch in kept]
    assert "Gross margin expansion lifts profitability" in theses
    assert "Revenue growth reflects stronger demand" in theses
    assert "Operating margin expansion supports earnings" not in theses


def test_all_fail_marks_insufficient_evidence():
    s = get_settings()
    # bull side has only uncited claims -> all fail the gate -> insufficient, NOT invented
    branches = [{"side": "bull", "claims": [_uncited_claim()]},
                {"side": "bear", "claims": [_cited_claim("margin fell", "fell")]}]
    scored = score_branches(branches, FakeLLM(), s)
    kept, insufficient = prune(scored, s)
    assert insufficient["bull"] is True
    assert all(b["side"] != "bull" for b in kept)  # no fabricated bull branch survives


# --- confidence from gap ---------------------------------------------------------------------------
def test_confidence_low_when_side_insufficient():
    assert confidence_from_gap([], {"bull": True, "bear": False}) == 0.3


def test_confidence_low_on_narrow_gap_vs_high_on_wide_gap():
    narrow = confidence_from_gap(
        [{"side": "bull", "support_score": 0.8}, {"side": "bear", "support_score": 0.78}],
        {"bull": False, "bear": False})
    wide = confidence_from_gap(
        [{"side": "bull", "support_score": 0.9}, {"side": "bear", "support_score": 0.2}],
        {"bull": False, "bear": False})
    assert wide > narrow  # a decisive, lopsided picture is more confident than a near-tie


# --- BFS bounded by depth (through the nodes) ------------------------------------------------------
def test_bfs_bounded_by_depth():
    s = get_settings()
    state = {"depth": 0, "evidence": [_cited_claim("rev grew", "grew")]}
    for _ in range(s.tot_max_depth + 3):  # spin the loop more than the cap
        state = {**state, **analyst_node(state, FakeLLM())}
        state = {**state, **critic_cycle_node(state, FakeLLM())}
        if state["depth"] >= s.tot_max_depth:
            break
    assert state["depth"] <= s.tot_max_depth


def test_critic_never_scores_uncited_above_zero():
    s = get_settings()
    scored = score_branches([{"side": "bull", "claims": [_uncited_claim()]}], FakeLLM(), s)
    assert scored[0]["support_score"] == 0.0 and scored[0]["gate_failed"] is True


# --- F-03: a NEVER-GENERATED side is insufficient, not silently decisive ---------------------------
def test_never_generated_side_is_insufficient():
    # The F-03 bug: if the Analyst produced ONLY bull branches (no bear at all), `prune` skipped the
    # missing side, so `insufficient["bear"]` was never set and the low-confidence floor didn't fire.
    s = get_settings()
    scored = score_branches([{"side": "bull", "claims": [_cited_claim("rev grew", "grew")]}], FakeLLM(), s)
    _kept, insufficient = prune(scored, s)
    assert insufficient["bear"] is True          # bear was never argued -> insufficient
    assert insufficient["bull"] is False


def test_one_sided_thesis_is_floored_to_low_confidence():
    # End-to-end of F-03: a well-supported bull with no bear must NOT read as a decisive, high-confidence
    # call. A missing side means we couldn't balance the case -> 0.3 floor (which can trigger HITL).
    s = get_settings()
    scored = score_branches([{"side": "bull", "claims": [_cited_claim("rev grew", "grew")]}], FakeLLM(), s)
    kept, insufficient = prune(scored, s)
    assert confidence_from_gap(kept, insufficient) == 0.3


# --- F-02: real LLM Analyst generator + Critic scorer ----------------------------------------------
def _evidence_pair():
    return [
        {"claim": "revenue grew", "kind": "text",
         "citation": {"source": "10-K", "company": "X", "period": "FY2025", "snippet": "grew"}},
        {"claim": "margin fell", "kind": "text",
         "citation": {"source": "10-K", "company": "X", "period": "FY2025", "snippet": "fell"}},
    ]


def test_llm_generator_grounds_claims_in_existing_evidence():
    # The Analyst references evidence BY INDEX; llm_generator resolves those to the real cited dicts.
    ev = _evidence_pair()
    llm = FakeLLM(structured_handler=lambda p, m: AnalystAngles(angles=[
        Angle(side="bull", thesis="growth", claims=[AngleClaim(evidence_index=0)]),
        Angle(side="bear", thesis="margin pressure", claims=[AngleClaim(evidence_index=1)]),
    ]))
    branches = llm_generator({"ticker": "X", "evidence": ev, "depth": 0}, llm)
    sides = {b["side"] for b in branches}
    assert sides == {"bull", "bear"}
    # each angle claim is the ACTUAL cited evidence dict, not a fresh fabrication
    assert branches[0]["claims"][0] is ev[0]


def test_llm_generator_preserves_rich_case_details_and_rationales():
    ev = _evidence_pair()
    llm = FakeLLM(structured_handler=lambda p, m: AnalystAngles(angles=[
        Angle(
            side="bear", thesis="margin pressure may persist",
            mechanism="lower margin would constrain earnings growth",
            time_horizon="next four reported quarters",
            catalysts=["a further reported margin decline"],
            watch_items=["gross margin"],
            invalidation_conditions=["reported margin recovery"],
            claims=[AngleClaim(evidence_index=1, rationale="the cited decline is the premise")],
        ),
    ]))
    branch = llm_generator({"ticker": "X", "evidence": ev, "depth": 0}, llm)[0]
    assert branch["mechanism"].startswith("lower margin")
    assert branch["watch_items"] == ["gross margin"]
    assert branch["evidence_rationales"] == ["the cited decline is the premise"]


def test_llm_generator_removes_internal_numeric_reference_markers_from_public_prose():
    ev = _evidence_pair()
    llm = FakeLLM(structured_handler=lambda p, m: AnalystAngles(angles=[
        Angle(
            side="bull", thesis="growth is supported [0]",
            mechanism="revenue growth [0] may offset margin pressure [1]",
            catalysts=["continued growth [0]"],
            claims=[AngleClaim(evidence_index=0, rationale="direct growth evidence [0]")],
        )
    ]))
    branch = llm_generator({"ticker": "X", "evidence": ev, "depth": 0}, llm)[0]
    assert branch["thesis"] == "growth is supported"
    assert branch["mechanism"] == "revenue growth may offset margin pressure"
    assert branch["catalysts"] == ["continued growth"]
    assert branch["evidence_rationales"] == ["direct growth evidence"]


def test_llm_generator_drops_offsource_reference():
    # An index the model invents (out of range) must be dropped; an angle left with no grounded claim
    # is not emitted at all -> a hallucinated reference can never become an uncited thesis.
    ev = _evidence_pair()
    llm = FakeLLM(structured_handler=lambda p, m: AnalystAngles(angles=[
        Angle(side="bull", thesis="real", claims=[AngleClaim(evidence_index=0)]),
        Angle(side="bear", thesis="hallucinated", claims=[AngleClaim(evidence_index=99)]),
    ]))
    branches = llm_generator({"ticker": "X", "evidence": ev, "depth": 0}, llm)
    assert [b["side"] for b in branches] == ["bull"]   # the off-source bear angle was dropped


def test_llm_generator_expands_survivors_at_depth():
    # True BFS depth (F-02): at depth>0 the prompt tells the Analyst to EXPAND survivors, and the
    # surviving branches are included so it grows new sub-claims rather than regenerating from scratch.
    ev = _evidence_pair()
    captured = {}

    def handler(prompt, model_cls):
        captured["prompt"] = prompt
        return AnalystAngles(angles=[Angle(side="bull", thesis="deeper", claims=[AngleClaim(evidence_index=0)])])

    llm = FakeLLM(structured_handler=handler)
    state = {"ticker": "X", "evidence": ev, "depth": 1,
             "scored_branches": [{"side": "bull", "thesis": "growth", "support_score": 0.8}]}
    llm_generator(state, llm)
    assert "depth 1" in captured["prompt"].lower()
    assert "growth" in captured["prompt"]              # the survivor is shown so it can be expanded


def test_llm_scorer_is_independent_bounded_call():
    # The Critic is a SEPARATE structured call returning a bounded score (F-02); rubric_score uses it.
    captured = {}
    branch = {"side": "bull", "thesis": "growth", "claims": [_cited_claim("revenue grew")]}

    def handler(prompt, model_cls):
        captured["prompt"] = prompt
        return CriticScore(support_score=0.82)

    llm = FakeLLM(structured_handler=handler)
    assert llm_scorer(branch, llm) == 0.82
    assert "10-K | X | FY2025" in captured["prompt"]
    assert "<evidence>revenue grew</evidence>" in captured["prompt"]


def test_score_branches_uses_injected_llm_scorer():
    # Wiring check: when score_branches is given scorer=llm_scorer, a gated branch is scored by the
    # Critic call (not the deterministic fraction-cited proxy).
    s = get_settings()
    branches = [{"side": "bull", "claims": [_cited_claim("rev grew", "grew")]}]
    llm = FakeLLM(structured_handler=lambda p, m: CriticScore(support_score=0.91))
    scored = score_branches(branches, llm, s, scorer=llm_scorer)
    assert scored[0]["support_score"] == 0.91 and scored[0]["survived"] is True


def test_batch_critic_scores_all_branches_in_one_structured_call():
    branches = [
        {"side": "bull", "thesis": "growth", "claims": [_cited_claim("revenue grew")]},
        {"side": "bear", "thesis": "risk", "claims": [_cited_claim("supply is constrained")]},
    ]

    def handler(prompt, model_cls):
        assert model_cls is CriticAssessments
        assert "BRANCH 0" in prompt and "BRANCH 1" in prompt
        return CriticAssessments(assessments=[
            CriticAssessment(branch_index=0, evidence_support=0.9, consistency=0.8,
                             materiality=0.7, survival=0.6),
            CriticAssessment(branch_index=1, evidence_support=0.8, consistency=0.8,
                             materiality=0.8, survival=0.8),
        ])

    llm = FakeLLM(structured_handler=handler)
    assert llm_batch_scorer(branches, llm) == [0.75, 0.8]
    assert len(llm.calls) == 1


def test_score_branches_batch_path_fails_missing_assessment_closed():
    s = get_settings()
    branches = [
        {"side": "bull", "claims": [_cited_claim("revenue grew")]},
        {"side": "bear", "claims": [_cited_claim("cost increased")]},
    ]
    scored = score_branches(
        branches,
        FakeLLM(),
        s,
        batch_scorer=lambda batch, llm: [0.85],
    )
    assert scored[0]["survived"] is True
    assert scored[1]["support_score"] == 0.0 and scored[1]["survived"] is False
