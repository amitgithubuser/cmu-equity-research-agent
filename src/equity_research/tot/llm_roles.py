"""Real LLM Analyst (generator) + Critic (scorer) for the ToT loop (fix F-02).

The ToT engine is a deterministic controller: it expands, gates, scores, prunes, checks depth. The
two *judgement* steps inside it — GENERATING angles and SCORING them — are LLM calls, and they are
INJECTED so the controller stays testable offline. Before this module they were injected as `None`,
so a "real" run silently fell back to the deterministic proxies (`_default_angles` keyword-bucketing +
fraction-of-claims-cited scoring) and never actually asked the reasoning model to think. This module
supplies the real functions `build_real_nodes` now wires in.

Two design rules the course cares about are enforced here:

  * GROUNDING (CP 3.1): the Analyst does not mint facts. It selects evidence already on the blackboard
    by index; each angle claim resolves back to a real, cited Evidence dict. Anything the model
    references off-source (index out of range) is dropped, and the evidence gate then rejects any
    angle left with no supported claims — so an LLM hallucination cannot survive into a thesis.
  * INDEPENDENCE (§4): the Critic is a SEPARATE call with a SEPARATE prompt from the Analyst. It never
    sees "this is your own thesis"; it scores a branch cold. `llm_scorer` is what `rubric_score`
    invokes as its `scorer=`, so the Critic path is structurally distinct from generation.

True BFS depth (also F-02): at depth 0 the Analyst proposes root angles across sides; at depth>0 it is
asked to EXPAND the surviving branches into deeper, more specific sub-claims off the same evidence —
new thoughts grown from survivors, not a fresh regeneration. That is what makes the loop a tree.
"""

from __future__ import annotations

import re

from ..agents.prompts import ANALYST_PROMPT, CRITIC_BATCH_PROMPT, CRITIC_PROMPT
from ..schemas import AnalystAngles, CriticAssessments, CriticScore
from ..text_cleaning import clean_public_text

# extra guidance appended when we are expanding survivors rather than seeding roots (depth > 0).
_EXPAND_SUFFIX = """

You are now at depth {depth}. These angles SURVIVED the previous round:
{survivors}

Do not repeat them. EXPAND them: for each surviving side, propose a deeper, more specific sub-claim
that sharpens the case, grounded in the SAME evidence (reference it by index). If an angle cannot be
deepened with the available evidence, omit it rather than padding."""

_RAW_EVIDENCE_INDEX = re.compile(r"\s*\[(?:\d+(?:\s*,\s*\d+)*)\]")


def _clean_model_prose(text: str) -> str:
    """Remove Analyst-only numeric index markers before prose reaches the public report."""
    return clean_public_text(_RAW_EVIDENCE_INDEX.sub("", str(text or "")))


def _evidence_block(evidence: list[dict], plan: list[dict] | None = None) -> str:
    """Show the Analyst the cited fact, value, passage, and intended role for each item."""
    roles = {
        task.get("id"): task.get("evidence_role", "question_specific")
        for task in (plan or []) if task.get("id")
    }
    lines = []
    for i, e in enumerate(evidence):
        cit = e.get("citation", {}) or {}
        src = f"{cit.get('source', '?')} ({cit.get('company', '?')} {cit.get('period', '?')})"
        role = roles.get(e.get("source_task"), "question_specific")
        value = f" | value={e.get('value')}" if e.get("value") is not None else ""
        snippet = str(cit.get("snippet", ""))[:600]
        lines.append(
            f"[{i}] role={role} | kind={e.get('kind', 'text')}{value}\n"
            f"claim: {e.get('claim', '')}\nsource: {src}\n"
            f"<evidence>{snippet}</evidence>"
        )
    return "\n".join(lines) if lines else "(no evidence gathered)"


def _survivor_block(scored_branches: list[dict]) -> str:
    if not scored_branches:
        return "(none yet)"
    return "\n".join(
        f"- [{b.get('side')}] {b.get('thesis') or _first_claim_text(b)} "
        f"(score={b.get('support_score', 0.0)})"
        for b in scored_branches
    )


def _first_claim_text(branch: dict) -> str:
    claims = branch.get("claims", []) or []
    return claims[0].get("claim", "") if claims else ""


def _resolve_claims(angle, evidence: list[dict]) -> list[dict]:
    """Turn an `Angle`'s index references back into the real, cited Evidence dicts (grounding).

    Out-of-range / negative indices are dropped — the model is not allowed to invent evidence, and a
    dangling reference must not become an uncited claim that sneaks past the gate.
    """
    out: list[dict] = []
    for ac in angle.claims:
        idx = ac.evidence_index
        if 0 <= idx < len(evidence):
            out.append(evidence[idx])
    return out


def llm_generator(state: dict, llm) -> list[dict]:
    """Real Analyst: ask the reasoning model for grounded angles; return engine-shaped branch dicts.

    Injected as `generator=` on `analyst_node`. Each returned branch is
    `{"side", "thesis", "claims": [<real cited Evidence dict>, ...]}`. Angles that resolve to zero
    real claims are dropped here (and would fail the evidence gate anyway).
    """
    from ..config import get_settings

    settings = get_settings()
    evidence = state.get("evidence", []) or []
    depth = state.get("depth", 0)
    n = settings.tot_root_branches

    prompt = ANALYST_PROMPT.format(
        ticker=state.get("ticker", ""),
        evidence=_evidence_block(evidence, state.get("plan", []) or []), n=n,
    )
    if depth > 0:
        prompt += _EXPAND_SUFFIX.format(
            depth=depth, survivors=_survivor_block(state.get("scored_branches", []) or []),
        )

    result = llm.with_structured_output(AnalystAngles).invoke(prompt)
    branches: list[dict] = []
    for angle in result.angles:
        claims = _resolve_claims(angle, evidence)
        if not claims:
            continue  # no grounded claim -> not a thesis; drop before it reaches the gate
        rationales = [
            ac.rationale for ac in angle.claims
            if 0 <= ac.evidence_index < len(evidence)
        ]
        branches.append({
            "side": angle.side,
            "thesis": _clean_model_prose(angle.thesis),
            "claims": claims,
            "mechanism": _clean_model_prose(angle.mechanism),
            "time_horizon": _clean_model_prose(angle.time_horizon),
            "catalysts": [_clean_model_prose(item) for item in angle.catalysts],
            "watch_items": [_clean_model_prose(item) for item in angle.watch_items],
            "invalidation_conditions": [
                _clean_model_prose(item) for item in angle.invalidation_conditions
            ],
            "evidence_rationales": [_clean_model_prose(item) for item in rationales],
        })
    return branches


def llm_scorer(branch: dict, llm) -> float:
    """Real Critic: independently score one gated branch 0..1. Injected as `scorer=` on `rubric_score`.

    Kept deliberately narrow — one branch, one bounded score — so the panel (borderline branches get
    3 votes) just calls this repeatedly. Grounding is already guaranteed by the gate upstream; the
    Critic judges *quality* (support strength, consistency, materiality, survival), not existence.
    """
    claim_lines = "\n".join(_critic_claim_line(c) for c in branch.get("claims", []) or [])
    branch_desc = _branch_description(branch, claim_lines)
    prompt = CRITIC_PROMPT.format(branches=branch_desc)
    result = llm.with_structured_output(CriticScore).invoke(prompt)
    return float(result.support_score)


def llm_batch_scorer(branches: list[dict], llm) -> list[float]:
    """Score all gated branches in one independent Critic call.

    Each branch still receives the four CP 4.1 rubric judgments. Batching removes the live system's
    largest latency multiplier without merging generation and evaluation or delegating arithmetic to
    the model. Missing/duplicate indices fail closed to 0.0.
    """
    indexed = []
    for index, branch in enumerate(branches):
        claim_lines = "\n".join(_critic_claim_line(c) for c in branch.get("claims", []) or [])
        indexed.append(f"BRANCH {index}\n{_branch_description(branch, claim_lines)}")
    prompt = CRITIC_BATCH_PROMPT.format(branches="\n\n".join(indexed))
    result = llm.with_structured_output(CriticAssessments).invoke(prompt)
    by_index = {}
    for assessment in result.assessments:
        if assessment.branch_index not in by_index:
            by_index[assessment.branch_index] = assessment.support_score
    return [float(by_index.get(index, 0.0)) for index in range(len(branches))]


def _critic_claim_line(claim: dict) -> str:
    """Show the independent Critic the evidence it is being asked to judge.

    Previously the Critic saw only the rewritten claim text while its rubric asked whether that claim
    was supported by a citation or computed number. That made a well-grounded branch look unsupported
    to the model and could prune every bull/bear branch. The citation remains untrusted evidence, not
    an instruction, and is length-bounded before entering the prompt.
    """
    citation = claim.get("citation", {}) or {}
    source = str(citation.get("source", ""))[:240]
    company = str(citation.get("company", ""))[:40]
    period = str(citation.get("period", ""))[:80]
    snippet = str(citation.get("snippet", ""))[:600]
    value = claim.get("value")
    value_line = f"\n  computed value: {value}" if value is not None else ""
    return (
        f"- ({claim.get('kind', 'text')}) {claim.get('claim', '')}{value_line}\n"
        f"  citation: {source} | {company} | {period}\n"
        f"  supporting evidence: <evidence>{snippet}</evidence>"
    )


def _branch_description(branch: dict, claim_lines: str) -> str:
    """Render every analytical field for the independent Critic."""
    def joined(name: str) -> str:
        return "; ".join(str(item) for item in (branch.get(name, []) or [])) or "(not stated)"

    return (
        f"[{branch.get('side')}] {branch.get('thesis', '')}\n"
        f"mechanism: {branch.get('mechanism', '') or '(not stated)'}\n"
        f"time horizon: {branch.get('time_horizon', '') or '(not stated)'}\n"
        f"catalysts: {joined('catalysts')}\n"
        f"watch items: {joined('watch_items')}\n"
        f"invalidation conditions: {joined('invalidation_conditions')}\n"
        f"claim rationales: {joined('evidence_rationales')}\n"
        f"cited claims:\n{claim_lines}"
    )
