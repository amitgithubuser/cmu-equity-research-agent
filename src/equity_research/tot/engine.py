"""Tree-of-Thought engine — BFS with per-level pruning (architecture §8, CP 4.1).

Why BFS, not DFS: BFS generates all angles first, scores them, keeps the top few, THEN expands one
level deeper — so bull and bear are held **side by side**. DFS would take one branch to the bottom
before looking at the other, which is exactly the *premature commitment* ToT is meant to avoid. The
usual "BFS explores everything" cost objection is handled by **pruning before each new level**.

This module is the deterministic controller (expand → gate → score → prune → check depth). The
Analyst (generation) and Critic (scoring) LLM calls are injected, keeping the loop testable and the
Critic structurally independent of the Analyst.
"""

from __future__ import annotations

from .rubric import critic_panel, evidence_gate, is_borderline, rubric_score

SIDES = ("bull", "bear", "risk")


def score_branches(
    branches: list[dict],
    critic_llm,
    settings,
    scorer=None,
    batch_scorer=None,
) -> list[dict]:
    """Gate then score each branch. Gate failures get score 0 and `gate_failed=True` (never invented).

    Borderline survivors (within ±critic_panel_margin of the pass score) are re-scored by a 3-vote
    panel for a steadier keep/cut decision (OQ-5).
    """
    gated = [dict(branch) for branch in branches if evidence_gate(branch)]
    batch_scores = batch_scorer(gated, critic_llm) if batch_scorer and gated else []
    gated_index = 0
    scored = []
    for b in branches:
        branch = dict(b)
        if not evidence_gate(branch):
            branch.update(support_score=0.0, survived=False, gate_failed=True)
            scored.append(branch)
            continue
        if batch_scorer:
            s = batch_scores[gated_index] if gated_index < len(batch_scores) else 0.0
            gated_index += 1
        else:
            s = rubric_score(branch, critic_llm, scorer=scorer)
        if not batch_scorer and is_borderline(
            s, settings.critic_pass_score, settings.critic_panel_margin
        ):
            s = critic_panel(branch, critic_llm, scorer=scorer)
        branch.update(support_score=s, survived=s >= settings.critic_pass_score, gate_failed=False)
        scored.append(branch)
    return scored


def prune(scored: list[dict], settings) -> tuple[list[dict], dict[str, bool]]:
    """Keep top-k survivors per side; drop gate-fails, below-threshold, and duplicates (§8.3).

    Returns `(kept, insufficient)` where `insufficient[side]` is True when EVERY branch on that side
    failed — that side is reported as 'insufficient evidence', never invented (CP 3.1 rule).
    """
    kept: list[dict] = []
    insufficient: dict[str, bool] = {}
    for side in SIDES:
        side_branches = [b for b in scored if b.get("side") == side]
        survivors = [b for b in side_branches if b.get("survived") and not b.get("gate_failed")]
        survivors.sort(key=lambda b: b.get("support_score", 0.0), reverse=True)
        survivors = dedupe_branches(survivors)
        top = survivors[: settings.tot_keep_per_side]
        kept.extend(top)
        # F-03: `insufficient` is set for bull/bear whether or not the Analyst generated that side.
        # A side with NO branch at all (never argued) is just as insufficient as a side whose every
        # branch was starved out — both mean "we can't support this side from the evidence". The old
        # `if not side_branches: continue` silently dropped the never-generated case, so a one-sided
        # thesis skipped the low-confidence floor in `confidence_from_gap` and read as decisive.
        if side in ("bull", "bear"):
            insufficient[side] = len(top) == 0
    return kept, insufficient


_TOPIC_TERMS = {
    "margins": ("margin", "profitability", "gross profit", "operating income", "earnings"),
    "growth": ("growth", "revenue", "sales", "demand", "traffic", "expansion"),
    "cash_balance": ("cash flow", "liquidity", "debt", "leverage", "balance sheet", "buyback"),
    "supply_execution": ("supply", "inventory", "capacity", "execution", "product transition"),
    "competition": ("competition", "competitor", "market share", "pricing", "positioning"),
    "regulation": ("regulation", "regulatory", "export", "litigation", "tax", "compliance"),
}


def branch_topic(branch: dict) -> str:
    """Return the dominant decision topic for deterministic same-side overlap control."""
    narrative = " ".join((str(branch.get("thesis", "")), str(branch.get("mechanism", "")))).lower()
    evidence = " ".join(
        str(claim.get("claim", "")) for claim in (branch.get("claims", []) or [])
    ).lower()
    scores = {
        topic: 3 * sum(narrative.count(term) for term in terms)
        + sum(evidence.count(term) for term in terms)
        for topic, terms in _TOPIC_TERMS.items()
    }
    topic, score = max(scores.items(), key=lambda item: item[1])
    return topic if score else ""


def dedupe_branches(branches: list[dict]) -> list[dict]:
    """Drop exact duplicates and lower-ranked same-side branches about the same main topic."""
    seen_claims: set[frozenset] = set()
    seen_topics: set[tuple[str, str]] = set()
    out = []
    for branch in branches:
        key = frozenset(c.get("claim", "") for c in branch.get("claims", []))
        topic = branch_topic(branch)
        topic_key = (str(branch.get("side", "")), topic)
        if key in seen_claims or (topic and topic_key in seen_topics):
            continue
        seen_claims.add(key)
        if topic:
            seen_topics.add(topic_key)
        out.append(branch)
    return out


def confidence_from_gap(kept: list[dict], insufficient: dict[str, bool] | None = None) -> float:
    """Confidence from the gap between how well each side is supported (§8.4).

    - If either side is insufficient, confidence is low (we can't balance an unsupported side).
    - Otherwise confidence rises with the *gap* between the stronger and weaker side: a lopsided,
      well-supported picture is more decisive; a near-tie is genuinely uncertain -> low confidence
      (which can trigger HITL escalation downstream).
    """
    insufficient = insufficient or {}
    if insufficient.get("bull") or insufficient.get("bear"):
        return 0.3

    def best(side: str) -> float:
        scores = [b.get("support_score", 0.0) for b in kept if b.get("side") == side]
        return max(scores) if scores else 0.0

    bull, bear = best("bull"), best("bear")
    if bull == 0.0 and bear == 0.0:
        return 0.3
    gap = abs(bull - bear)             # how decisively one side outweighs the other
    strength = (bull + bear) / 2.0     # overall evidential strength of the picture
    # Monotonic and defensible: confidence grows with BOTH the gap (decisiveness) and the strength.
    # A well-supported near-tie stays moderate (and can trigger HITL); a lopsided, well-supported
    # picture is high; a weakly-supported picture is low regardless of the gap.
    return round(min(1.0, strength * (0.5 + gap)), 4)
