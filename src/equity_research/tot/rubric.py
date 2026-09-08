"""ToT evaluation — deterministic gate + Critic rubric scaffold (architecture §8.3).

Two-stage evaluation, in order (the design's biggest reliability lever):
  1. `evidence_gate` — DETERMINISTIC. A branch with any uncited/unsupported claim fails automatically.
     No LLM opinion can rescue a claim with no source. Runs first, cheaply.
  2. `rubric_score` — the independent **Critic** scores only the survivors on the 4-criteria rubric
     (support / consistency / materiality / survival). The Critic never grades its own generation.

`critic_panel` runs a 3-vote panel ONLY for borderline branches (within ±margin of the pass score),
per OQ-5 — spending extra votes only where the keep/cut decision is genuinely close.
"""

from __future__ import annotations

from statistics import median


def evidence_gate(branch: dict) -> bool:
    """Deterministic pre-filter: every claim must have a citation OR a computed number (§8.3).

    A branch with no claims fails (nothing to support). Mirrors the schema: a `number` claim carries
    a value; a `text` claim carries a citation.
    """
    claims = branch.get("claims", [])
    if not claims:
        return False
    for c in claims:
        has_citation = bool(c.get("citation"))
        has_number = c.get("value") is not None
        if not (has_citation or has_number):
            return False
    return True


def rubric_score(branch: dict, critic_llm, scorer=None) -> float:
    """Score one gated branch 0..1 on the rubric via the independent Critic.

    `scorer(branch, critic_llm) -> float` is the pluggable scoring fn (a structured LLM call on a
    real run; injected in tests). Defaults to a conservative deterministic proxy so the engine is
    testable without an LLM: mean support presence across claims.
    """
    if scorer is not None:
        return _clamp(scorer(branch, critic_llm))
    # deterministic fallback proxy: fraction of claims that are cited/numeric, lightly rewarding depth
    claims = branch.get("claims", [])
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.get("citation") or c.get("value") is not None)
    base = supported / len(claims)
    return _clamp(base)


def critic_panel(branch: dict, critic_llm, scorer=None, votes: int = 3) -> float:
    """Median of `votes` independent Critic scores — used for borderline branches (steadier)."""
    scores = [rubric_score(branch, critic_llm, scorer=scorer) for _ in range(votes)]
    return _clamp(median(scores))


def is_borderline(score: float, pass_score: float, margin: float) -> bool:
    """True if a branch scores within ±margin of the pass threshold (OQ-5 panel trigger)."""
    return abs(score - pass_score) <= margin


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
