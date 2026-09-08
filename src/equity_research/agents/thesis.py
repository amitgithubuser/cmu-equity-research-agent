"""Thesis Analyst + Critic graph nodes (architecture §4 agents #3/#4, §8.5).

The Analyst proposes grounded bull, bear, and risk angles. The independent Critic applies the
deterministic evidence gate, model rubric, borderline panel, and top-k pruning in one meaningful
checkpoint. `nodes_real.critic_cycle_node` then advances the bounded search depth. The Critic never
grades its own generation.
"""

from __future__ import annotations

from ..config import get_settings
from ..tot.engine import SIDES, confidence_from_gap, dedupe_branches, prune, score_branches


# --------------------------------------------------------------------------- Analyst (generation)
def analyst_node(state: dict, llm, generator=None) -> dict:
    """Generate competing angles from the evidence. `generator(state, llm) -> list[branch dict]` is
    injectable; the default builds one grounded angle per side from the gathered evidence so the
    loop runs deterministically in tests without an LLM.
    """
    settings = get_settings()
    evidence = state.get("evidence", []) or []
    if generator is not None:
        branches = generator(state, llm)
    else:
        branches = _default_angles(evidence, n=settings.tot_root_branches)
    depth = state.get("depth", 0)
    return {"branches": branches, "tot_iterations": state.get("tot_iterations", 0) + 1,
            "trajectory": [{
                "action": "thesis_analyst",
                "args": {"depth": depth, "evidence_count": len(evidence)},
                "observation": f"generated {len(branches)} thesis branches",
                "status": "ok",
            }],
            "log": [f"analyst: {len(branches)} angles at depth {depth}"]}


def _default_angles(evidence: list[dict], n: int) -> list[dict]:
    """A deterministic angle generator: partition evidence into bull/bear/risk buckets by simple cues.

    Real runs replace this with an LLM `generator`. This keeps the ToT controller exercisable offline.
    """
    def make(side: str, claims: list[dict]) -> dict:
        return {"side": side, "claims": claims}

    bull, bear, risk = [], [], []
    for e in evidence:
        text = (e.get("claim", "") + " " + str(e.get("citation", {}).get("snippet", ""))).lower()
        if any(w in text for w in ("risk", "concentration", "dependence", "lawsuit", "decline")):
            risk.append(e)
        elif any(w in text for w in ("grew", "growth", "expanded", "rose", "strong", "record")):
            bull.append(e)
        else:
            bear.append(e)
    angles = [make("bull", bull), make("bear", bear), make("risk", risk)]
    return [a for a in angles if a["claims"]][:n]


# --------------------------------------------------------------------------- Critic (evaluation)
def critic_node(state: dict, llm, scorer=None, batch_scorer=None) -> dict:
    """Independently score branches (gate -> rubric -> panel-if-borderline) and prune to top-k/side."""
    settings = get_settings()
    branches = state.get("branches", []) or []
    scored = score_branches(
        branches, llm, settings, scorer=scorer, batch_scorer=batch_scorer
    )
    kept, _insufficient = prune(scored, settings)

    # accumulate survivors across BFS levels on the blackboard
    prior = state.get("scored_branches", []) or []
    merged = _merge_keep_best(prior + kept, settings.tot_keep_per_side)
    final_insufficient = {
        side: not any(branch.get("side") == side for branch in merged)
        for side in ("bull", "bear")
    }
    conf = confidence_from_gap(merged, final_insufficient)
    return {"scored_branches": merged, "confidence": conf,
            "insufficient": final_insufficient,
            "trajectory": [{
                "action": "critic",
                "args": {"candidate_count": len(branches)},
                "observation": f"kept {len(kept)} branches; confidence={conf}",
                "status": "ok" if kept else "refused",
            }],
            "log": [f"critic: kept {len(kept)} (conf={conf}, insufficient={final_insufficient})"]}


def _merge_keep_best(branches: list[dict], keep_per_side: int) -> list[dict]:
    out = []
    for side in SIDES:
        side_b = sorted([b for b in branches if b.get("side") == side],
                        key=lambda b: b.get("support_score", 0.0), reverse=True)
        out.extend(_dedupe_keep(side_b)[:keep_per_side])
    return out


def _dedupe_keep(branches: list[dict]) -> list[dict]:
    return dedupe_branches(branches)
