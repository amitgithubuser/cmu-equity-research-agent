"""Stage 3 — post-generation `source_guard` (architecture §9.3, Lab 6.1
`verify_claims_against_sources`).

For each claim, measure how well its own citation snippet supports it. Lab 6.1 used Jaccard as a
fast baseline; here we upgrade to **embedding similarity** (the project embedder — real HF model on a
live run, deterministic hashing embedder offline), with a Jaccard fallback. A claim below the support
floor is **unsupported**; the graph routes back to the Researcher (bounded) or the claim is dropped.

Groundedness = supported claims / total claims; the CI gate requires ≥ `groundedness_target`.
"""

from __future__ import annotations

import re

import numpy as np

from ..config import get_settings
from ..rag.embeddings import get_embeddings


def jaccard(a: str, b: str) -> float:
    """Token-overlap baseline (Lab 6.1)."""
    ta = {w.lower().strip(".,!?:;\"'()") for w in a.split()}
    tb = {w.lower().strip(".,!?:;\"'()") for w in b.split()}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def embedding_similarity(a: str, b: str, embeddings=None) -> float:
    """Cosine similarity between two texts using the project embedder (normalized vectors)."""
    embeddings = embeddings or get_embeddings()
    va = np.array(embeddings.embed_query(a), dtype=np.float32)
    vb = np.array(embeddings.embed_query(b), dtype=np.float32)
    return float(va @ vb)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def claim_coverage(claim: str, snippet: str) -> float:
    """Fraction of meaningful claim tokens present in its source snippet.

    Jaccard unfairly penalizes a short, accurate claim cited to a long passage because every extra
    snippet word enlarges the union. Claim-side coverage asks the right directional question: how much
    of what the report asserted is visible in the evidence?
    """
    def tokenize(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if token not in _STOPWORDS and len(token) > 1
        }
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & tokenize(snippet)) / len(claim_tokens)


def support_score(claim: str, snippet: str, embeddings=None) -> float:
    """Best semantic/lexical support signal, including directional claim-token coverage."""
    return max(
        embedding_similarity(claim, snippet, embeddings),
        jaccard(claim, snippet),
        claim_coverage(claim, snippet),
    )


def verify_claims(brief: dict, embeddings=None, floor: float | None = None) -> dict:
    """Check every claim against its citation snippet. Returns support stats + the unsupported list."""
    s = get_settings()
    floor = s.source_support_floor if floor is None else floor
    embeddings = embeddings or get_embeddings()

    claims = []
    for group in (
        "bull_case", "bear_case", "key_risks", "recent_developments", "management_commentary"
    ):
        for ev in brief.get(group, []) or []:
            claims.append((group, ev))
    if brief.get("company_overview"):
        claims.append(("company_overview", brief["company_overview"]))

    if not claims:
        return {"groundedness": 1.0, "unsupported": [], "n_claims": 0, "supported": 0}

    unsupported = []
    supported = 0
    for group, ev in claims:
        snippet = (ev.get("citation") or {}).get("snippet", "")
        # numbers are backed by a computed value + provenance snippet -> count as supported
        if ev.get("kind") == "number" and ev.get("value") is not None:
            supported += 1
            continue
        score = support_score(ev.get("claim", ""), snippet, embeddings)
        if score >= floor:
            supported += 1
        else:
            # carry the snippet too: it maps the claim back to its source task (F-07b) even when the
            # confidence_guard softened the claim TEXT (snippets are never rewritten by softening).
            unsupported.append({"group": group, "claim": ev.get("claim", ""),
                                "snippet": snippet, "score": round(score, 4)})

    groundedness = supported / len(claims)
    return {"groundedness": round(groundedness, 4), "unsupported": unsupported,
            "n_claims": len(claims), "supported": supported}


def _reopen_ids(unsupported: list[dict], evidence: list[dict]) -> list[str]:
    """Map each unsupported brief claim back to the plan task that produced it (F-07b targeted repair).

    The `source_task` provenance tag the Researcher stamps (`_stamp`) lives on the raw evidence dicts on
    the blackboard — NOT on the brief claim, because the Editor re-validates each claim through the
    `Evidence` pydantic model, which drops non-schema keys. So we resolve the id by matching the
    unsupported claim text against `state["evidence"]` (claim text is preserved verbatim through every
    `model_validate` hop). Returns the DISTINCT originating task ids, so the back-edge reopens exactly
    the failing task(s) instead of the old no-op ("all covered -> plan exhausted") back-edge.
    """
    by_claim: dict[str, str] = {}
    by_snippet: dict[str, str] = {}
    for ev in evidence or []:
        tid = ev.get("source_task")
        if not tid:
            continue
        claim = ev.get("claim")
        snippet = (ev.get("citation") or {}).get("snippet")
        if claim and claim not in by_claim:
            by_claim[claim] = tid
        if snippet and snippet not in by_snippet:
            by_snippet[snippet] = tid
    ids: list[str] = []
    for item in unsupported:
        # prefer the claim text; fall back to the (never-rewritten) snippet if softening changed the text
        tid = by_claim.get(item.get("claim", "")) or by_snippet.get(item.get("snippet", ""))
        if tid and tid not in ids:
            ids.append(tid)
    return ids


def source_guard_node(state: dict, embeddings=None) -> dict:
    """Post-gen support verification; flag `unsupported` to trigger the back-edge (§5.1 edge).

    The back-edge to the Researcher is bounded by `editor_reretrieve_max` (the router reads
    `editor_rounds`). Since this is a *re-retrieve* trigger like the Editor's own, we increment the
    SAME counter here when we route back — otherwise the source_guard→researcher cycle is unbounded
    when the Editor's structural check passes but embedding-support falls below the floor.

    When we route back, we also populate `reopen` with the ids of the plan tasks that produced the
    unsupported claims (F-07b). This turns a previously no-op back-edge — the Researcher saw every task
    as `covered` and immediately reported "plan exhausted" — into a *targeted* repair of exactly the
    task(s) whose evidence failed support verification.
    """
    s = get_settings()
    brief = state.get("draft_brief", {}) or {}
    result = verify_claims(brief, embeddings=embeddings)
    unsupported = result["groundedness"] < s.groundedness_target and bool(result["unsupported"])
    rounds = state.get("editor_rounds", 0)
    out = {"groundedness": result["groundedness"], "source_report": result,
           "unsupported": unsupported}
    # only count a round / reopen a task when we will actually route back (unsupported AND under budget)
    if unsupported and rounds < s.editor_reretrieve_max:
        rounds += 1
        reopen = _reopen_ids(result["unsupported"], state.get("evidence"))
        out["reopen"] = reopen
        out["log"] = [(f"source_guard: groundedness={result['groundedness']} unsupported=True "
                      f"rounds={rounds} reopen={reopen}")]
    else:
        out["log"] = [(f"source_guard: groundedness={result['groundedness']} "
                      f"unsupported={unsupported} rounds={rounds}")]
    out["editor_rounds"] = rounds
    return out
