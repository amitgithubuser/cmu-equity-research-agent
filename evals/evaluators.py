"""Evaluators — Lab 6.1's checks promoted to metrics (architecture §11.1, implementation-plan §11.2).

Each evaluator is a scoring function over a brief (and optionally a reference). On a live run these are
registered as **LangSmith Evaluators** over `(input, output, reference)`; offline they are pure Python
so the ablation runner is deterministic and CI-safe.

| Metric | Target | Implementation |
|---|---|---|
| Groundedness / citation coverage | ≥ 0.95 | `source_guard.verify_claims` support rate |
| Numeric correctness              | 1.0    | every `number` claim's value matches a reference |
| Hallucination rate               | ≈ 0    | fraction of claims below the support floor |
| Overconfidence                   | PASS   | `confidence_guard.scan_overconfidence` on the prose |
| Calibration (ECE)                | low    | stated confidence vs. correctness across a bucketed set |
| Faithfulness (RAG triad)         | ≥ 0.95 | claim entailed by *some* retrieved snippet, not just its own citation |
| Answer relevance (RAG triad)     | high   | similarity of the brief to the originating question |
| Retrieval relevance (RAG triad)  | high   | mean similarity of retrieved snippets to the question |

The last three are the classic **RAG evaluation triad** (context → answer → question). They reuse
`source_guard.support_score` — `max(embedding_similarity, jaccard)` — so they stay meaningful offline
where the deterministic hashing embedder is not semantic (the Jaccard token-overlap term carries the
signal). `evaluate_brief` deliberately stays the single-brief bundle (the four self-contained checks);
the triad needs a `question` and/or retrieved `context`, so it is exposed separately via `rag_triad`.
"""

from __future__ import annotations

from equity_research.config import get_settings
from equity_research.guardrails.confidence_guard import _brief_prose, scan_overconfidence
from equity_research.guardrails.source_guard import support_score, verify_claims


def groundedness(brief: dict, embeddings=None) -> float:
    """Fraction of claims supported by their own citation snippet (source_guard support rate)."""
    return verify_claims(brief, embeddings=embeddings)["groundedness"]


def hallucination_rate(brief: dict, embeddings=None) -> float:
    """Fraction of claims that are NOT supported = 1 − groundedness."""
    result = verify_claims(brief, embeddings=embeddings)
    if result["n_claims"] == 0:
        return 0.0
    return round(len(result["unsupported"]) / result["n_claims"], 4)


def numeric_correctness(brief: dict, reference: dict | None = None, *, rel_tol: float = 0.01) -> float:
    """Fraction of `number` claims whose stated value MATCHES the reference (F-09).

    With a `reference` map (`{claim_text: ground_truth_value}`), each number claim scores correct only
    if its value is within `rel_tol` of the reference — so a *wrong* number fails. The old check only
    tested that a number was non-null, which let any wrong figure through; that was the F-09 bug.

    Absent a reference (or for a claim not in it) we fall back to the weaker "carries a computed value"
    check — numbers must still come from `calc`, not prose — because you can only verify a figure where
    a ground truth exists. Supply a `numeric_reference` from the gold set to get the real comparison.
    """
    nums = [ev for group in ("bull_case", "bear_case", "key_risks")
            for ev in (brief.get(group, []) or []) if ev.get("kind") == "number"]
    if not nums:
        return 1.0  # vacuously correct — no numeric claims to get wrong
    reference = reference or {}
    good = 0
    for ev in nums:
        value = ev.get("value")
        if value is None:
            continue  # a number with no computed value is never correct
        ref = reference.get(ev.get("claim", ""))
        if ref is None:
            good += 1  # no ground truth for this claim -> presence check only
        else:
            denom = abs(float(ref)) or 1.0
            if abs(float(value) - float(ref)) / denom <= rel_tol:
                good += 1
    return round(good / len(nums), 4)


def overconfidence_verdict(brief: dict) -> str:
    """PASS / WARN / FAIL from the confidence-marker scan over the brief prose."""
    return scan_overconfidence(_brief_prose(brief))["verdict"]


def expected_calibration_error(confidences: list[float], correct: list[int], n_bins: int = 5) -> float:
    """ECE: |mean confidence − accuracy| per bin, weighted by bin size (lower is better)."""
    if not confidences:
        return 0.0
    n = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c <= hi)]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return round(ece, 4)


def _brief_claims(brief: dict) -> list[str]:
    """Every claim's text across the three case groups (for triad scoring)."""
    return [ev.get("claim", "")
            for group in ("bull_case", "bear_case", "key_risks")
            for ev in (brief.get(group, []) or [])]


def faithfulness(brief: dict, context: list[str] | None = None, embeddings=None) -> float:
    """RAG-triad #1: fraction of claims entailed by *some* retrieved snippet (F-09).

    `groundedness` checks a claim against its OWN citation; faithfulness is stricter — it asks whether
    the claim is actually supported by the retrieved context set (any snippet), which catches a claim
    stapled to a citation that doesn't really support it. With no `context`, we fall back to each claim's
    own citation snippet so the metric still returns a sensible score offline.
    """
    claims_and_snips = []
    for group in ("bull_case", "bear_case", "key_risks"):
        for ev in brief.get(group, []) or []:
            own = (ev.get("citation") or {}).get("snippet", "")
            claims_and_snips.append((ev.get("claim", ""), own))
    if not claims_and_snips:
        return 1.0
    floor = get_settings().source_support_floor
    pool = list(context) if context else None
    good = 0
    for claim, own in claims_and_snips:
        snippets = pool if pool else [own]
        if any(support_score(claim, s, embeddings) >= floor for s in snippets):
            good += 1
    return round(good / len(claims_and_snips), 4)


def answer_relevance(brief: dict, question: str, embeddings=None) -> float:
    """RAG-triad #2: does the brief actually answer the question it was asked? (F-09)

    Similarity of the brief's claim text to the originating question — a brief that drifts off-topic
    scores low even if every claim is individually grounded.
    """
    claims = " ".join(_brief_claims(brief)).strip()
    if not question or not claims:
        return 0.0
    return round(support_score(question, claims, embeddings), 4)


def retrieval_relevance(question: str, context: list[str], embeddings=None) -> float:
    """RAG-triad #3: did retrieval fetch snippets relevant to the question? (F-09)

    Mean similarity of each retrieved snippet to the question — low means the retriever pulled
    off-topic context, the upstream cause of an unfaithful or irrelevant answer.
    """
    context = [c for c in (context or []) if c]
    if not question or not context:
        return 0.0
    return round(sum(support_score(question, s, embeddings) for s in context) / len(context), 4)


def rag_triad(brief: dict, question: str, context: list[str] | None = None, embeddings=None) -> dict:
    """The RAG evaluation triad in one call (F-09) — kept SEPARATE from `evaluate_brief`.

    Needs a `question` (and ideally the retrieved `context`), which the single-brief bundle doesn't
    carry, so it lives here rather than bloating `evaluate_brief`'s four self-contained checks.
    """
    return {
        "faithfulness": faithfulness(brief, context, embeddings),
        "answer_relevance": answer_relevance(brief, question, embeddings),
        "retrieval_relevance": retrieval_relevance(question, context or [], embeddings),
    }


def evaluate_brief(brief: dict, embeddings=None) -> dict:
    """Run all single-brief evaluators at once → a metric dict.

    Deliberately the four *self-contained* checks (no question/context needed). The RAG triad
    (faithfulness / answer- / retrieval-relevance) is exposed separately via `rag_triad`.
    """
    return {
        "groundedness": groundedness(brief, embeddings),
        "hallucination_rate": hallucination_rate(brief, embeddings),
        "numeric_correctness": numeric_correctness(brief),
        "overconfidence": overconfidence_verdict(brief),
    }
