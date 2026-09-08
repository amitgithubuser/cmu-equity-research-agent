"""RAG retrieval — online, at question time (architecture §6.5, CP 3.1 §5).

The four safeguards, IN ORDER, that defend against "pulling the wrong company or wrong period":
  (1) FILTER FIRST by {company, period} — a competitor or an old year *cannot* be returned.
  (2) SEARCH by meaning — fetch more candidates than needed.
  (3) MMR RE-RANK — relevance balanced against diversity (Lab 3.2); keep a focused few.
  (4) REFUSE WEAK — if nothing clears the similarity floor, return [] so the agent skips the
      question rather than forcing a weak passage in (CP 3.1 negative-rejection rule).

Returns `Doc`s carrying `{company, period, section, source, url}` metadata — enough to cite and to
self-review (§6.5 step 6).
"""

from __future__ import annotations

from ..config import get_settings
from .store import Doc, InMemoryVectorStore, get_store


def retrieve(
    query: str,
    company: str,
    period: str | None = None,
    document_type: str | None = None,
    transcript_periods: list[str] | None = None,
    settings=None,
    store=None,
) -> list[Doc]:
    """Filter → MMR search → refuse-weak. An empty result means 'insufficient evidence', never a
    fabricated passage.
    """
    settings = settings or get_settings()
    store = store if store is not None else get_store(settings)

    # (1) FILTER FIRST — build a Chroma-style filter; period is optional (some queries span periods).
    # F-05: normalize the requested period to the SAME canonical label ingestion stored ("FY2025"),
    # so "--as-of FY2025" matches a chunk indexed from report date "2025-01-26" (both -> "FY2025").
    from ..periods import canonical_label

    clauses = [{"company": company}]
    if period:
        clauses.append({"period": canonical_label(period) or period})
    if document_type:
        clauses.append({"document_type": document_type})
    if transcript_periods:
        clauses.append({"transcript_period": {"$in": list(dict.fromkeys(transcript_periods))}})
    flt = clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # (2)+(3) SEARCH + MMR RE-RANK
    docs = store.mmr_search(
        query, k=settings.retrieve_k, fetch_k=settings.retrieve_fetch_k,
        lambda_mult=settings.mmr_lambda, flt=flt,
    )

    # (4) REFUSE WEAK — drop anything below the floor; [] if nothing survives.
    floor = (
        min(settings.refuse_floor, settings.transcript_refuse_floor)
        if document_type == "earnings_call_transcript"
        else settings.refuse_floor
    )
    kept = [d for d in docs if d.score >= floor]
    return kept


def build_test_store(chunks: list[dict]) -> InMemoryVectorStore:
    """Helper for tests/fixtures: an in-memory store pre-loaded with `[{"text","metadata"}]`."""
    store = InMemoryVectorStore()
    store.add(chunks)
    return store
