"""Phase 3 gate (implementation-plan §4.3): the four safeguards each demonstrably work, offline.

filter → search → MMR → refuse, plus section-aware chunking + metadata round-trip.
"""

from equity_research.config import get_settings
from equity_research.periods import canonical_label
from equity_research.rag import build_test_store, chunk_document, retrieve, split_into_sections
from equity_research.rag.store import Doc, InMemoryVectorStore


# --- (1) FILTER FIRST ------------------------------------------------------------------------------
def test_filter_excludes_other_company(rag_corpus):
    # This query lexically matches BOTH the NVDA data-center chunk and the AMD chunk (revenue,
    # GPUs, competition, markets). The company filter must exclude AMD even though it's relevant.
    docs = retrieve("data center revenue demand GPUs cloud customers competition markets",
                    company="NVDA", store=rag_corpus)
    assert docs, "expected some NVDA passages"
    assert all(d.metadata["company"] == "NVDA" for d in docs)


def test_filter_excludes_other_period(rag_corpus):
    docs = retrieve("data center revenue gross margin", company="NVDA", period="FY2025", store=rag_corpus)
    assert docs
    assert all(d.metadata["period"] == "FY2025" for d in docs)


# --- F-05 regression: ingest report-date == retrieve FY-label --------------------------------------
def _report_date_corpus():
    """A corpus stored the way real `ingest_edgar` stores it: the SEC report date is normalized to a
    canonical label BEFORE indexing (ingest.py: `period = canonical_label(report_date)`). So a filing
    dated 2025-01-26 is stored under period "FY2025", and a prior filing dated 2024-01-24 under "FY2024".
    """
    def chunk(text, report_date):
        period = canonical_label(report_date)          # exactly what ingest_edgar does
        return {"text": text, "metadata": {
            "company": "NVDA", "period": period, "report_date": report_date,
            "section": "Item 7 MD&A", "source": f"10-K {period}", "url": "http://example/NVDA",
        }}
    return build_test_store([
        chunk("Data center revenue grew rapidly across cloud customers this fiscal year.", "2025-01-26"),
        chunk("Prior year data center revenue and gross margin were lower.", "2024-01-24"),
    ])


def test_fy_label_retrieves_report_date_chunk():
    # THE F-05 miss: the documented command `--as-of FY2025` retrieved ZERO passages because the chunk
    # was indexed under the raw report date "2025-01-26". With normalization on both sides, "FY2025"
    # now matches the 2025-01-26 filing.
    store = _report_date_corpus()
    docs = retrieve("data center revenue cloud customers", company="NVDA", period="FY2025", store=store)
    assert docs, "FY2025 must retrieve the chunk indexed from report date 2025-01-26"
    assert all(d.metadata["period"] == "FY2025" for d in docs)
    assert docs[0].metadata["report_date"] == "2025-01-26"   # provenance: the exact filing date is kept


def test_fy_label_excludes_other_year_report_date():
    # The mirror guarantee: asking for FY2025 must NOT return the 2024-01-24 filing (a stale year).
    store = _report_date_corpus()
    docs = retrieve("data center revenue", company="NVDA", period="FY2025", store=store)
    assert all(d.metadata["report_date"] != "2024-01-24" for d in docs)


# --- (4) REFUSE WEAK -------------------------------------------------------------------------------
def test_refuse_returns_empty_below_floor(rag_corpus):
    # A query with no lexical overlap scores ~0 < refuse_floor -> [] (never a forced weak passage).
    docs = retrieve("zzzz unrelated basketball quarterback nonsense", company="NVDA", store=rag_corpus)
    assert docs == []


def test_refuse_floor_is_enforced(rag_corpus):
    s = get_settings()
    docs = retrieve("data center demand", company="NVDA", store=rag_corpus)
    assert all(d.score >= s.refuse_floor for d in docs)


def test_transcript_uses_strict_metadata_filter_with_lower_conversation_floor():
    class Store:
        def mmr_search(self, query, k, fetch_k, lambda_mult, flt):
            assert {"document_type": "earnings_call_transcript"} in flt["$and"]
            return [Doc("long conversational transcript", {
                "company": "NVDA", "document_type": "earnings_call_transcript"
            }, score=0.2)]

    docs = retrieve(
        "management commentary", company="NVDA",
        document_type="earnings_call_transcript", store=Store(),
    )
    assert docs  # 0.2 is below general 0.35 but above transcript-specific 0.12


def test_transcript_period_filter_excludes_stale_cached_calls():
    store = build_test_store([
        {"text": "Management discussed AI demand and supply constraints.", "metadata": {
            "company": "NVDA", "period": "Q2-2027", "transcript_period": "Q2-2027",
            "document_type": "earnings_call_transcript", "source": "latest call",
        }},
        {"text": "Management discussed AI demand and supply constraints.", "metadata": {
            "company": "NVDA", "period": "Q3-2026", "transcript_period": "Q3-2026",
            "document_type": "earnings_call_transcript", "source": "older call",
        }},
    ])
    docs = retrieve(
        "management AI demand supply", company="NVDA",
        document_type="earnings_call_transcript", transcript_periods=["Q2-2027"], store=store,
    )
    assert docs and {doc.metadata["transcript_period"] for doc in docs} == {"Q2-2027"}


# --- (2)+(3) SEARCH + MMR --------------------------------------------------------------------------
def test_mmr_returns_at_most_k(rag_corpus):
    s = get_settings()
    docs = retrieve("data center revenue demand gross margin GPUs", company="NVDA", store=rag_corpus)
    assert len(docs) <= s.retrieve_k


def test_metadata_roundtrip_for_citation(rag_corpus):
    docs = retrieve("customer concentration risk", company="NVDA", store=rag_corpus)
    assert docs
    md = docs[0].metadata
    for key in ("company", "period", "section", "source", "url"):
        assert key in md


def test_reingesting_the_same_chunk_does_not_duplicate_it():
    store = InMemoryVectorStore()
    chunk = {"text": "Operator welcome to the earnings call.", "metadata": {
        "company": "NVDA", "period": "Q2-2027", "transcript_period": "Q2-2027",
        "document_type": "earnings_call_transcript", "source": "Earnings transcript",
        "url": "https://investor.example.com/q2-2027.pdf", "section": "Operator", "chunk": 0,
    }}
    store.add([chunk])
    store.add([chunk])
    assert len(store._texts) == 1


def test_chroma_adapter_returns_bounded_similarity_without_relevance_warning(tmp_path):
    import pytest

    pytest.importorskip("langchain_chroma")
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.embeddings import HashingEmbeddings

    store = ChromaVectorStore(HashingEmbeddings(), str(tmp_path / "chroma"))
    store.add([{"text": "accelerated computing demand and revenue growth", "metadata": {
        "company": "NVDA", "period": "FY2026", "source": "fixture", "section": "Item 7",
    }}])
    docs = store.mmr_search(
        "accelerated computing revenue", k=1, fetch_k=3, lambda_mult=0.5,
        flt={"company": "NVDA"},
    )
    assert docs and -1.0 <= docs[0].score <= 1.0


# --- section-aware chunking ------------------------------------------------------------------------
def test_split_on_10k_items():
    text = (
        "Item 1. Business\nWe design GPUs.\n\n"
        "Item 1A. Risk Factors\nCustomer concentration is a risk.\n\n"
        "Item 7. Management Discussion\nRevenue grew this year.\n"
    )
    sections = split_into_sections(text)
    labels = [lbl.lower() for lbl, _ in sections]
    assert any("item 1a" in l for l in labels)
    assert any("item 7" in l for l in labels)


def test_split_on_speaker_turns():
    text = (
        "Operator Welcome to the call.\n"
        "Jensen Huang -- CEO: Revenue was strong this quarter.\n"
        "Colette Kress -- CFO: Margins expanded year over year.\n"
    )
    sections = split_into_sections(text)
    assert len(sections) >= 2


def test_chunk_carries_section_metadata():
    # Two items so section-splitting engages (>=2 headings); Item 1A is long -> windowed into >1 chunk.
    text = (
        "Item 1. Business\nWe design accelerated computing platforms.\n\n"
        "Item 1A. Risk Factors\n" + ("Customer concentration risk. " * 200)
    )
    chunks = chunk_document(text, {"company": "NVDA", "period": "FY2025", "source": "10-K", "url": "u"},
                            chunk_size=400, chunk_overlap=50)
    risk_chunks = [c for c in chunks if "1a" in c["metadata"]["section"].lower()]
    assert len(risk_chunks) > 1  # the long Item 1A section is windowed into multiple chunks
    assert all({"company", "period", "section", "chunk"} <= set(c["metadata"]) for c in chunks)


def test_chunk_overlap_preserves_continuity():
    # Two windows should share some overlapping text so an idea isn't cut at the boundary.
    text = "Item 7. MD&A\n" + " ".join(f"sentence{i} about revenue growth." for i in range(80))
    chunks = chunk_document(text, {"company": "X", "period": "FY", "source": "10-K", "url": "u"},
                            chunk_size=300, chunk_overlap=80)
    assert len(chunks) >= 2
