"""RAG over filings/transcripts (architecture §6.4/§6.5, decision D-9).

External memory for the agent: section-aware chunking + local open-source embeddings + a vector
store, with a filter→MMR→refuse retrieval path. Text goes through RAG; numbers do not (D-4).
"""

from .chunking import chunk_document, split_into_sections
from .ingest import (
    ingest_edgar,
    ingest_filing,
    ingest_filing_window,
    ingest_transcript_window,
    transcript_inventory,
)
from .retrieve import build_test_store, retrieve
from .store import Doc, InMemoryVectorStore, get_store, mmr

__all__ = [
    "Doc",
    "InMemoryVectorStore",
    "build_test_store",
    "chunk_document",
    "get_store",
    "ingest_edgar",
    "ingest_filing",
    "ingest_filing_window",
    "ingest_transcript_window",
    "mmr",
    "retrieve",
    "split_into_sections",
    "transcript_inventory",
]
