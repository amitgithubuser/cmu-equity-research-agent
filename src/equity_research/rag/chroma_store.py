"""Chroma-backed store adapter (architecture §6, decision D-9).

Wraps `langchain_chroma.Chroma` behind the SAME `.add / .similarity_search / .mmr_search` interface
as `InMemoryVectorStore`, so `retrieve.py` is oblivious to which store it's talking to. This is the
real, persistent, local vector DB the design targets; it's only imported when `.[real]` is present.
"""

from __future__ import annotations

from .store import Doc, chunk_id


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma's squared-L2 distance for normalized vectors to cosine similarity."""
    return max(-1.0, min(1.0, 1.0 - float(distance) / 2.0))


class ChromaVectorStore:  # pragma: no cover - exercised only with .[real] installed
    """Persistent local Chroma store presenting the project's store interface."""

    def __init__(self, embeddings, persist_directory: str):
        from langchain_chroma import Chroma

        self._store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self._store.add_texts(
            texts=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[chunk_id(c) for c in chunks],
        )

    def inventory(self, flt: dict | None = None) -> list[dict]:
        """Return stored metadata without source text, for safe cached-source coverage checks."""
        result = self._store.get(where=flt, include=["metadatas"])
        return [dict(item) for item in (result.get("metadatas") or [])]

    def similarity_search(self, query: str, k: int, flt: dict | None = None) -> list[Doc]:
        pairs = self._store.similarity_search_with_score(query, k=k, filter=flt)
        return [Doc(d.page_content, d.metadata, _distance_to_similarity(distance))
                for d, distance in pairs]

    def mmr_search(self, query: str, k: int, fetch_k: int, lambda_mult: float,
                   flt: dict | None = None) -> list[Doc]:
        docs = self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=flt
        )
        # Chroma's MMR method doesn't return scores; attach relevance from a parallel scored query
        # so the refuse-floor still has a number to compare against.
        scored = {
            d.page_content: _distance_to_similarity(distance)
            for d, distance in self._store.similarity_search_with_score(
                query, k=fetch_k, filter=flt
            )
        }
        return [Doc(d.page_content, d.metadata, scored.get(d.page_content, -1.0)) for d in docs]
