"""Vector store abstraction (architecture §6.4/§6.5, D-9).

`InMemoryVectorStore` is a small numpy-backed store that supports metadata filtering, cosine
similarity search, and **explicit MMR** (Lab 3.2). It backs the offline tests and doubles as a
zero-dependency fallback. On a real run `get_store()` returns a Chroma-backed store with the same
interface, so `retrieve.py` never branches.

Keeping MMR explicit here (rather than delegating to Chroma's `search_type="mmr"`) is deliberate:
it's the Lab 3.2 algorithm the capstone is meant to demonstrate — relevance balanced against
diversity — and it's unit-testable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .embeddings import get_embeddings


@dataclass
class Doc:
    """A retrieved chunk: its text, its citation-and-filter metadata, and (optionally) its score."""

    text: str
    metadata: dict
    score: float = 0.0


def chunk_id(chunk: dict) -> str:
    """Return a stable document-chunk id so re-fetching a source updates instead of duplicates.

    The identity uses provenance and chunk position, not the text. If a public transcript page is
    corrected between runs, Chroma can replace the existing chunk under the same id.
    """
    metadata = chunk.get("metadata", {}) or {}
    identity = {
        key: metadata.get(key, "")
        for key in (
            "company", "period", "transcript_period", "document_type", "source", "url",
            "section", "chunk", "section_part",
        )
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matches_filter(meta: dict, flt: dict | None) -> bool:
    """Support a small subset of Chroma-style filters: {k: v} equality, and {"$and": [...]}."""
    if not flt:
        return True
    if "$and" in flt:
        return all(_matches_filter(meta, sub) for sub in flt["$and"])
    for key, expected in flt.items():
        if isinstance(expected, dict) and "$in" in expected:
            if meta.get(key) not in expected["$in"]:
                return False
        elif meta.get(key) != expected:
            return False
    return True


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_mult: float) -> list[int]:
    """Maximal Marginal Relevance (Lab 3.2): iteratively pick the candidate that maximizes
    `λ·relevance(query) − (1−λ)·max similarity to already-selected`. Returns selected indices.

    All vectors are L2-normalized, so dot products are cosine similarities.
    """
    if len(cand_vecs) == 0:
        return []
    relevance = cand_vecs @ query_vec
    selected: list[int] = []
    remaining = list(range(len(cand_vecs)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel = cand_vecs[selected]
            def mmr_score(i: int, selected_vecs=sel) -> float:
                redundancy = float(np.max(selected_vecs @ cand_vecs[i]))
                return lambda_mult * float(relevance[i]) - (1 - lambda_mult) * redundancy
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


class InMemoryVectorStore:
    """Numpy-backed store with metadata filter + cosine search + MMR. Interface mirrors the real store."""

    def __init__(self, embeddings=None):
        self.embeddings = embeddings or get_embeddings()
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._ids: set[str] = set()

    def add(self, chunks: list[dict]) -> None:
        """Add `[{"text", "metadata"}]` chunks; embeds and stores them."""
        fresh = [chunk for chunk in chunks if chunk_id(chunk) not in self._ids]
        if not fresh:
            return
        vecs = np.array(self.embeddings.embed_documents([c["text"] for c in fresh]), dtype=np.float32)
        self._vecs = vecs if self._vecs.size == 0 else np.vstack([self._vecs, vecs])
        self._texts.extend(c["text"] for c in fresh)
        self._metas.extend(c["metadata"] for c in fresh)
        self._ids.update(chunk_id(chunk) for chunk in fresh)

    def _filtered_indices(self, flt: dict | None) -> list[int]:
        return [i for i, m in enumerate(self._metas) if _matches_filter(m, flt)]

    def inventory(self, flt: dict | None = None) -> list[dict]:
        """Return metadata only, for source-coverage checks without loading document bodies."""
        return [dict(self._metas[i]) for i in self._filtered_indices(flt)]

    def similarity_search(self, query: str, k: int, flt: dict | None = None) -> list[Doc]:
        """Top-k by cosine similarity within the metadata filter, each Doc carrying its score."""
        idx = self._filtered_indices(flt)
        if not idx:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        sims = self._vecs[idx] @ q
        order = np.argsort(-sims)[:k]
        return [Doc(self._texts[idx[o]], self._metas[idx[o]], float(sims[o])) for o in order]

    def mmr_search(self, query: str, k: int, fetch_k: int, lambda_mult: float,
                   flt: dict | None = None) -> list[Doc]:
        """Fetch `fetch_k` by relevance, then re-rank to `k` with MMR. Scores are query relevance."""
        idx = self._filtered_indices(flt)
        if not idx:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        sims = self._vecs[idx] @ q
        top = list(np.argsort(-sims)[:fetch_k])
        cand_vecs = self._vecs[[idx[t] for t in top]]
        chosen = mmr(q, cand_vecs, k=k, lambda_mult=lambda_mult)
        docs = []
        for c in chosen:
            gi = idx[top[c]]
            docs.append(Doc(self._texts[gi], self._metas[gi], float(sims[top[c]])))
        return docs


@lru_cache(maxsize=8)
def _cached_persistent_store(chroma_dir: str, embeddings_backend: str, embedding_model: str):
    """Reuse one Chroma client and embedding model instead of reopening both for every query."""
    class _EmbeddingSettings:
        pass

    embed_settings = _EmbeddingSettings()
    embed_settings.embeddings_backend = embeddings_backend
    embed_settings.embedding_model = embedding_model
    embeddings = get_embeddings(embed_settings)
    try:
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(embeddings, persist_directory=chroma_dir)
    except Exception:  # noqa: BLE001 - chroma not installed / offline
        return InMemoryVectorStore(embeddings=embeddings)


def get_store(settings=None, persist: bool = True):
    """Return a persistent Chroma-backed store for real runs, else the in-memory store.

    The RAG layer always speaks the same store interface (`add`/`similarity_search`/`mmr_search`),
    so it's oblivious to which backend it gets. Chroma (D-9) is used when installed; otherwise the
    numpy in-memory store backs tests and offline use. Embeddings are the real HF model when
    available (see `get_embeddings`).
    """
    from ..config import get_settings

    settings = settings or get_settings()
    if persist:
        return _cached_persistent_store(
            str(settings.chroma_dir), str(settings.embeddings_backend), str(settings.embedding_model)
        )
    return InMemoryVectorStore(embeddings=get_embeddings(settings))
