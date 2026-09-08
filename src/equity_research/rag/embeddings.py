"""Embedding backends (architecture §6.4, D-9).

Real runs use the local open-source `sentence-transformers` model (`BAAI/bge-small-en-v1.5`) — no
API key, nothing leaves the machine. Tests use `HashingEmbeddings`: a tiny deterministic bag-of-words
hasher with the SAME interface (`embed_documents` / `embed_query`), so the whole RAG pipeline —
chunking, cosine similarity, MMR, the refuse-floor — is exercised offline with zero downloads.

Both return L2-normalized vectors, so a dot product IS cosine similarity.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


class HashingEmbeddings:
    """Deterministic hashed bag-of-words embeddings for offline tests (no network, no model download)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            vec[hash(tok) % self.dim] += 1.0
        return _normalize(vec).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


@lru_cache(maxsize=8)
def _cached_embeddings(backend: str, model_name: str):
    """Construct each configured local embedding model once per Python process."""
    if backend == "hashing":
        return HashingEmbeddings()

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:  # offline / not installed
        if backend == "hf":
            raise
        return HashingEmbeddings()


def get_embeddings(settings=None):
    """Return the embedder chosen by `settings.embeddings_backend` (D-9).

    - ``"hashing"`` → always the deterministic offline embedder (the test suite pins this so runs are
      reproducible and never download a model — see conftest).
    - ``"hf"``      → always the real `BAAI/bge-small-en-v1.5` model; raises if it isn't installed.
    - ``"auto"``    → real model if `langchain_huggingface` imports, else transparent hashing fallback.

    Keeping the switch here means callers never branch on install tier, and the real vs. offline
    score distributions (which differ — the reason for OQ-3's Phase-10 calibration) stay isolated
    to this one function.
    """
    from ..config import get_settings

    settings = settings or get_settings()
    backend = getattr(settings, "embeddings_backend", "auto")
    return _cached_embeddings(str(backend), str(settings.embedding_model))
