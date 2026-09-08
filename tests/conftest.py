"""Pytest fixtures shared across the suite. Tests run offline — no tokens, no network.

The tiny RAG corpus (2 companies × 2 periods) uses the deterministic HashingEmbeddings, so lexical
overlap drives similarity — enough to exercise filter / MMR / refuse-weak deterministically.
"""

import pytest

from equity_research.config import get_settings
from equity_research.rag import build_test_store


@pytest.fixture(autouse=True, scope="session")
def _pin_offline_embeddings():
    """Force the deterministic hashing embedder for the whole suite (D-9 / OQ-3).

    Once `.[real]` is installed, `get_embeddings("auto")` would silently download and use the real
    `bge-small` model — whose cosine distribution is NOT zero-centred, so an unrelated string still
    scores ~0.35 and the refuse-floor / support-floor assertions calibrated for the hashing embedder
    break. Pinning the backend keeps tests reproducible and offline regardless of install tier; the
    real-embedder floors are (re)calibrated from data in Phase 10, not asserted here.
    """
    import os

    os.environ["EMBEDDINGS_BACKEND"] = "hashing"
    # Long-term memory (§6.1): empty memory_dir => in-memory store, so the suite never writes a
    # `.memory/longterm.json` to disk and each run starts from a clean, isolated store.
    os.environ["MEMORY_DIR"] = ""
    get_settings.cache_clear()
    yield
    os.environ.pop("EMBEDDINGS_BACKEND", None)
    os.environ.pop("MEMORY_DIR", None)
    get_settings.cache_clear()


def _chunk(text, company, period, section):
    return {
        "text": text,
        "metadata": {
            "company": company, "period": period, "section": section,
            "source": f"10-K {period} {section}", "url": f"http://example/{company}/{period}",
        },
    }


@pytest.fixture
def rag_corpus():
    """A small hand-written corpus. Metadata carries {company, period, section, source, url}."""
    chunks = [
        _chunk("Gross margin expanded for three consecutive quarters driven by data center pricing power and strong demand.",
               "NVDA", "FY2025", "Item 7 MD&A"),
        _chunk("Data center revenue grew rapidly as demand for accelerated computing GPUs increased across cloud customers.",
               "NVDA", "FY2025", "Item 7 MD&A"),
        _chunk("Risk factors include customer concentration and dependence on a few large cloud purchasers of GPUs.",
               "NVDA", "FY2025", "Item 1A Risk Factors"),
        _chunk("Prior year data center revenue was smaller and gross margin was lower than the most recent fiscal year.",
               "NVDA", "FY2024", "Item 7 MD&A"),
        _chunk("Advanced Micro Devices reported client segment revenue and competition in the CPU and GPU markets.",
               "AMD", "FY2025", "Item 7 MD&A"),
    ]
    return build_test_store(chunks)
