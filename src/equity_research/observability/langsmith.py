"""Explicit LangSmith tracing without relying on process-wide environment mutation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ..config import get_settings


def tracing_enabled(settings=None) -> bool:
    settings = settings or get_settings()
    # Require the current, explicit LangSmith flag. A developer's legacy LANGCHAIN_TRACING_V2 value
    # must not make offline tests or local diagnostics upload traces unexpectedly.
    return bool(settings.langsmith_api_key and settings.langsmith_tracing)


def run_tracing_context(
    *,
    ticker: str,
    question: str,
    as_of: str,
    thread_id: str,
    settings=None,
):
    """Return a LangSmith context or a no-op context when tracing is disabled.

    Only safe operational metadata is attached. API keys, request headers, provider URLs, and source
    document bodies never enter this metadata object.
    """
    settings = settings or get_settings()
    if not tracing_enabled(settings):
        return nullcontext()
    try:
        from langsmith import Client, tracing_context
    except ModuleNotFoundError:
        return nullcontext()

    client = Client(api_key=settings.langsmith_api_key)
    metadata: dict[str, Any] = {
        "ticker": ticker,
        "as_of": as_of or "latest",
        "thread_id": thread_id,
        "request_type": "broad" if "research" in (question or "").lower() else "focused",
    }
    return tracing_context(
        enabled=True,
        client=client,
        project_name=settings.langsmith_project,
        tags=["equity-research", "capstone"],
        metadata=metadata,
    )


def graph_run_config(*, ticker: str, as_of: str, thread_id: str) -> dict:
    """LangGraph invocation config shared by initial runs and HITL resumes."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "equity-research-analysis",
        "tags": ["equity-research", ticker],
        "metadata": {"ticker": ticker, "as_of": as_of or "latest", "thread_id": thread_id},
    }
