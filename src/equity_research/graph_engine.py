"""Graph engine bindings (architecture §5).

The capstone runs on **real LangGraph** — the plan makes it a core dependency (implementation-plan
§1.1: `StateGraph`, `interrupt()`, `Command`, checkpointer). This module is the single place the rest
of the code imports those primitives from, plus three small helpers that smooth over LangGraph's
HITL surface:

  * `new_checkpointer()` — an in-memory checkpointer (real LangGraph *requires* one to serve
    `interrupt()` / `Command(resume=...)`);
  * `interrupt_payload(result)` / `is_interrupted(result)` — normalize the paused-graph shape, since
    LangGraph surfaces `{"__interrupt__": [Interrupt(value=<payload>), ...]}` (a *list* of `Interrupt`
    objects) and callers want the payload dict directly.

Tests still run fully offline and token-free: LangGraph executes locally with no network, and the node
`llm`/`tools`/`retrieve_fn` seams inject fakes (guiding principle #4, the "mockable LLM boundary").
There is deliberately no home-grown fallback engine — a second engine diverged from LangGraph's channel
semantics and produced false-green tests, so it was removed in favor of the real thing everywhere.
"""

from __future__ import annotations

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
except ModuleNotFoundError as exc:  # pragma: no cover - guides the reader to the right install
    raise ModuleNotFoundError(
        "LangGraph is required. Install the runtime with `pip install -e .[real]` "
        "(architecture §2; implementation-plan §1.1)."
    ) from exc

# Kept True so any lingering `if HAVE_LANGGRAPH:` reads correctly; LangGraph is now always present.
HAVE_LANGGRAPH = True


def new_checkpointer():
    """Return a fresh in-memory checkpointer.

    Real LangGraph *requires* a checkpointer to serve `interrupt()` / `Command(resume=...)` — without
    one it raises ``Cannot use Command(resume=...) without checkpointer``. `build_app`/`run_analysis`
    default to this so the HITL round-trip works out of the box; pass a durable checkpointer (e.g.
    `langgraph-checkpoint-sqlite`) to persist across processes.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def interrupt_payload(result):
    """Normalize the paused-graph payload, or return None if the graph didn't pause.

    LangGraph surfaces ``{"__interrupt__": [Interrupt(value=<payload>, id=...), ...]}`` — a *list* of
    `Interrupt` objects with the payload at ``.value``. Callers (run.py, the UI) use this so they read
    the reviewer payload uniformly without touching LangGraph's internal shape.
    """
    if not isinstance(result, dict) or "__interrupt__" not in result:
        return None
    raw = result["__interrupt__"]
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        first = raw[0]
        return getattr(first, "value", first)
    return raw   # already a payload (defensive; LangGraph returns a list)


def is_interrupted(result) -> bool:
    """True if the graph paused at a HITL interrupt."""
    return interrupt_payload(result) is not None


__all__ = [
    "END",
    "HAVE_LANGGRAPH",
    "START",
    "Command",
    "StateGraph",
    "interrupt",
    "interrupt_payload",
    "is_interrupted",
    "new_checkpointer",
]
