"""End-to-end orchestrator (implementation-plan §12, Phase 11).

`run_analysis(...)` is the single entry point the CLI and the Streamlit UI both call. It:

  1. builds the real graph (`nodes_real.build_real_nodes` + `graph.build_graph`) with an injected
     reasoning LLM, routing LLM, tool allow-list, retriever, and embedder;
  2. invokes it on `{ticker, question, as_of}`;
  3. if the graph **pauses at the HITL interrupt** (`__interrupt__` in the result), **records the
     escalation in a `ReviewQueue`** (so a UI can list what is pending review), calls the supplied
     `decide` callback with the review payload, **resumes** with `Command(resume=...)` through the
     checkpointer, and **resolves the queue item** with the human's decision — so the human gate
     (§10.3) is honored *and* auditable end-to-end (F-09: the queue is the record; the interrupt is
     the pause);
  4. returns the final state (`brief`, `confidence`, `log`, `suspicion`, ...).

Everything is injected, so the SAME function runs live (real Claude + Chroma + yfinance/EDGAR) or fully
offline in a test (FakeLLM + fake tools + an in-memory store) with zero tokens. Live runs use an
explicit LangSmith tracing context, so tracing works with settings loaded from `.env` without mutating
the whole Python process.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from .config import get_settings
from .graph import build_graph
from .graph_engine import Command, interrupt_payload, new_checkpointer
from .nodes_real import build_real_nodes
from .observability.langsmith import graph_run_config, run_tracing_context
from .observability.review_queue import ReviewItem


def build_app(
    *,
    reasoning_llm,
    routing_llm=None,
    tools: dict | None = None,
    retrieve_fn=None,
    embeddings=None,
    known_symbols: set[str] | None = None,
    memory=None,
    transcript_ingest_fn=None,
    filing_ingest_fn=None,
    transcript_inventory_fn=None,
    checkpointer=None,
):
    """Compile the real graph. Separated from `run_analysis` so a UI can build once and reuse.

    A checkpointer is REQUIRED for the HITL `interrupt()`/`Command(resume)` round-trip: real LangGraph
    raises without one. We default to a fresh in-memory checkpointer so callers get working HITL for free
    (pass one explicitly to persist across processes).

    `memory` is the optional long-term store (§6.1). When omitted, the durable store from
    `get_longterm_store()` is used on real runs (JSON-backed if `memory_dir` is set); pass an explicit
    store (e.g. an in-memory one) to isolate a test or a batch run.
    """
    if memory is None:
        from .memory import get_longterm_store
        memory = get_longterm_store()
    nodes = build_real_nodes(
        reasoning_llm=reasoning_llm, routing_llm=routing_llm, tools=tools,
        retrieve_fn=retrieve_fn, embeddings=embeddings, known_symbols=known_symbols, memory=memory,
        transcript_ingest_fn=transcript_ingest_fn, filing_ingest_fn=filing_ingest_fn,
        transcript_inventory_fn=transcript_inventory_fn,
    )
    return build_graph(nodes, checkpointer=checkpointer or new_checkpointer())


def invoke_with_progress(
    app,
    graph_input,
    *,
    config: dict,
    on_progress: Callable[[str, dict], None] | None = None,
) -> dict:
    """Invoke a compiled graph and optionally report safe, node-level progress.

    LangGraph's ``updates`` stream exposes the name of each completed node.  The callback receives
    only that node name and its partial update; presentation layers must summarize the update rather
    than display prompts, source bodies, messages, or credentials.  Lightweight test doubles that
    only implement ``invoke`` continue to work unchanged.
    """
    started = perf_counter()
    if not hasattr(app, "stream") or not hasattr(app, "get_state"):
        result = app.invoke(graph_input, config=config)
        if isinstance(result, dict):
            result["runtime_profile"] = {
                "total_seconds": round(perf_counter() - started, 3), "stages": [],
            }
        return result

    interrupts = None
    last_event = started
    stage_seconds: dict[str, float] = {}
    for event in app.stream(graph_input, config=config, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        if "__interrupt__" in event:
            interrupts = event["__interrupt__"]
        for node_name, update in event.items():
            if node_name == "__interrupt__":
                continue
            now = perf_counter()
            name = str(node_name)
            stage_seconds[name] = stage_seconds.get(name, 0.0) + (now - last_event)
            last_event = now
            if on_progress is not None:
                on_progress(name, update if isinstance(update, dict) else {})

    snapshot = app.get_state(config)
    result = dict(getattr(snapshot, "values", {}) or {})
    if interrupts is not None:
        result["__interrupt__"] = interrupts
    result["runtime_profile"] = {
        "total_seconds": round(perf_counter() - started, 3),
        "stages": [
            {"stage": name, "seconds": round(seconds, 3)}
            for name, seconds in sorted(stage_seconds.items(), key=lambda item: item[1], reverse=True)
        ],
    }
    return result


def merge_runtime_profiles(*profiles: dict | None) -> dict:
    """Combine separately timed graph segments, including a post-review resume."""
    total = 0.0
    stages: dict[str, float] = {}
    for profile in profiles:
        if not profile:
            continue
        total += float(profile.get("total_seconds", 0.0) or 0.0)
        for row in profile.get("stages", []) or []:
            name = str(row.get("stage", "unknown"))
            stages[name] = stages.get(name, 0.0) + float(row.get("seconds", 0.0) or 0.0)
    return {
        "total_seconds": round(total, 3),
        "stages": [
            {"stage": name, "seconds": round(seconds, 3)}
            for name, seconds in sorted(stages.items(), key=lambda item: item[1], reverse=True)
        ],
    }


def run_analysis(
    ticker: str,
    question: str = "",
    *,
    reasoning_llm,
    routing_llm=None,
    tools: dict | None = None,
    retrieve_fn=None,
    embeddings=None,
    known_symbols: set[str] | None = None,
    memory=None,
    transcript_ingest_fn=None,
    filing_ingest_fn=None,
    transcript_inventory_fn=None,
    as_of: str = "",
    use_source_cache: bool = False,
    analysis_depth: int | None = None,
    decide: Callable[[dict], dict] | None = None,
    thread_id: str = "run",
    checkpointer=None,
    app=None,
    review_queue=None,
    on_progress: Callable[[str, dict], None] | None = None,
) -> dict:
    """Run one ticker end-to-end, honoring the HITL interrupt. Returns the final state dict.

    `review_queue` (optional `ReviewQueue`): when supplied, every HITL pause is enqueued as a
    `ReviewItem` and resolved with the reviewer's decision — the durable record of what was escalated
    and how it was decided (F-09). Omit it and behavior is unchanged (the interrupt still gates).
    """
    app = app or build_app(
        reasoning_llm=reasoning_llm, routing_llm=routing_llm, tools=tools,
        retrieve_fn=retrieve_fn, embeddings=embeddings, known_symbols=known_symbols,
        memory=memory, transcript_ingest_fn=transcript_ingest_fn, checkpointer=checkpointer,
        filing_ingest_fn=filing_ingest_fn, transcript_inventory_fn=transcript_inventory_fn,
    )
    # Seed loop caps from settings so the routers honor configured values (not their hardcoded
    # fallbacks). The planner sets depth/retries/editor_rounds counters; these are the ceilings.
    s = get_settings()
    depth_limit = s.tot_max_depth if analysis_depth is None else max(1, int(analysis_depth))
    initial = {"ticker": ticker, "question": question, "as_of": as_of,
               "use_source_cache": use_source_cache,
               "trajectory_id": thread_id,
               "max_depth": depth_limit, "editor_reretrieve_max": s.editor_reretrieve_max,
               "balance_repair_max": s.evidence_balance_retries,
               "report_repair_max": s.report_repair_retries,
               "human_revision_rounds": 0, "max_human_revisions": s.human_revision_max}
    config = graph_run_config(ticker=ticker, as_of=as_of, thread_id=thread_id)

    with run_tracing_context(
        ticker=ticker, question=question, as_of=as_of, thread_id=thread_id, settings=s
    ):
        result = invoke_with_progress(
            app, initial, config=config, on_progress=on_progress
        )
        runtime_profiles = [result.get("runtime_profile")]

        # HITL: if paused at the review interrupt, ask the reviewer, then resume. `interrupt_payload`
        # normalizes the LangGraph interrupt objects into the review payload expected by the UI/CLI.
        steps = 0
        while (payload := interrupt_payload(result)) is not None and steps < 5:
            steps += 1
            if review_queue is not None:
                review_queue.enqueue(ReviewItem(
                    trajectory_id=str(payload.get("trajectory_id", thread_id)),
                    brief=payload.get("brief", {}),
                    suspicion=float(payload.get("suspicion", 0.0) or 0.0),
                    reason=str(payload.get("reason", "")),
                ))
            if decide is None:
                # Fail closed: return the paused graph state. A caller can display the payload and
                # resume through the same checkpointer/thread after an explicit decision.
                break
            decision = decide(payload)
            if review_queue is not None:
                verdict = (
                    decision.get("decision", "allow") if isinstance(decision, dict) else str(decision)
                )
                note = decision.get("note", "") if isinstance(decision, dict) else ""
                review_queue.resolve(str(payload.get("trajectory_id", thread_id)), verdict, note)
            result = invoke_with_progress(
                app, Command(resume=decision), config=config, on_progress=on_progress
            )
            runtime_profiles.append(result.get("runtime_profile"))
    result["runtime_profile"] = merge_runtime_profiles(*runtime_profiles)
    return result
