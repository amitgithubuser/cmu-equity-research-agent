"""Real node provider for the twelve-node application graph.

The Researcher and one LangGraph ToolNode form the standard model/action/observation loop. Thin trace
markers from the original graph have been folded into the meaningful step that owns their behavior:
research completeness lives in `evidence_check`, thesis gating and top-k pruning live in `critic`, and
the three post-generation guards live in `quality_gate`.

The two LLMs, the tool allow-list, the retriever, and the embedder are all **injected**, honoring the
"mockable LLM boundary" (guiding principle #4): on a real run they are `get_llm(...)` + `TOOLS` +
`retrieve` + `get_embeddings()`; in a test they are `FakeLLM(...)` + fake tools, so the full graph runs
end-to-end offline with zero tokens.

The lower-level evidence executor remains reusable and independently tested. The ToolNode invokes it
through four allow-listed research actions: numeric, filing/transcript text, news, and ratings.
"""

from __future__ import annotations

from functools import partial

from .agents import (
    analyst_node,
    critic_node,
    editor_node,
    planner_node,
)
from .agents.researcher import build_research_tool_node, researcher_agent_node
from .guardrails import input_guard_node, quality_gate_node
from .observability import human_review_node
from .observability.review_queue import apply_decision
from .tot import llm_batch_scorer, llm_generator
from .window import resolve_analysis_window


def source_setup_node(
    state: dict,
    tools: dict,
    transcript_ingest_fn=None,
    filing_ingest_fn=None,
    transcript_inventory_fn=None,
) -> dict:
    """Resolve one time window, then index mandatory transcripts and relevant SEC filings."""
    window = resolve_analysis_window(state.get("question", ""), state.get("as_of", ""))
    out = {"analysis_window": window.to_dict()}
    uncertainties: list[str] = []
    logs: list[str] = []
    use_cache = bool(state.get("use_source_cache", False))

    # Earnings transcripts are attempted every run, using the exact requested quarter window.
    fetch_tool = tools.get("transcript_fetch")
    transcript_cache_ready = False
    if use_cache:
        try:
            if transcript_inventory_fn is None:
                from .rag.ingest import transcript_inventory

                cached = transcript_inventory(
                    state.get("ticker", ""), n_quarters=window.n_quarters, as_of=window.cutoff,
                )
            else:
                cached = transcript_inventory_fn(
                    state.get("ticker", ""), n_quarters=window.n_quarters, as_of=window.cutoff,
                )
            cached_periods = list(cached.get("periods", []) or [])[:window.n_quarters]
            transcript_cache_ready = len(cached_periods) >= window.n_quarters
            if transcript_cache_ready:
                out.update({
                    "transcript_status": "cached",
                    "transcript_periods": cached_periods,
                    "transcript_chunks": int(cached.get("chunks", 0) or 0),
                })
                logs.append(
                    f"source_setup: reused {len(cached_periods)} cached transcript periods"
                )
        except Exception:  # noqa: BLE001 - cache miss falls through to live refresh
            transcript_cache_ready = False

    if transcript_cache_ready:
        pass
    elif fetch_tool is None:
        out.update({
            "transcript_status": "unavailable",
            "transcript_error": "transcript_fetch tool is not configured",
        })
        uncertainties.append("earnings-call transcripts were not available")
        logs.append(f"source_setup: window={window.label}; transcript tool unavailable")
    else:
        try:
            if transcript_ingest_fn is None:
                from .rag.ingest import ingest_transcript_window

                transcript_report = ingest_transcript_window(
                    state.get("ticker", ""), n_quarters=window.n_quarters,
                    as_of=window.cutoff, fetch_tool=fetch_tool,
                )
            else:
                transcript_report = transcript_ingest_fn(
                    state.get("ticker", ""), n_quarters=window.n_quarters,
                    as_of=window.cutoff, fetch_tool=fetch_tool,
                )
            count = int(transcript_report.get("transcripts", 0) or 0)
            complete = count >= window.n_quarters
            out.update({
                "transcript_status": "ingested" if complete else "partial",
                "transcript_periods": transcript_report.get("periods", []),
                "transcript_chunks": int(transcript_report.get("chunks", 0) or 0),
            })
            logs.append(
                f"source_setup: window={window.label}; transcripts={count}/{window.n_quarters} "
                f"chunks={transcript_report.get('chunks', 0)}"
            )
            if not complete:
                uncertainties.append(
                    f"only {count} of {window.n_quarters} required earnings-call transcripts were available"
                )
        except Exception as exc:  # noqa: BLE001 - source failure is routed to review, not hidden
            cached = {}
            if transcript_inventory_fn is not None or transcript_ingest_fn is None:
                try:
                    if transcript_inventory_fn is None:
                        from .rag.ingest import transcript_inventory

                        cached = transcript_inventory(
                            state.get("ticker", ""),
                            n_quarters=window.n_quarters,
                            as_of=window.cutoff,
                        )
                    else:
                        cached = transcript_inventory_fn(
                            state.get("ticker", ""),
                            n_quarters=window.n_quarters,
                            as_of=window.cutoff,
                        )
                except Exception:  # noqa: BLE001 - a cache miss must not hide the refresh failure
                    cached = {}
            cached_periods = list(cached.get("periods", []) or [])[:window.n_quarters]
            if cached_periods:
                out.update({
                    "transcript_status": "cached",
                    "transcript_periods": cached_periods,
                    "transcript_chunks": int(cached.get("chunks", 0) or 0),
                    "transcript_error": f"{type(exc).__name__}: live transcript refresh unavailable",
                })
                uncertainties.append(
                    "live transcript refresh failed; previously indexed transcripts were used"
                )
                logs.append(
                    f"source_setup: window={window.label}; live refresh failed; "
                    f"cached transcripts={len(cached_periods)}"
                )
            else:
                out.update({
                    "transcript_status": "unavailable",
                    # Never serialize exception text because clients can include secret-bearing URLs.
                    "transcript_error": f"{type(exc).__name__}: transcript source unavailable",
                })
                uncertainties.append("required earnings-call transcript fetch failed")
                logs.append(
                    f"source_setup: window={window.label}; transcript fetch failed "
                    f"({type(exc).__name__})"
                )

    # SEC filings are also automatic. Missing filings remain visible but do not hide transcript work.
    filing_tool = tools.get("edgar_fetch")
    filing_cache_ready = False
    if use_cache:
        try:
            from .rag.ingest import filing_inventory

            cached_filings = filing_inventory(
                state.get("ticker", ""), n_quarters=window.n_quarters, as_of=window.cutoff,
            )
            form_counts = cached_filings.get("form_counts", {}) or {}
            required_q = max(1, min(window.n_quarters, 4))
            filing_cache_ready = (
                int(form_counts.get("10-K", 0) or 0) >= 1
                and int(form_counts.get("10-Q", 0) or 0) >= required_q
            )
            if filing_cache_ready:
                out.update({
                    "filing_status": "cached",
                    "filing_forms": cached_filings.get("forms", []),
                    "filing_periods": cached_filings.get("periods", []),
                    "filing_chunks": int(cached_filings.get("chunks", 0) or 0),
                })
                logs.append(
                    f"source_setup: reused {cached_filings.get('filings', 0)} cached filings"
                )
        except Exception:  # noqa: BLE001 - cache miss falls through to live refresh
            filing_cache_ready = False

    if filing_cache_ready:
        pass
    elif filing_tool is None:
        out.update({"filing_status": "unavailable", "filing_error": "edgar_fetch tool is not configured"})
        uncertainties.append("SEC filings were not available")
        logs.append("source_setup: EDGAR tool unavailable")
    else:
        try:
            if filing_ingest_fn is None:
                from .rag.ingest import ingest_filing_window

                filing_report = ingest_filing_window(
                    state.get("ticker", ""), n_quarters=window.n_quarters, fetch_tool=filing_tool,
                )
            else:
                filing_report = filing_ingest_fn(
                    state.get("ticker", ""), n_quarters=window.n_quarters, fetch_tool=filing_tool,
                )
            filing_count = int(filing_report.get("filings", 0) or 0)
            out.update({
                "filing_status": "ingested" if filing_count else "partial",
                "filing_forms": filing_report.get("forms", []),
                "filing_periods": filing_report.get("periods", []),
                "filing_chunks": int(filing_report.get("chunks", 0) or 0),
            })
            logs.append(
                f"source_setup: filings={filing_count} chunks={filing_report.get('chunks', 0)}"
            )
            if not filing_count:
                uncertainties.append("no SEC filing could be indexed")
        except Exception as exc:  # noqa: BLE001 - keep provider details out of state and traces
            out.update({
                "filing_status": "unavailable",
                "filing_error": f"{type(exc).__name__}: SEC source unavailable",
            })
            uncertainties.append("required SEC filing fetch failed")
            logs.append(f"source_setup: filing fetch failed ({type(exc).__name__})")

    if uncertainties:
        out["open_uncertainties"] = uncertainties
    out["trajectory"] = [{
        "action": "source_setup",
        "args": {"ticker": state.get("ticker", ""), "window": window.label},
        "observation": (
            f"transcripts={out.get('transcript_status')}; filings={out.get('filing_status')}"
        ),
        "status": "ok" if out.get("transcript_status") in {"ingested", "cached"} else "refused",
    }]
    out["log"] = logs
    return out


# --------------------------------------------------------------------------- control and thesis helpers
def evidence_check_node(state: dict) -> dict:
    """Compare the plan with gathered task ids and decide whether research has a gap."""
    covered = set(state.get("covered", []) or [])
    covered -= set(state.get("reopen", []) or [])
    pending = [
        task.get("id") or task.get("claim", "")
        for task in (state.get("plan", []) or [])
        if (task.get("id") or task.get("claim", "")) not in covered
    ]
    under_cap = state.get("retries", 0) < state.get("max_retries", len(pending) + 2)
    need_more = bool(pending) and under_cap
    out = {
        "need_more": need_more,
        "log": [f"evidence_check: pending={pending} need_more={need_more}"],
        "trajectory": [{
            "action": "evidence_check",
            "args": {"pending": pending},
            "observation": "continue research" if need_more else "research complete",
            "status": "ok" if not pending else ("ok" if under_cap else "refused"),
        }],
    }
    if pending and not under_cap:
        out["open_uncertainties"] = [
            "research retry limit reached before tasks completed: " + ", ".join(pending)
        ]
    if not pending:
        required = [task for task in (state.get("plan", []) or []) if task.get("required")]
        evidence = state.get("evidence", []) or []
        missing = [
            task for task in required
            if not any(item.get("source_task") == task.get("id") for item in evidence)
        ]
        gaps = [task.get("evidence_role", task.get("claim", "required evidence")) for task in missing]
        rounds = int(state.get("balance_repair_rounds", 0) or 0)
        maximum = int(state.get("balance_repair_max", 1) or 0)
        if missing and rounds < maximum and under_cap:
            ids = [task.get("id") for task in missing if task.get("id")]
            out.update({
                "balance_gaps": gaps,
                "balance_repair_rounds": rounds + 1,
                "reopen": ids,
                "need_more": bool(ids),
                "log": [f"evidence_check: repairing required roles={gaps}"],
            })
        else:
            out.update({"balance_gaps": gaps, "reopen": [], "need_more": False})
            if gaps:
                out["open_uncertainties"] = [
                    "required evidence unavailable after targeted retry: " + ", ".join(gaps)
                ]
    return out


def critic_cycle_node(state: dict, llm, scorer=None, batch_scorer=None) -> dict:
    """Run the independent Critic, including evidence gating/top-k, then advance BFS depth."""
    out = critic_node(state, llm=llm, scorer=scorer, batch_scorer=batch_scorer)
    depth = state.get("depth", 0) + 1
    out["depth"] = depth
    out["log"] = [*out.get("log", []), f"critic: completed depth {depth}"]
    return out


# --------------------------------------------------------------------------- finalize (honors HITL)
def finalize_node(s: dict) -> dict:
    """Produce the final `brief`, applying any human decision (§10.3/§10.4 fail-safe).

    allow  → publish the draft as the final brief.
    revise → normally routes back to the Planner; reaching finalize means the revision cap was met.
    block  → withhold the brief (return a refusal notice, not the unsafe content).
    """
    decision = apply_decision(s)
    draft = s.get("draft_brief", {}) or {}
    if decision.get("blocked"):
        withheld = {"ticker": s.get("ticker", ""), "as_of": s.get("as_of", ""),
                    "withheld": True, "reason": s.get("escalation_reason", "human review: blocked"),
                    "disclaimer": "This is research, not financial advice."}
        return {"brief": withheld, "blocked": True,
                "trajectory": [{
                    "action": "finalize", "args": {"decision": "block"},
                    "observation": "report withheld", "status": "refused",
                }],
                "log": [*decision["log"], "finalize: brief WITHHELD (blocked)"]}
    return {"brief": draft, "blocked": False,
            "revision_requested": decision.get("revision_requested", False),
            "trajectory": [{
                "action": "finalize", "args": {"decision": s.get("human_decision", "allow")},
                "observation": "report published", "status": "ok",
            }],
            "log": [*decision["log"], "finalize: brief published"]}


# --------------------------------------------------------------------------- provider
def build_real_nodes(
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
) -> dict:
    """Wire the real twelve-node map. All external dependencies are injected.

    `memory` is passed to the numeric research tool so verified findings persist across runs and
    drive the quarter-over-quarter comparison.
    """
    from .tools import TOOLS

    routing_llm = routing_llm or reasoning_llm
    tools = tools if tools is not None else TOOLS

    research_tools = build_research_tool_node(
        llm=reasoning_llm,
        tools=tools,
        retrieve_fn=retrieve_fn,
        memory=memory,
    )

    return {
        "input_guard": partial(input_guard_node, known_symbols=known_symbols),
        "source_setup": partial(source_setup_node, tools=tools,
                                transcript_ingest_fn=transcript_ingest_fn,
                                filing_ingest_fn=filing_ingest_fn,
                                transcript_inventory_fn=transcript_inventory_fn),
        "planner": partial(planner_node, llm=routing_llm),
        "researcher": partial(researcher_agent_node, llm=reasoning_llm),
        "tools": research_tools,
        "evidence_check": evidence_check_node,
        "thesis_analyst": partial(analyst_node, llm=reasoning_llm, generator=llm_generator),
        "critic": partial(
            critic_cycle_node, llm=reasoning_llm, batch_scorer=llm_batch_scorer
        ),
        "editor": partial(editor_node, llm=reasoning_llm),
        "quality_gate": partial(quality_gate_node, embeddings=embeddings),
        "human_review": human_review_node,
        "finalize": finalize_node,
    }
