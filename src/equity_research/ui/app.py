"""Streamlit UI (Phase 12 — chosen interface: "Streamlit + CLI core", real services).

Run with:  `streamlit run src/equity_research/ui/app.py`  (needs `pip install -e .[real,ui]`).

The user types a ticker + question; the app runs the SAME `run.py` orchestrator the CLI uses and
renders the cited brief. The HITL gate (§10.3) is realized interactively: when the graph pauses at the
review interrupt, the app surfaces the flags and Allow/Revise/Block buttons, then resumes the SAME
thread through the checkpointer on the next Streamlit run — so the human actually decides in the UI.

Heavy imports are lazy/guarded so this file is importable offline (the presenter logic it relies on is
unit-tested in `test_ui.py`); the Streamlit-specific glue runs only under `streamlit run`.
"""

from __future__ import annotations

# Package-safe imports (F-10). `streamlit run src/equity_research/ui/app.py` executes this file as a
# top-level script with NO parent package, so the relative imports below raise ImportError. We try the
# normal package-relative form first (works under `python -m` / pytest / `streamlit run -m`), and fall
# back to putting the `src/` root on sys.path and importing absolutely — so the documented run command
# actually works instead of crashing on startup.
try:
    from ..config import get_settings
    from ..graph_engine import Command, interrupt_payload
    from ..observability.langsmith import graph_run_config, run_tracing_context
    from ..run import invoke_with_progress
    from ..text_cleaning import clean_public_text
    from .presenter import (
        brief_markdown,
        case_evaluation_sections,
        evidence_reference,
        evidence_source_rows,
        evidence_strength_label,
        financial_snapshot_rows,
        grouped_case_content,
        monitor_summary,
        progress_label,
        runtime_profile_rows,
        trajectory_lines,
        trend_change_rows,
        trends_chart_data,
    )
except ImportError:  # pragma: no cover - only taken when run as a bare script
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from equity_research.config import get_settings
    from equity_research.graph_engine import Command, interrupt_payload
    from equity_research.observability.langsmith import graph_run_config, run_tracing_context
    from equity_research.run import invoke_with_progress
    from equity_research.text_cleaning import clean_public_text
    from equity_research.ui.presenter import (
        brief_markdown,
        case_evaluation_sections,
        evidence_reference,
        evidence_source_rows,
        evidence_strength_label,
        financial_snapshot_rows,
        grouped_case_content,
        monitor_summary,
        progress_label,
        runtime_profile_rows,
        trajectory_lines,
        trend_change_rows,
        trends_chart_data,
    )


def _build_services():
    """Real reasoning/routing LLMs + tools + retriever + long-term memory (requires `.[real]`)."""
    from equity_research.llm import get_llm
    from equity_research.memory import get_longterm_store
    from equity_research.rag.embeddings import get_embeddings
    from equity_research.rag.retrieve import retrieve
    from equity_research.tools import TOOLS

    settings = get_settings()
    return {
        "reasoning_llm": get_llm("reasoning"),
        "routing_llm": get_llm("routing"),
        "tools": TOOLS,
        "retrieve_fn": lambda q, company, period=None, document_type=None,
                              transcript_periods=None: retrieve(
            q, company, period, document_type=document_type,
            transcript_periods=transcript_periods, settings=settings
        ),
        "embeddings": get_embeddings(settings),
        "memory": get_longterm_store(settings),   # JSON-backed (§6.1) → QoQ survives across UI sessions
    }


def main() -> None:
    from time import perf_counter

    import streamlit as st

    st.set_page_config(page_title="Equity Research Analyst", page_icon="📊", layout="wide")
    st.title("📊 Equity Research Analyst")
    st.caption("Detailed, cited company research with competing bull and bear theses.")

    with st.sidebar:
        st.header("Query")
        ticker = st.text_input("Ticker", value="NVDA").strip().upper()
        question = st.text_area("Question", value="Can you research NVDA?")
        as_of = st.text_input("As-of period (optional)", value="")
        refresh_sources = st.checkbox("Refresh transcript and filing cache", value=False)
        st.caption(
            "Cached filings/transcripts are reused when complete. Enable refresh when you need the "
            "newest source documents. Current market data and news are still refreshed each run."
        )
        go = st.button("Run analysis", type="primary")

    # session-scoped app so the HITL round-trip survives Streamlit reruns. A test seeds
    # `st.session_state.app` with a fake app first, so real services are never constructed offline.
    if "app" not in st.session_state:
        from equity_research.run import build_app

        service_started = perf_counter()
        try:
            services = _build_services()
        except ImportError as exc:  # pragma: no cover - only without `.[real,ui]`
            st.error(f"Real services require `pip install -e .[real,ui]` — {exc}")
            return
        st.session_state.app = build_app(**services)
        st.session_state.services = services
        st.session_state.pending_service_setup_seconds = perf_counter() - service_started

    app = st.session_state.app
    settings = get_settings()

    if go:
        st.session_state.thread = f"ui-{ticker}"
        progress = st.empty()

        def show_progress(node_name: str, update: dict) -> None:
            progress.info(progress_label(node_name, update))

        with st.spinner("Running the research workflow …"):
            config = graph_run_config(ticker=ticker, as_of=as_of, thread_id=st.session_state.thread)
            with run_tracing_context(
                ticker=ticker, question=question, as_of=as_of,
                thread_id=st.session_state.thread, settings=settings,
            ):
                result = invoke_with_progress(
                    app,
                    {"ticker": ticker, "question": question, "as_of": as_of,
                     "trajectory_id": st.session_state.thread,
                     "use_source_cache": not refresh_sources,
                     "max_depth": settings.tot_max_depth,
                     "editor_reretrieve_max": settings.editor_reretrieve_max,
                     "balance_repair_max": settings.evidence_balance_retries,
                     "report_repair_max": settings.report_repair_retries,
                     "human_revision_rounds": 0,
                     "max_human_revisions": settings.human_revision_max},
                    config=config,
                    on_progress=show_progress,
                )
        st.session_state.result = result
        st.session_state.result_service_setup_seconds = st.session_state.pop(
            "pending_service_setup_seconds", 0.0
        )
        progress.empty()

    result = st.session_state.get("result")
    if not result:
        st.info("Enter a ticker and question, then **Run analysis**.")
        return

    # --- HITL: paused at the review interrupt -> show the gate ---
    payload = interrupt_payload(result)
    if payload is not None:
        st.warning(
            f"Human review required. Suspicion score: {payload.get('suspicion')}. "
            f"Flags: {payload.get('flags')}"
        )
        st.caption(
            "Allow publishes the current report. Revise sends the reviewer note back through "
            "planning and research. Block withholds the report."
        )
        st.markdown(
            brief_markdown(
                payload.get("brief", {}), include_trends=False, include_sources=False
            )
        )
        review_note = st.text_area(
            "Reviewer note",
            placeholder="Example: add a cited counterpoint on customer concentration.",
        )
        c1, c2, c3 = st.columns(3)
        decision = None
        if c1.button("✅ Allow"):
            decision = "allow"
        if c2.button("✏️ Revise"):
            decision = "revise"
        if c3.button("⛔ Block"):
            decision = "block"
        if decision:
            with st.spinner("Applying decision …"):
                ticker = payload.get("ticker", ticker)
                config = graph_run_config(
                    ticker=ticker, as_of=as_of, thread_id=st.session_state.thread
                )
                with run_tracing_context(
                    ticker=ticker, question=question, as_of=as_of,
                    thread_id=st.session_state.thread, settings=settings,
                ):
                    prior_profile = (st.session_state.result or {}).get("runtime_profile")
                    resumed = invoke_with_progress(
                        app,
                        Command(resume={"decision": decision, "note": review_note}),
                        config=config,
                    )
                    from equity_research.run import merge_runtime_profiles

                    resumed["runtime_profile"] = merge_runtime_profiles(
                        prior_profile, resumed.get("runtime_profile")
                    )
                    st.session_state.result = resumed
            st.rerun()
        return

    # --- final brief ---
    brief = result.get("brief", {})
    col_main, col_side = st.columns([3, 1])
    with col_main:
        st.markdown(
            brief_markdown(
                brief, include_trends=False, include_sources=False,
                part="overview", include_financial_text=False,
            ),
            unsafe_allow_html=False,
        )
        _render_financial_snapshot(st, brief)
        _render_trends(st, brief)
        _render_investment_cases(st, brief)
        st.markdown(
            brief_markdown(
                brief, include_header=False, include_trends=False, include_sources=False,
                part="closing",
            )
        )
        _render_sources(st, brief)
    with col_side:
        _render_monitor(st, result, settings)
        _render_runtime_profile(
            st, result, float(st.session_state.get("result_service_setup_seconds", 0.0) or 0.0)
        )
    with st.expander("Trajectory / trace"):
        for line in trajectory_lines(result):
            st.text(line)


def _render_trends(st, brief: dict) -> None:  # pragma: no cover - Streamlit glue (logic tested via presenter)
    """Draw revenue/growth and margin charts directly inside Financial analysis."""
    import altair as alt
    import pandas as pd

    data = trends_chart_data(brief)
    if not data["periods"] or (not data["margins"] and not data["levels"]):
        return
    periods = data["periods"]
    if data["levels"]:
        st.markdown("#### Revenue and growth")
        revenue = data["levels"].get("Revenue", [None] * len(periods))
        qoq = data.get("growth", {}).get("Revenue growth (QoQ)", [None] * len(periods))
        frame = pd.DataFrame({
            "Period": periods,
            "Revenue ($B)": [_to_billions(value) for value in revenue],
            "QoQ growth (%)": [value * 100 if isinstance(value, (int, float)) else None for value in qoq],
        })
        bars = alt.Chart(frame).mark_bar(color="#4C78A8", opacity=0.75).encode(
            x=alt.X("Period:N", sort=None, title=None),
            y=alt.Y("Revenue ($B):Q", title="Revenue ($B)"),
            tooltip=["Period:N", alt.Tooltip("Revenue ($B):Q", format=".1f")],
        )
        line = alt.Chart(frame).mark_line(color="#F58518", point=True, strokeWidth=3).encode(
            x=alt.X("Period:N", sort=None),
            y=alt.Y("QoQ growth (%):Q", title="QoQ growth (%)"),
            tooltip=["Period:N", alt.Tooltip("QoQ growth (%):Q", format=".1f")],
        )
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), use_container_width=True)
    if data["margins"]:
        st.markdown("#### Margin progression")
        frame = pd.DataFrame(data["margins"], index=periods)
        frame.index.name = "Period"
        long = frame.reset_index().melt("Period", var_name="Metric", value_name="Percent")
        long["Percent"] = long["Percent"] * 100
        chart = alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Period:N", sort=None, title=None),
            y=alt.Y("Percent:Q", title="Percent", scale=alt.Scale(zero=False)),
            color=alt.Color("Metric:N", title=None),
            tooltip=["Period:N", "Metric:N", alt.Tooltip("Percent:Q", format=".1f")],
        )
        st.altair_chart(chart, use_container_width=True)

    changes = trend_change_rows(brief)
    if changes:
        st.markdown("#### Latest movement")
        st.dataframe(changes, use_container_width=True, hide_index=True)


def _render_financial_snapshot(st, brief: dict) -> None:  # pragma: no cover - Streamlit glue
    """Show the calculated metrics before the trend charts."""
    rows = financial_snapshot_rows(brief)
    if not rows:
        return
    st.markdown("#### Latest snapshot")
    columns = st.columns(min(3, len(rows)))
    for index, row in enumerate(rows):
        with columns[index % len(columns)]:
            st.metric(row["Metric"], row["Value"])
            st.caption(f"{row['Period']} · {row['Cross-check']}".strip(" ·"))
            if row.get("Citation"):
                st.markdown(f"Source: {row['Citation']}")


def _render_investment_cases(st, brief: dict) -> None:  # pragma: no cover - Streamlit glue
    """Render compact Bull/Bear/Risk tabs with evidence grouped by business theme."""
    st.markdown("### Investment cases")
    st.info(
        "Evidence score = the independent Critic’s equally weighted assessment of source support, "
        "consistency, materiality, and resilience. 0.60 is the minimum passing score. It is not a "
        "probability, return forecast, or price target."
    )
    theses = brief.get("theses", []) or []
    tabs = st.tabs(["🟢 Bull case", "🔴 Bear case", "🟠 Key risks"])
    for tab, (side, fallback_key) in zip(
        tabs, (("bull", "bull_case"), ("bear", "bear_case"), ("risk", "key_risks"))
    ):
        with tab:
            branches = [branch for branch in theses if branch.get("side") == side]
            if not branches:
                fallback = brief.get(fallback_key, []) or []
                if not fallback:
                    st.warning("Insufficient cited evidence — this case was not asserted.")
                for evidence in fallback:
                    st.markdown(
                        f"{evidence.get('claim', '')} {evidence_reference(brief, evidence)}".strip()
                    )
                continue

            for index, branch in enumerate(branches, 1):
                title = f"Angle {index}" if len(branches) > 1 else "Primary thesis"
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    st.markdown(
                        f"**{clean_public_text(branch.get('thesis') or 'Grounded thesis')}**"
                    )
                    score = float(branch.get("support_score", 0.0) or 0.0)
                    score_col, horizon_col = st.columns([1, 2])
                    with score_col:
                        st.metric(
                            "Evidence score",
                            f"{score:.2f} / 1.00",
                            evidence_strength_label(score),
                            delta_color="off",
                        )
                        st.progress(max(0.0, min(1.0, score)))
                    with horizon_col:
                        if branch.get("mechanism"):
                            st.markdown(
                                f"**How it works**  \n{clean_public_text(branch['mechanism'])}"
                            )
                        if branch.get("time_horizon"):
                            st.caption(f"Evaluation horizon: {branch['time_horizon']}")

                    for group in grouped_case_content(branch):
                        with st.expander(group["theme"], expanded=group["theme"] in {
                            "Margins & profitability", "Growth & demand",
                        }):
                            for item in group["evidence"]:
                                evidence = item["evidence"]
                                st.markdown(
                                    f"**Evidence:** {clean_public_text(evidence.get('claim', ''))} "
                                    f"{evidence_reference(brief, evidence)}".strip()
                                )
                                if item["rationale"]:
                                    st.caption(
                                        f"Why it matters: {clean_public_text(item['rationale'])}"
                                    )

                    evaluation = case_evaluation_sections(branch)
                    if evaluation:
                        st.markdown("##### How to evaluate this thesis")
                        st.caption(
                            "These conditions apply to the thesis as a whole, rather than to any "
                            "one evidence category."
                        )
                        for section in evaluation:
                            st.markdown(
                                f"**{section['heading']}.** " + " ".join(section["sentences"])
                            )


def _render_monitor(st, state: dict, settings) -> None:  # pragma: no cover - Streamlit glue
    """Explain the weighted suspicion score in reader language instead of raw JSON."""
    summary = monitor_summary(state)
    score = float(summary["suspicion"] or 0.0)
    threshold = float(settings.escalation_threshold)
    st.subheader("Quality monitor")
    status = "Review required" if summary["needs_review"] else "Below review threshold"
    st.metric("Suspicion score", f"{score:.0f} / 100", status)
    st.caption(
        f"This is a weighted warning score, not an error probability. A score of 25 usually means "
        f"one medium-severity signal fired. Automatic review begins at {threshold:.0f}, while a hard "
        "safety trigger requires review at any score."
    )
    signal_text = {
        "narrow_support_gap": "Bull and bear evidence scores are close, so the conclusion is less decisive.",
        "vendor_divergence": "A calculated metric differs from the vendor cross-check.",
        "source_conflict": "A severe calculated-versus-vendor conflict requires review.",
        "uncited_claim": "A factual claim lacks sufficient citation support.",
        "stale_period": "A numeric claim may not match the requested reporting period.",
        "injection_marker": "Source text contained possible instruction-like content.",
    }
    fired = summary.get("fired", []) or []
    if not fired:
        st.success("No behavioral warning signal fired.")
    else:
        for signal in fired:
            st.warning(signal_text.get(signal, signal.replace("_", " ").title()))
    if summary.get("review_triggers"):
        st.error("Review triggers: " + ", ".join(summary["review_triggers"]))
    with st.expander("Source coverage"):
        st.write(f"Analysis window: {summary['analysis_window']}")
        st.write(f"Transcripts: {summary['transcript_status']} · {summary['transcript_periods']}")
        st.write(f"Filings: {summary['filing_status']} · {summary['filing_forms']}")
        st.write(f"Quality decision: {summary['quality_decision']}")


def _render_runtime_profile(
    st, state: dict, service_setup_seconds: float = 0.0
) -> None:  # pragma: no cover - Streamlit glue
    profile = state.get("runtime_profile", {}) or {}
    if not profile:
        return
    total = float(profile.get("total_seconds", 0.0)) + service_setup_seconds
    st.caption(f"Completed in {total:.1f}s")
    stages = runtime_profile_rows(profile, service_setup_seconds)
    if stages:
        with st.expander("Where execution time was spent"):
            st.dataframe(stages, use_container_width=True, hide_index=True)


def _to_billions(value):
    if not isinstance(value, (int, float)):
        return None
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        return float(value) / 1_000_000_000
    if magnitude >= 1_000:
        return float(value) / 1_000
    return float(value)


def _render_sources(st, brief: dict) -> None:  # pragma: no cover - Streamlit glue
    """Keep full snippets and source links available without crowding the main thesis."""
    rows = evidence_source_rows(brief)
    if not rows:
        return
    with st.expander(f"Sources and evidence ({len(rows)} cited findings)"):
        for i, row in enumerate(rows, 1):
            source = f"{row['Source']} · {row['Period']}".strip(" ·")
            if row["URL"]:
                source = f"[{source}]({row['URL']})"
            marker = f"[S{i}]({row['URL']})" if row["URL"] else f"[S{i}]"
            st.markdown(f"**{marker} {row['Claim']}**  \n{source}  \n> {row['Evidence']}")


if __name__ == "__main__":  # pragma: no cover
    main()
