"""CLI entry point (implementation-plan §12, Phase 12 interface: "Streamlit + CLI core").

`equity-research NVDA --question "How are margins trending?"` runs one ticker end-to-end against the
**real services** the capstone targets (Claude + Chroma + yfinance/EDGAR), prints the cited brief, and
drops to an interactive prompt if the HITL gate fires.

The heavy, real dependencies are imported lazily inside `main` so this module imports offline (the test
suite imports `cli` without `.[real]` installed). The orchestration itself lives in `run.py`; the CLI
only wires real services and formats output. Complete cached transcripts and filings are reused by
default; ``--refresh-sources`` forces a new source-document refresh.
"""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from .config import get_settings


def _cli_decider(payload: dict) -> dict:
    """Interactive HITL reviewer: surface the flags and read a decision from the terminal (§10.3)."""
    print("\n" + "=" * 64)
    print("  ⚠  HUMAN REVIEW REQUIRED (monitor escalation)")
    print("=" * 64)
    print(f"  ticker     : {payload.get('ticker')}")
    print(f"  suspicion  : {payload.get('suspicion')}")
    print(f"  confidence : {payload.get('confidence')}")
    print(f"  reason     : {payload.get('reason')}")
    print(f"  flags      : {payload.get('flags')}")
    choice = ""
    while choice not in {"allow", "revise", "block"}:
        choice = input("  decision [allow/revise/block]: ").strip().lower() or "allow"
    return {"decision": choice, "note": "cli reviewer"}


def _cli_auto_allow(payload: dict) -> dict:
    """Explicit batch-mode policy selected only by the `--no-hitl` command-line flag."""
    return {"decision": "allow", "note": "explicit CLI --no-hitl batch policy"}


def _format_brief(brief: dict) -> str:
    """Use the same professional, source-numbered report in the CLI and Streamlit UI."""
    from .ui.presenter import brief_markdown

    return "\n" + brief_markdown(brief) + "\n"


def _format_runtime_profile(profile: dict, service_setup_seconds: float = 0.0) -> str:
    from .ui.presenter import runtime_profile_rows

    total = float(profile.get("total_seconds", 0.0) or 0.0) + service_setup_seconds
    lines = [f"[performance] total={total:.1f}s"]
    for row in runtime_profile_rows(profile, service_setup_seconds):
        lines.append(f"  {row['Stage']}: {row['Seconds']:.1f}s ({row['Share']})")
    return "\n".join(lines)


def _build_real_services():
    """Assemble the real reasoning/routing LLMs, tools, retriever, and long-term memory.

    Imported lazily so the module stays offline-importable; requires `pip install -e .[real]`.
    """
    from .llm import get_llm
    from .memory import get_longterm_store
    from .rag.embeddings import get_embeddings
    from .rag.retrieve import retrieve
    from .tools import TOOLS

    settings = get_settings()
    embeddings = get_embeddings(settings)

    return {
        "reasoning_llm": get_llm("reasoning"),
        "routing_llm": get_llm("routing"),
        "tools": TOOLS,
        "retrieve_fn": lambda q, company, period=None, document_type=None,
                              transcript_periods=None: retrieve(
            q, company, period, document_type=document_type,
            transcript_periods=transcript_periods, settings=settings
        ),
        "embeddings": embeddings,
        "memory": get_longterm_store(settings),   # durable QoQ record across CLI runs (§6.1)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="equity-research",
        description="Ticker-agnostic equity research assistant — cited bull/bear/risk briefs "
                    "(research, not financial advice).")
    parser.add_argument("ticker", help="US ticker symbol, e.g. NVDA")
    parser.add_argument("-q", "--question", default="",
                        help="Research question (default: general research)")
    parser.add_argument("--as-of", default="", help="Reporting-period anchor, e.g. FY2025")
    parser.add_argument("--no-hitl", action="store_true",
                        help="Auto-allow escalations instead of prompting (batch mode)")
    parser.add_argument("--refresh-sources", action="store_true",
                        help="Refresh transcripts and filings instead of reusing complete local cache")
    args = parser.parse_args(argv)

    # import the orchestrator lazily too, so a bad .[real] install fails with a clear message here
    from .run import run_analysis

    service_started = perf_counter()
    try:
        services = _build_real_services()
    except ImportError as exc:  # pragma: no cover - real-deps path
        print(f"error: real services require `pip install -e .[real]` — {exc}", file=sys.stderr)
        return 2
    service_setup_seconds = perf_counter() - service_started

    decide = _cli_auto_allow if args.no_hitl else _cli_decider
    result = run_analysis(
        args.ticker.strip().upper(), args.question,
        as_of=args.as_of, use_source_cache=not args.refresh_sources,
        decide=decide, thread_id=f"cli-{args.ticker.strip().upper()}",
        **services,
    )

    if not result.get("proceed", True):
        print(f"\n⛔ input rejected: {result.get('escalation_reason')} — {result.get('issues')}\n",
              file=sys.stderr)
        return 1

    print(_format_brief(result.get("brief", {})))
    window = result.get("analysis_window", {}) or {}
    print(f"[sources] window={window.get('label', 'not resolved')} "
          f"transcripts={result.get('transcript_status', 'not checked')} "
          f"periods={result.get('transcript_periods', [])} "
          f"filings={result.get('filing_status', 'not checked')} "
          f"forms={result.get('filing_forms', [])}")
    print(_format_runtime_profile(result.get("runtime_profile", {}), service_setup_seconds))
    if result.get("needs_review") or result.get("suspicion"):
        report = result.get("suspicion_report", {}) or {}
        print(f"[monitor] suspicion={result['suspicion']} needs_review={result.get('needs_review')} "
              f"fired={report.get('fired', [])} reason={result.get('escalation_reason', '')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
