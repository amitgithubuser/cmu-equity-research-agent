"""Guarded LangSmith adapter (F-09H; architecture §11, implementation-plan §11.2).

The review (F-09) said the evaluation layer "is not connected to LangSmith … no LangSmith client,
dataset creation, evaluator registration, [or] experiment execution." This module is that connection —
**guarded** so it never breaks the offline build:

  * If the `langsmith` package is missing OR no `LANGSMITH_API_KEY` is configured, every entry point
    returns a `{"status": "offline", ...}` report and touches no network. That is the CI path, and it
    is why importing this module costs zero tokens and never fails a keyless run.
  * If both are present, the SAME gold set (`GOLD_BRIEFS`) is pushed as a **LangSmith Dataset**, the
    SAME pure-Python evaluators (`evals.evaluators`) are wrapped as **LangSmith evaluators**, and an
    **experiment** is run over the dataset. Offline and live share one source of truth — the live path
    is a thin wrapper over the deterministic functions the tests already exercise, not a parallel
    implementation.

Design choice (D): keep the adapter a *thin, optional* seam rather than threading a LangSmith client
through the evaluators. The evaluators stay pure functions (CI-safe, unit-tested); this file is the
only place that imports `langsmith`, so the heavy/hosted dependency stays isolated behind one guard.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from equity_research.config import get_settings

from .datasets import GOLD_BRIEFS
from .evaluators import evaluate_brief, rag_triad


def _client() -> Any | None:
    """Return a live LangSmith `Client`, or `None` if unavailable (missing lib OR no key).

    This single guard is what makes every public function degrade gracefully offline: callers check
    for `None` and return an `{"status": "offline"}` report instead of raising. No network is touched
    on the `None` path.
    """
    key = get_settings().langsmith_api_key or os.environ.get("LANGSMITH_API_KEY", "")
    if not key:
        return None
    try:
        from langsmith import (
            Client,  # imported lazily so the package is only needed on the live path
        )
    except ModuleNotFoundError:
        return None
    return Client(api_key=key)


def is_live() -> bool:
    """True only when a real LangSmith client can be constructed (lib installed AND key set)."""
    return _client() is not None


# --- evaluator wrapping ----------------------------------------------------------------------------
def as_langsmith_evaluators() -> list[Callable[..., dict]]:
    """Wrap the pure-Python evaluators as LangSmith-style `(run, example) -> {key, score}` callables.

    LangSmith calls each evaluator with the run's `outputs` (our brief) and the example's `outputs`
    (the reference). We reuse `evaluate_brief` + `rag_triad` verbatim, so the live scores are exactly
    the offline ones — no drift between what CI checks and what the dashboard shows.
    """
    def _brief_of(obj: Any) -> dict:
        # tolerate either a raw brief dict or a {"brief": {...}} envelope
        if isinstance(obj, dict) and "brief" in obj and isinstance(obj["brief"], dict):
            return obj["brief"]
        return obj if isinstance(obj, dict) else {}

    def groundedness_evaluator(run: Any, example: Any = None) -> dict:
        brief = _brief_of(getattr(run, "outputs", run))
        return {"key": "groundedness", "score": evaluate_brief(brief)["groundedness"]}

    def faithfulness_evaluator(run: Any, example: Any = None) -> dict:
        outputs = getattr(run, "outputs", run)
        inputs = getattr(run, "inputs", {}) or {}
        brief = _brief_of(outputs)
        question = (inputs.get("question", "") if isinstance(inputs, dict) else "")
        return {"key": "faithfulness", "score": rag_triad(brief, question)["faithfulness"]}

    return [groundedness_evaluator, faithfulness_evaluator]


# --- dataset + experiment --------------------------------------------------------------------------
def push_dataset(name: str = "equity-research-gold") -> dict:
    """Create/replace the gold set as a LangSmith Dataset (live) or report offline (keyless CI).

    Returns a small status dict either way, so callers and tests can assert on the shape without a key.
    """
    client = _client()
    if client is None:
        return {"status": "offline", "dataset": name, "n": len(GOLD_BRIEFS),
                "reason": "no LANGSMITH_API_KEY or langsmith package"}
    # live path (only runs when a key + the package are present):
    if client.has_dataset(dataset_name=name):  # pragma: no cover - requires a live key
        client.delete_dataset(dataset_name=name)
    ds = client.create_dataset(dataset_name=name, description="Equity Research gold briefs (F-09).")
    for entry in GOLD_BRIEFS:  # pragma: no cover - requires a live key
        client.create_example(
            inputs={
                "ticker": entry["brief"].get("ticker"),
                "question": entry.get("question", ""),
                "as_of": entry.get("as_of", ""),
            },
            outputs={"brief": entry["brief"], "reference": entry.get("reference", {})},
            dataset_id=ds.id,
        )
    return {"status": "live", "dataset": name, "n": len(GOLD_BRIEFS)}


def make_application_target(
    *,
    reasoning_llm,
    routing_llm=None,
    tools: dict | None = None,
    retrieve_fn=None,
    embeddings=None,
    memory=None,
    transcript_ingest_fn=None,
    filing_ingest_fn=None,
) -> Callable[[dict], dict]:
    """Build a LangSmith target that runs the real application for every dataset example."""
    from equity_research.run import run_analysis

    def target(inputs: dict) -> dict:
        state = run_analysis(
            str(inputs.get("ticker", "")),
            str(inputs.get("question", "")),
            as_of=str(inputs.get("as_of", "")),
            reasoning_llm=reasoning_llm,
            routing_llm=routing_llm,
            tools=tools,
            retrieve_fn=retrieve_fn,
            embeddings=embeddings,
            memory=memory,
            transcript_ingest_fn=transcript_ingest_fn,
            filing_ingest_fn=filing_ingest_fn,
            decide=lambda payload: {"decision": "allow", "note": "evaluation review record"},
            thread_id=f"eval-{inputs.get('ticker', 'case')}",
        )
        return {
            "brief": state.get("brief", {}),
            "analysis_window": state.get("analysis_window", {}),
            "quality_decision": state.get("quality_decision", ""),
            "source_coverage": (state.get("brief", {}) or {}).get("source_coverage", {}),
        }

    return target


def run_experiment(
    dataset_name: str = "equity-research-gold",
    prefix: str = "equity-research-eval",
    target: Callable[[dict], dict] | None = None,
) -> dict:
    """Run the evaluators over the dataset as a LangSmith experiment (live) or offline (CI).

    Offline (no key), we still produce a real report: we score every gold brief with the same
    evaluators LangSmith would call, so the experiment shape is meaningful in CI — it just runs locally.
    """
    client = _client()
    if client is None:
        results = [
            {"id": e["id"], **evaluate_brief(e["brief"]),
             **rag_triad(e["brief"], e.get("question", ""))}
            for e in GOLD_BRIEFS
        ]
        mean_grounded = round(sum(r["groundedness"] for r in results) / len(results), 4)
        return {"status": "offline", "n": len(results), "mean_groundedness": mean_grounded,
                "results": results}
    if target is None:  # pragma: no cover - requires a live key
        return {
            "status": "blocked",
            "dataset": dataset_name,
            "reason": "a real application target is required; empty placeholder outputs are forbidden",
        }
    from langsmith import evaluate as ls_evaluate  # pragma: no cover - requires a live key
    push_dataset(dataset_name)  # pragma: no cover
    ls_evaluate(  # pragma: no cover
        target,
        data=dataset_name,
        evaluators=as_langsmith_evaluators(),
        experiment_prefix=prefix,
    )
    return {"status": "live", "dataset": dataset_name, "experiment_prefix": prefix}  # pragma: no cover
