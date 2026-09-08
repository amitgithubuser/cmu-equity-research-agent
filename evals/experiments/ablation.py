"""Ablation study (architecture §11.2, implementation-plan §11.3, Lab 6.1).

Run the guardrail matrix `{no_guardrails, pre_only, during_only, post_only, all}` over the gold briefs
and show each guardrail produces a **measurable drop in hallucination / overconfidence** — i.e. it
*earns its cost*. On a live run these are **LangSmith Experiments**; offline the matrix is deterministic.

Each configuration decides, for each brief, whether an unsafe brief would be **caught** (blocked /
routed back / flagged) by the guardrails that are ENABLED in that configuration:
  * `pre`     (input_guard)      — would catch an advice query before generation.
  * `during`  (confidence_guard) — catches overconfident prose (WARN/FAIL).
  * `post`    (source_guard)     — catches an unsupported/hallucinated claim (groundedness < target).

The headline metric per config is the **residual** hallucination / overconfidence rate: the fraction
of unsafe briefs that would slip through *because the relevant guard was off*.
"""

from __future__ import annotations

from equity_research.config import get_settings
from equity_research.guardrails.input_guard import detect_advice

from ..datasets import GOLD_BRIEFS
from ..evaluators import groundedness, overconfidence_verdict

# Each config maps to which of the three guard STAGES is enabled, mirroring the live graph:
#   pre    = input_guard    (catches an advice query BEFORE generation)
#   during = confidence_guard (catches overconfident prose)
#   post   = source_guard   (catches an unsupported / hallucinated claim)
CONFIGS = {
    "no_guardrails": {"pre": False, "during": False, "post": False},
    "pre_only": {"pre": True, "during": False, "post": False},
    "during_only": {"pre": False, "during": True, "post": False},
    "post_only": {"pre": False, "during": False, "post": True},
    "all": {"pre": True, "during": True, "post": True},
}


def _brief_is_ungrounded(brief: dict, embeddings=None) -> bool:
    return groundedness(brief, embeddings) < get_settings().groundedness_target


def _brief_is_overconfident(brief: dict) -> bool:
    return overconfidence_verdict(brief) in ("WARN", "FAIL")


def _brief_is_advice(entry: dict) -> bool:
    """True if the run ORIGINATED from an advice-seeking question the pre-guard should catch."""
    return detect_advice(entry.get("question", ""))


def run_ablation(briefs: list[dict] | None = None, embeddings=None) -> dict:
    """Return `{config: {n, missed_*_rate, caught}}` over the gold set.

    For each brief we know its true failure modes (ungrounded / overconfident / advice-origin). A
    failure "slips through" a configuration when the failure exists but the guard STAGE that would
    catch it is OFF. The `pre` stage is what `no_guardrails` and `pre_only` differ by — before this
    fix the loop ignored `cfg["pre"]` entirely, so those two columns were identical (F-09).
    """
    briefs = briefs or GOLD_BRIEFS
    results: dict[str, dict] = {}

    # pre-compute each brief's true failure modes once
    facts = []
    for entry in briefs:
        b = entry["brief"]
        facts.append({"ungrounded": _brief_is_ungrounded(b, embeddings),
                      "overconfident": _brief_is_overconfident(b),
                      "advice": _brief_is_advice(entry)})

    for name, cfg in CONFIGS.items():
        missed_hall = missed_over = missed_advice = caught = 0
        for f in facts:
            # a failure "slips through" if it exists but the guard that would catch it is OFF
            hall_slips = f["ungrounded"] and not cfg["post"]
            over_slips = f["overconfident"] and not cfg["during"]
            advice_slips = f["advice"] and not cfg["pre"]
            if hall_slips:
                missed_hall += 1
            if over_slips:
                missed_over += 1
            if advice_slips:
                missed_advice += 1
            any_failure = f["ungrounded"] or f["overconfident"] or f["advice"]
            if any_failure and not (hall_slips or over_slips or advice_slips):
                caught += 1
        n = len(facts)
        results[name] = {
            "n": n,
            "missed_hallucination_rate": round(missed_hall / n, 4) if n else 0.0,
            "missed_overconfidence_rate": round(missed_over / n, 4) if n else 0.0,
            "missed_advice_rate": round(missed_advice / n, 4) if n else 0.0,
            "caught": caught,
        }
    return results


def format_ablation(results: dict) -> str:
    """A small text table for the demo slide / report (architecture §11.2)."""
    rows = ["config          | miss_halluc | miss_overconf | miss_advice | caught",
            "----------------|-------------|---------------|-------------|-------"]
    for name, r in results.items():
        rows.append(f"{name:<15} | {r['missed_hallucination_rate']:>11} | "
                    f"{r['missed_overconfidence_rate']:>13} | {r['missed_advice_rate']:>11} | "
                    f"{r['caught']:>6}")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover - manual run for the demo slide
    print(format_ablation(run_ablation()))
