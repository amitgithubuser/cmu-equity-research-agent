"""Calibration runner (architecture §11.3, implementation-plan §11.4, Lab 6.2).

Score every labeled run through the SAME `monitor.score_suspicion` the live graph uses, then hand the
`(suspicion, is_risky)` pairs to `calibrate_threshold` to pick the escalation cutoff from the ROC curve
(lowest threshold with FPR ≤ `target_fpr`). This closes **OQ-7**: the gate is configured by data.

Two things the review (F-09) asked for and this now does:
  * **Persist** the chosen cutoff. `run_calibration(persist=True)` writes the calibrated value back to
    the live `Settings` (and stages the env var) so the running graph — and the next process — uses the
    data-derived threshold instead of the hand-set placeholder. Previously it returned a value and
    wrote nothing.
  * **ECE calibration report.** `confidence_calibration` wires `expected_calibration_error` into a real
    experiment: it measures whether a brief's self-reported `confidence` tracks whether the brief is
    actually sound. This is the "stated confidence vs. correctness across a bucketed set" the evaluators
    table promises — ECE belongs in a set-level experiment, not the single-brief `evaluate_brief` bundle.
"""

from __future__ import annotations

import os

from equity_research.config import get_settings
from equity_research.observability.calibration import CalibrationResult, calibrate_threshold
from equity_research.observability.monitor import score_suspicion

from ..datasets import GOLD_BRIEFS, LABELED_RUNS
from ..evaluators import expected_calibration_error


def score_runs(runs: list[dict] | None = None) -> tuple[list[float], list[int]]:
    """Run each labeled example through the monitor → (suspicion_scores, labels)."""
    runs = runs or LABELED_RUNS
    scores, labels = [], []
    for r in runs:
        report = score_suspicion(r["brief"], as_of=r.get("as_of", ""),
                                 scored_branches=r.get("scored_branches"),
                                 evidence=r.get("evidence"))
        scores.append(report["suspicion"])
        labels.append(int(r["is_risky"]))
    return scores, labels


def apply_calibrated_threshold(threshold: float) -> float:
    """Persist the calibrated escalation threshold back into settings (F-09 write-back).

    Mutates the cached `Settings` singleton so the running process picks up the data-derived cutoff
    immediately, AND stages the matching env var so a fresh process (`get_settings.cache_clear()` then
    reload, or a real restart) starts with it too. Returns the value written.
    """
    os.environ["ESCALATION_THRESHOLD"] = str(threshold)
    get_settings().escalation_threshold = float(threshold)
    return float(threshold)


def run_calibration(
    runs: list[dict] | None = None,
    target_fpr: float | None = None,
    *,
    persist: bool = False,
) -> CalibrationResult:
    """Compute the data-driven escalation threshold (OQ-7); optionally persist it (F-09).

    `persist=True` writes the chosen threshold back to the live settings so the gate is actually
    configured by data end-to-end — not just reported. Off by default so tests/report runs can inspect
    the result without mutating global state unless they ask for it.
    """
    target_fpr = get_settings().target_fpr if target_fpr is None else target_fpr
    scores, labels = score_runs(runs)
    result = calibrate_threshold(scores, labels, target_fpr=target_fpr)
    if persist:
        apply_calibrated_threshold(result.threshold)
    return result


def confidence_calibration(entries: list[dict] | None = None) -> dict:
    """ECE of a brief's self-reported `confidence` vs. whether the brief is actually sound (F-09).

    "Sound" = the reference does not mark the brief risky. Returns `{ece, n}`; lower ECE means the
    agent's stated confidence is better calibrated (neither over- nor under-confident on average). This
    is the concrete experiment that consumes `expected_calibration_error`.
    """
    entries = entries or GOLD_BRIEFS
    confidences: list[float] = []
    correct: list[int] = []
    for e in entries:
        conf = e.get("brief", {}).get("confidence")
        if conf is None:
            continue
        confidences.append(float(conf))
        correct.append(0 if e.get("reference", {}).get("expect_risky") else 1)
    return {"ece": expected_calibration_error(confidences, correct), "n": len(confidences)}


if __name__ == "__main__":  # pragma: no cover - manual run for the report
    result = run_calibration()
    print(f"chosen escalation_threshold = {result.threshold}  "
          f"(AUC={result.auc}, FPR={result.fpr_at_threshold}, TPR={result.tpr_at_threshold})")
    cal = confidence_calibration()
    print(f"confidence ECE = {cal['ece']}  (n={cal['n']})")
