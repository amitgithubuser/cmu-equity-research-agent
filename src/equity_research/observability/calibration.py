"""Threshold calibration (architecture §11.3, implementation-plan §11.4, Lab 6.2 `roc_auc`/
`calibrate_threshold`).

The monitor's `escalation_threshold` (§10.2) is a placeholder (50). It must be **set by data, not
guessed** (OQ-7). Given a labeled set of runs — each a `(suspicion_score, is_risky)` pair — we:

  1. sweep every candidate threshold,
  2. compute the ROC curve (TPR vs. FPR) and its AUC,
  3. pick the **lowest threshold whose false-positive rate on benign runs ≤ `target_fpr`** (the CP 6.1
     decision rule: "at most 10% false alarms on safe runs").

`sklearn` is used on a real install (`roc_auc_score`); a pure-Python fallback keeps this offline-
testable and dependency-light. The chosen value is written back into `config.escalation_threshold` —
**evaluation configures the gate**, closing OQ-7.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


@dataclass
class CalibrationResult:
    threshold: float          # chosen escalation cutoff
    auc: float                # ROC AUC of the suspicion score as a risk classifier
    fpr_at_threshold: float   # false-positive rate on benign runs at the chosen threshold
    tpr_at_threshold: float   # true-positive rate (recall of risky runs) at the chosen threshold


def _confusion(scores: list[float], labels: list[int], threshold: float) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) for `flag if score >= threshold`; label 1 = risky (should be flagged)."""
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        flagged = score >= threshold
        if label == 1 and flagged:
            tp += 1
        elif label == 1 and not flagged:
            fn += 1
        elif label == 0 and flagged:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def roc_points(scores: list[float], labels: list[int]) -> list[tuple[float, float, float]]:
    """Return sorted [(threshold, fpr, tpr), ...] over all candidate thresholds."""
    # candidate thresholds: each distinct score, plus one above the max so nothing is flagged
    candidates = sorted(set(scores) | {max(scores) + 1.0}) if scores else [0.0]
    pts = []
    for t in candidates:
        tp, fp, tn, fn = _confusion(scores, labels, t)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        pts.append((t, fpr, tpr))
    return pts


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve. Uses sklearn if present; else the trapezoidal rule on `roc_points`."""
    try:  # pragma: no cover - real path when scikit-learn is installed
        from sklearn.metrics import roc_auc_score

        if len(set(labels)) < 2:
            return 0.5
        return float(roc_auc_score(labels, scores))
    except Exception:  # noqa: BLE001
        pts = sorted(((fpr, tpr) for _, fpr, tpr in roc_points(scores, labels)))
        area = 0.0
        for (x0, y0), (x1, y1) in pairwise(pts):
            area += (x1 - x0) * (y0 + y1) / 2.0
        return round(area, 4)


def calibrate_threshold(scores: list[float], labels: list[int], target_fpr: float = 0.10) -> CalibrationResult:
    """Pick the LOWEST threshold with FPR ≤ `target_fpr` (most sensitive gate within the alarm budget).

    Lower threshold → catches more risky runs (higher TPR) but more false alarms. We take the lowest
    threshold that still respects the false-alarm budget, maximizing recall of risky runs.
    """
    if not scores:
        return CalibrationResult(threshold=50.0, auc=0.5, fpr_at_threshold=0.0, tpr_at_threshold=0.0)

    pts = roc_points(scores, labels)
    feasible = [(t, fpr, tpr) for (t, fpr, tpr) in pts if fpr <= target_fpr]
    # among thresholds within the FPR budget, choose the smallest (highest sensitivity)
    chosen = min(feasible, key=lambda p: p[0]) if feasible else max(pts, key=lambda p: p[0])
    t, fpr, tpr = chosen
    return CalibrationResult(threshold=round(float(t), 2), auc=roc_auc(scores, labels),
                             fpr_at_threshold=round(fpr, 4), tpr_at_threshold=round(tpr, 4))
