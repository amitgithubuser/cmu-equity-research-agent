"""Evaluation layer (architecture §11, implementation-plan Phase 10).

Grades the agent *on average* (evaluation), complementing `tests/` which checks specific parts *work*.
Three pieces, all deterministic + offline-testable, promoted to LangSmith Datasets/Evaluators/
Experiments on a live run:
  * `datasets/`    — fixed gold briefs + labeled runs (incl. an adversarial subset).
  * `evaluators`   — Lab 6.1 checks as metrics (groundedness, numeric correctness, hallucination, ECE).
  * `experiments/` — the guardrail ablation matrix + the ROC/AUC threshold calibration (closes OQ-7).
"""
