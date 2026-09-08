"""Phase 10 gate (implementation-plan §11): the ablation matrix runs and each guardrail shows a
measurable benefit; the escalation threshold is picked from ROC/AUC (data, not a guess), closing OQ-7.
"""

from equity_research.config import get_settings
from equity_research.observability.calibration import calibrate_threshold, roc_auc
from evals.datasets import GOLD_BRIEFS
from evals.evaluators import (
    answer_relevance,
    evaluate_brief,
    expected_calibration_error,
    faithfulness,
    groundedness,
    hallucination_rate,
    numeric_correctness,
    overconfidence_verdict,
    rag_triad,
    retrieval_relevance,
)
from evals.experiments.ablation import run_ablation
from evals.experiments.calibrate import confidence_calibration, run_calibration, score_runs


# --- evaluators ------------------------------------------------------------------------------------
def _by_id(bid):
    return next(e for e in GOLD_BRIEFS if e["id"] == bid)["brief"]


def test_groundedness_high_on_clean_brief():
    assert groundedness(_by_id("grounded-clean")) >= 0.95


def test_hallucination_detected_on_bad_brief():
    brief = _by_id("hallucinated-claim")
    assert groundedness(brief) < 0.95 and hallucination_rate(brief) > 0.0


def test_numeric_correctness_all_numbers_have_values():
    # every number claim in the clean brief carries a computed value
    assert numeric_correctness(_by_id("grounded-clean")) == 1.0


def test_overconfidence_flagged_on_prose():
    assert overconfidence_verdict(_by_id("overconfident-prose")) in ("WARN", "FAIL")


def test_evaluate_brief_bundle():
    m = evaluate_brief(_by_id("grounded-clean"))
    assert set(m) == {"groundedness", "hallucination_rate", "numeric_correctness", "overconfidence"}


def test_ece_perfect_calibration_is_low():
    # confidences equal to accuracy per bucket -> ECE ~ 0
    conf = [0.9, 0.9, 0.1, 0.1]
    correct = [1, 1, 0, 0]
    assert expected_calibration_error(conf, correct) < 0.15


# --- ablation: each guard earns its cost -----------------------------------------------------------
def test_ablation_matrix_runs_all_configs():
    r = run_ablation()
    assert set(r) == {"no_guardrails", "pre_only", "during_only", "post_only", "all"}


def test_post_guard_reduces_hallucination():
    r = run_ablation()
    # turning the post guard ON (post_only / all) eliminates missed hallucinations vs. no_guardrails
    assert r["no_guardrails"]["missed_hallucination_rate"] > r["all"]["missed_hallucination_rate"]
    assert r["all"]["missed_hallucination_rate"] == 0.0
    assert r["post_only"]["missed_hallucination_rate"] < r["no_guardrails"]["missed_hallucination_rate"]


def test_during_guard_reduces_overconfidence():
    r = run_ablation()
    assert r["no_guardrails"]["missed_overconfidence_rate"] > r["all"]["missed_overconfidence_rate"]
    assert r["all"]["missed_overconfidence_rate"] == 0.0


def test_all_guards_catch_the_most():
    r = run_ablation()
    assert r["all"]["caught"] >= max(r[c]["caught"] for c in r if c != "all")


# --- F-09 regressions: the specific bugs the review named --------------------------------------------
def test_pre_guard_earns_its_cost_not_a_clone_of_no_guardrails():
    # F-09 bug: run_ablation ignored cfg["pre"], so pre_only was identical to no_guardrails and the
    # pre-guard showed zero benefit. With an advice-origin brief in the gold set, turning `pre` ON must
    # now catch an advice query the no-guardrails column misses.
    r = run_ablation()
    assert r["no_guardrails"]["missed_advice_rate"] > 0.0        # advice slips with no pre-guard
    assert r["pre_only"]["missed_advice_rate"] < r["no_guardrails"]["missed_advice_rate"]
    assert r["all"]["missed_advice_rate"] == 0.0                 # all guards on -> nothing slips
    # the two columns are no longer identical across every dimension
    assert r["pre_only"] != r["no_guardrails"]


def test_numeric_correctness_fails_a_wrong_number():
    # F-09 bug: numeric_correctness only checked a number was non-null, so any WRONG figure passed.
    # The advice-origin brief states gross margin 0.42 but the reference says 0.60 -> must fail.
    entry = next(e for e in GOLD_BRIEFS if e["id"] == "advice-origin")
    score = numeric_correctness(entry["brief"], entry["numeric_reference"])
    assert score < 1.0                                           # the wrong number is caught


def test_numeric_correctness_passes_a_right_number_against_reference():
    entry = next(e for e in GOLD_BRIEFS if e["id"] == "grounded-clean")
    assert numeric_correctness(entry["brief"], entry["numeric_reference"]) == 1.0


# --- threshold calibration from ROC/AUC (OQ-7) -----------------------------------------------------
def test_roc_auc_separates_risky_from_benign():
    scores, labels = score_runs()
    # the monitor's suspicion score is a strong risk classifier on the labeled set
    assert roc_auc(scores, labels) >= 0.9


def test_calibration_picks_threshold_within_fpr_budget():
    result = run_calibration(target_fpr=0.10)
    assert result.fpr_at_threshold <= 0.10       # respects the false-alarm budget
    assert result.tpr_at_threshold >= 0.75        # still catches most risky runs
    assert 0.0 < result.threshold <= 100.0        # a real, data-derived cutoff


def test_calibrate_threshold_prefers_lower_within_budget():
    # two thresholds both within budget -> pick the lower (more sensitive)
    scores = [10.0, 20.0, 30.0, 40.0]
    labels = [0, 0, 1, 1]
    result = calibrate_threshold(scores, labels, target_fpr=0.0)
    assert result.threshold == 30.0  # lowest cutoff that flags both risky without any benign


# --- F-09: calibration is actually WRITTEN back, and ECE is a real experiment ------------------------
def test_run_calibration_persists_threshold_into_settings():
    # F-09 bug: run_calibration returned a value but never configured anything — the gate stayed at its
    # hand-set placeholder. With persist=True the data-derived cutoff must land in live settings.
    try:
        result = run_calibration(target_fpr=0.10, persist=True)
        assert get_settings().escalation_threshold == result.threshold
    finally:
        # don't leak the mutated env/singleton into other tests
        import os

        os.environ.pop("ESCALATION_THRESHOLD", None)
        get_settings.cache_clear()


def test_run_calibration_without_persist_leaves_settings_untouched():
    get_settings.cache_clear()
    before = get_settings().escalation_threshold
    run_calibration(target_fpr=0.10)          # persist defaults to False
    assert get_settings().escalation_threshold == before


def test_confidence_calibration_reports_ece_over_the_gold_set():
    # F-09: ECE is wired into a real experiment (not the single-brief bundle). It must consume the gold
    # set's confidences and return a bounded score over a non-empty sample.
    report = confidence_calibration()
    assert report["n"] >= 3
    assert 0.0 <= report["ece"] <= 1.0


# --- F-09: the RAG evaluation triad (faithfulness / answer- / retrieval-relevance) ------------------
def test_faithfulness_higher_for_grounded_than_hallucinated():
    # a clean brief's claims are entailed by their snippets; a hallucinated one's are not
    assert faithfulness(_by_id("grounded-clean")) > faithfulness(_by_id("hallucinated-claim"))


def test_answer_relevance_rewards_the_on_topic_question():
    brief = _by_id("grounded-clean")
    on_topic = answer_relevance(brief, "What drove TESTCO revenue and gross margin?")
    off_topic = answer_relevance(brief, "What is the boiling point of water?")
    assert on_topic > off_topic


def test_retrieval_relevance_rewards_on_topic_context():
    # offline the embedder is non-semantic, so the Jaccard token-overlap term carries the signal:
    # on-topic context shares the question's words; off-topic shares none.
    q = "What drove revenue growth this year?"
    good = retrieval_relevance(q, ["Revenue growth accelerated this year"])
    bad = retrieval_relevance(q, ["The cafeteria menu changed on Tuesday"])
    assert good > bad


def test_rag_triad_returns_three_named_scores():
    triad = rag_triad(_by_id("grounded-clean"), "What is the investment case for TESTCO?")
    assert set(triad) == {"faithfulness", "answer_relevance", "retrieval_relevance"}


def test_rag_triad_is_separate_from_the_single_brief_bundle():
    # evaluate_brief stays the four self-contained checks; the triad is NOT folded in
    assert set(evaluate_brief(_by_id("grounded-clean"))) == {
        "groundedness", "hallucination_rate", "numeric_correctness", "overconfidence"}


# --- F-09H: the guarded LangSmith adapter (offline path is the CI path) ------------------------------
def test_langsmith_adapter_is_offline_without_a_key(monkeypatch):
    # no key + (maybe) no langsmith lib -> is_live() is False and every entry point degrades gracefully
    # An explicit empty environment value overrides any developer-local `.env` file.
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()
    from evals.langsmith_adapter import is_live, push_dataset, run_experiment

    assert is_live() is False
    ds = push_dataset()
    assert ds["status"] == "offline" and ds["n"] >= 3          # reports the gold set size, no network
    exp = run_experiment()
    assert exp["status"] == "offline"
    assert exp["n"] >= 3 and 0.0 <= exp["mean_groundedness"] <= 1.0   # real local scores, CI-safe
    get_settings.cache_clear()


def test_langsmith_evaluators_reuse_the_pure_functions():
    # the wrapped evaluators score a brief with the SAME functions the offline tests use (no drift)
    from evals.langsmith_adapter import as_langsmith_evaluators

    evaluators = as_langsmith_evaluators()
    assert len(evaluators) >= 2
    # a LangSmith evaluator is called with a run-like object carrying `outputs`; feed it a clean brief
    class _Run:
        def __init__(self):
            self.outputs = {"brief": _by_id("grounded-clean")}
            self.inputs = {"question": "What is the investment case for TESTCO?"}
    scored = [ev(_Run()) for ev in evaluators]
    keys = {s["key"] for s in scored}
    assert "groundedness" in keys
    assert all(0.0 <= s["score"] <= 1.0 for s in scored)       # every wrapped score is a valid metric
