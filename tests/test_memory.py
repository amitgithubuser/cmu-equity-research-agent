"""Memory tier gate (architecture §6.1/§6.2, CP 2.1) — the two memory modules and their integration.

Working memory: context management (prune/compact) must compress the *narrative* while NEVER touching
findings (evidence/metrics/uncertainties) — the one rule from §6.2.

Long-term memory: a cross-run record of prior-quarter numbers (+ tone) that survives, with a
quarter-over-quarter comparison read. Tests cover both backends (in-memory + JSON) and the Researcher
wiring that writes findings as-found and computes QoQ.
"""

from equity_research.memory import (
    FINDINGS_KEYS,
    InMemoryLongTermStore,
    JSONLongTermStore,
    PeriodRecord,
    compact_log,
    compact_state,
    compare_to_prior,
    get_longterm_store,
    prune_observation,
    would_overflow,
)


# =========================================================================== working memory (§6.2)
def test_prune_observation_shrinks_bulky_text_but_keeps_head():
    raw = "IMPORTANT NUMBER 42. " + ("filler " * 500)
    pruned = prune_observation(raw, keep_chars=40)
    assert pruned.startswith("IMPORTANT NUMBER 42.")   # the recognizable head is kept
    assert "elided" in pruned                          # the bulk is marked elided, not silently dropped
    assert len(pruned) < len(raw)


def test_prune_observation_leaves_short_text_untouched():
    assert prune_observation("short", keep_chars=300) == "short"


def test_compact_log_below_budget_is_identity():
    log = ["a", "b", "c"]
    assert compact_log(log, char_budget=10_000, keep_recent=3) == log


def test_compact_log_summarizes_older_keeps_recent_verbatim():
    log = [f"step {i}: " + ("x" * 50) for i in range(20)]
    out = compact_log(log, char_budget=100, keep_recent=3)
    assert len(out) == 4                     # one summary line + the 3 most recent verbatim
    assert out[1:] == log[-3:]               # recent tail preserved exactly
    assert "compacted" in out[0]             # older turns rolled into a summary


def test_compact_log_accepts_custom_summarizer():
    log = [f"line {i}" * 20 for i in range(10)]
    out = compact_log(log, char_budget=1, keep_recent=2, summarizer=lambda older: f"SUMMARY({len(older)})")
    assert out[0] == "SUMMARY(8)"


def test_compact_state_never_touches_findings():
    # THE RULE (§6.2): compressing the narrative is fine; findings are never discarded.
    state = {
        "log": [f"turn {i}: " + ("y" * 80) for i in range(30)],
        "evidence": [{"claim": "gross margin 40%", "citation": {"snippet": "..."}}],
        "metrics": {"gross_margin": {"value": 0.4}},
        "open_uncertainties": ["some gap"],
        "history": {"line_items": {"2024": {"revenue": 1.0}}},
        "qoq": {"has_prior": True},
    }
    update = compact_state(state, char_budget=100, keep_recent=3)
    # it only rewrites the narrative log ...
    assert "log" in update and len(update["log"]) < len(state["log"])
    # ... and returns NOTHING that would overwrite a findings key
    assert not (set(update) & set(FINDINGS_KEYS)), "compaction must not emit findings keys"


def test_would_overflow_flags_over_budget():
    assert would_overflow({"log": ["x" * 100]}, char_budget=10) is True
    assert would_overflow({"log": ["x"]}, char_budget=10) is False


# =========================================================================== long-term memory (§6.1)
def test_inmemory_put_get_roundtrip():
    store = InMemoryLongTermStore()
    store.put(PeriodRecord(company="NVDA", period="FY2024", metrics={"gross_margin": 0.56}))
    rec = store.get("nvda", "FY2024")            # case-insensitive company key
    assert rec is not None and rec.metrics["gross_margin"] == 0.56


def test_history_is_oldest_first_across_period_formats():
    store = InMemoryLongTermStore()
    for p in ["FY2025", "FY2023", "FY2024"]:
        store.put(PeriodRecord(company="NVDA", period=p, metrics={}))
    assert [r.period for r in store.history("NVDA")] == ["FY2023", "FY2024", "FY2025"]


def test_prior_returns_most_recent_strictly_older():
    store = InMemoryLongTermStore()
    for p in ["FY2023", "FY2024", "FY2025"]:
        store.put(PeriodRecord(company="NVDA", period=p, metrics={}))
    assert store.prior("NVDA", "FY2025").period == "FY2024"
    assert store.prior("NVDA", "FY2023") is None      # nothing older than the earliest


def test_quarter_period_ordering():
    store = InMemoryLongTermStore()
    for p in ["Q1-2025", "Q4-2024", "Q2-2025"]:
        store.put(PeriodRecord(company="NVDA", period=p, metrics={}))
    assert [r.period for r in store.history("NVDA")] == ["Q4-2024", "Q1-2025", "Q2-2025"]


def test_compare_to_prior_computes_metric_deltas_and_direction():
    store = InMemoryLongTermStore()
    store.put(PeriodRecord(company="NVDA", period="FY2024",
                           metrics={"gross_margin": 0.60}, tone=0.2))
    qoq = compare_to_prior(store, "NVDA", "FY2025",
                           {"gross_margin": 0.56}, current_tone=-0.1)
    assert qoq["has_prior"] and qoq["prior_period"] == "FY2024"
    d = qoq["metric_deltas"]["gross_margin"]
    assert d["direction"] == "down" and round(d["delta"], 4) == -0.04
    assert qoq["tone_direction"] == "more cautious"
    assert "gross_margin down" in qoq["summary"] and "cautious" in qoq["summary"]


def test_compare_to_prior_without_prior_is_graceful():
    store = InMemoryLongTermStore()
    qoq = compare_to_prior(store, "NVDA", "FY2025", {"gross_margin": 0.56})
    assert qoq["has_prior"] is False and qoq["metric_deltas"] == {}


def test_compare_skips_metrics_absent_in_prior_no_fabrication():
    store = InMemoryLongTermStore()
    store.put(PeriodRecord(company="NVDA", period="FY2024", metrics={"gross_margin": 0.60}))
    qoq = compare_to_prior(store, "NVDA", "FY2025",
                           {"gross_margin": 0.56, "net_margin": 0.3})   # net_margin has no baseline
    assert set(qoq["metric_deltas"]) == {"gross_margin"}                # net_margin skipped, not guessed


def test_json_store_persists_across_instances(tmp_path):
    store = JSONLongTermStore(str(tmp_path))
    store.put(PeriodRecord(company="NVDA", period="FY2024", metrics={"gross_margin": 0.56}))
    assert (tmp_path / "longterm.json").exists()
    # a fresh instance pointed at the same dir sees the record (the cross-run promise)
    reopened = JSONLongTermStore(str(tmp_path))
    assert reopened.get("NVDA", "FY2024").metrics["gross_margin"] == 0.56


def test_json_store_survives_corrupt_file(tmp_path):
    (tmp_path / "longterm.json").write_text("{ not json")
    store = JSONLongTermStore(str(tmp_path))          # must not raise
    assert store.history("NVDA") == []


def test_get_longterm_store_selects_backend(tmp_path):
    class _S:
        memory_dir = str(tmp_path)
    assert isinstance(get_longterm_store(_S()), JSONLongTermStore)

    class _Empty:
        memory_dir = ""
    assert isinstance(get_longterm_store(_Empty()), InMemoryLongTermStore)


# =========================================================================== researcher integration
def _market_lookup(ticker, period="annual"):
    return {"ticker": ticker, "period": period, "price": 100.0, "as_of": "FY2025",
            "line_items": {"revenue": 100.0, "cogs": 44.0}}


def _market_quarters(ticker, n=4):
    rows = [
        {"period": "2025-01-31", "line_items": {"revenue": 25.0, "cogs": 11.0}},
        {"period": "2024-10-31", "line_items": {"revenue": 25.0, "cogs": 11.0}},
        {"period": "2024-07-31", "line_items": {"revenue": 25.0, "cogs": 11.0}},
        {"period": "2024-04-30", "line_items": {"revenue": 25.0, "cogs": 11.0}},
    ]
    return {"ticker": ticker, "quarters": rows[:n]}


class _FakeTool:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, args):
        return self._fn(**args)


def test_researcher_writes_finding_to_longterm_and_reads_qoq():
    from equity_research.agents import researcher_node
    from equity_research.llm import FakeLLM

    store = InMemoryLongTermStore()
    # seed a prior year so the run has a QoQ baseline
    store.put(PeriodRecord(company="NVDA", period="FY2024", metrics={"gross_margin": 0.50}))

    tools = {"market_lookup": _FakeTool(_market_lookup),
             "market_quarters": _FakeTool(_market_quarters), "calc": __import__(
        "equity_research.tools", fromlist=["calc"]).calc}
    state = {"ticker": "NVDA", "as_of": "FY2025",
             "plan": [{"claim": "gross margin", "kind": "number", "metric": "gross_margin"}]}
    out = researcher_node(state, FakeLLM(), tools, retrieve_fn=lambda *a, **k: [], memory=store)

    # Four-quarter gross margin = (100-44)/100 = 0.56 → up 6pp vs FY2024's 0.50.
    assert out["qoq"]["has_prior"] and out["qoq"]["prior_period"] == "FY2024"
    assert out["qoq"]["metric_deltas"]["gross_margin"]["direction"] == "up"
    # The finding is stored under the exact TTM window that produced it, not mislabeled as a filing.
    assert store.get("NVDA", "TTM ending 2025-01-31").metrics["gross_margin"] == 0.56


def test_researcher_without_memory_still_works():
    from equity_research.agents import researcher_node
    from equity_research.llm import FakeLLM

    tools = {"market_lookup": _FakeTool(_market_lookup),
             "market_quarters": _FakeTool(_market_quarters), "calc": __import__(
        "equity_research.tools", fromlist=["calc"]).calc}
    state = {"ticker": "NVDA", "as_of": "FY2025",
             "plan": [{"claim": "gross margin", "kind": "number", "metric": "gross_margin"}]}
    out = researcher_node(state, FakeLLM(), tools, retrieve_fn=lambda *a, **k: [])   # no memory
    assert "qoq" not in out                     # gracefully absent
    assert out["evidence"][0]["value"] == 0.56  # evidence unaffected


def test_researcher_qoq_read_precedes_write_no_self_compare():
    # First-ever run for a company: reading QoQ must NOT see the record we're about to write this run.
    from equity_research.agents import researcher_node
    from equity_research.llm import FakeLLM

    store = InMemoryLongTermStore()
    tools = {"market_lookup": _FakeTool(_market_lookup),
             "market_quarters": _FakeTool(_market_quarters), "calc": __import__(
        "equity_research.tools", fromlist=["calc"]).calc}
    state = {"ticker": "NVDA", "as_of": "FY2025",
             "plan": [{"claim": "gross margin", "kind": "number", "metric": "gross_margin"}]}
    out = researcher_node(state, FakeLLM(), tools, retrieve_fn=lambda *a, **k: [], memory=store)
    assert out.get("qoq", {"has_prior": False}).get("has_prior") is False  # no prior on first run
    assert store.get("NVDA", "TTM ending 2025-01-31") is not None            # but it WAS recorded
