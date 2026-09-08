"""Phase 12 gate: the UI presentation logic is framework-free and correct (the Streamlit shell itself
is exercised only under `streamlit run`). Renders a brief to Markdown, summarizes the monitor, and
handles the withheld/insufficient cases.
"""

from equity_research.ui import (
    brief_markdown,
    case_evaluation_sections,
    grouped_case_content,
    monitor_summary,
    progress_label,
    runtime_profile_rows,
    trajectory_lines,
    trend_change_rows,
    trends_chart_data,
    trends_table_markdown,
)


def _brief(trends=None):
    bull = {"claim": "Data center revenue grew", "kind": "text",
            "citation": {"source": "Earnings call transcript", "company": "NVDA", "period": "FY2025",
                         "snippet": "Demand for accelerated computing continued to grow.",
                         "url": "http://x"}}
    risk = {"claim": "Customer concentration", "kind": "text",
            "citation": {"source": "10-K Item 1A", "company": "NVDA", "period": "FY2025",
                         "snippet": "A limited number of customers represent a material share."}}
    return {
        "ticker": "NVDA", "as_of": "FY2025",
        "bull_case": [bull],
        "bear_case": [],
        "key_risks": [risk],
        "theses": [{"side": "bull", "thesis": "AI demand supports growth", "claims": [bull],
                    "support_score": 0.82, "survived": True}],
        "metrics": {"gross_margin": {"name": "gross_margin", "value": 0.73, "inputs": {}}},
        "evidence_appendix": [bull, risk],
        "open_questions": ["How durable is customer demand?"],
        "confidence": 0.62, "confidence_rationale": "bull support exceeds bear",
        "trends": trends or [],
        "disclaimer": "This is research, not financial advice.",
    }


_TRENDS = [
    {"period": "2023-01-31", "revenue": 26974.0, "gross_margin": 0.569, "net_margin": 0.162},
    {"period": "2024-01-31", "revenue": 60922.0, "gross_margin": 0.727, "net_margin": 0.489,
     "revenue_growth": 1.259},
]


def test_brief_markdown_has_sections_and_citations():
    md = brief_markdown(_brief())
    assert "NVDA — Equity Research Report" in md
    assert "Bull case" in md and "Bear case" in md and "Key risks" in md
    assert "[S1]" in md and "[Earnings call transcript · FY2025](http://x)" in md
    assert "[S1](http://x)" in md
    assert "How to read the evidence score" in md
    assert "Insufficient cited evidence" in md                     # empty bear side, not invented
    assert "AI demand supports growth" in md and "Management commentary" in md
    assert "Open questions and evidence gaps" in md and "Sources and evidence" in md
    assert "<sub>" not in md
    assert "not financial advice" in md


def test_brief_markdown_withheld():
    md = brief_markdown({"ticker": "NVDA", "withheld": True, "reason": "blocked",
                         "disclaimer": "This is research, not financial advice."})
    assert "withheld" in md.lower() and "blocked" in md


def test_brief_markdown_empty():
    assert "No brief" in brief_markdown({})


def test_monitor_summary_fields():
    state = {"suspicion": 50.0, "needs_review": True,
             "suspicion_report": {"fired": ["uncited_claim"]},
             "review_triggers": ["source_conflict"],
             "escalation_reason": "uncited_claim", "blocked": True, "human_decision": "block"}
    summary = monitor_summary(state)
    assert summary["suspicion"] == 50.0 and summary["needs_review"] is True
    assert summary["fired"] == ["uncited_claim"] and summary["blocked"] is True
    assert summary["review_triggers"] == ["source_conflict"]


def test_trajectory_lines_passthrough():
    assert trajectory_lines({"log": ["a", "b", "c"]}) == ["a", "b", "c"]
    assert trajectory_lines({}) == []


def test_progress_label_reports_safe_counts_not_source_content():
    update = {
        "transcript_periods": ["Q4-2025"],
        "filing_periods": ["FY2025", "Q3-2025"],
        "messages": ["secret prompt body"],
    }
    label = progress_label("source_setup", update)
    assert "1 transcript period" in label and "2 filing period" in label
    assert "secret" not in label
    assert progress_label("planner", {"plan": [{}, {}, {}]}).endswith("3 evidence task(s)")


def test_runtime_profile_rows_are_reader_friendly_and_include_setup():
    rows = runtime_profile_rows({
        "total_seconds": 10.0,
        "stages": [{"stage": "planner", "seconds": 6.0}, {"stage": "critic", "seconds": 4.0}],
    }, service_setup_seconds=2.0)
    assert rows[0] == {"Stage": "Building the research plan", "Seconds": 6.0, "Share": "50%"}
    assert any(row["Stage"] == "Starting models and local services" for row in rows)


# --- trend charts ----------------------------------------------------------------------------------
def test_trends_chart_data_splits_margins_from_levels():
    data = trends_chart_data(_brief(trends=_TRENDS))
    assert data["periods"] == ["2023-01-31", "2024-01-31"]
    assert "Revenue" in data["levels"] and "Revenue" not in data["margins"]   # level on its own axis
    assert "Gross margin" in data["margins"] and "Net margin" in data["margins"]
    # a metric missing in a period contributes None (a line gap), not a zero
    assert data["growth"]["Revenue growth (QoQ)"] == [None, 1.259]


def test_trends_chart_data_empty_when_no_trends():
    data = trends_chart_data(_brief())
    assert data["periods"] == [] and data["margins"] == {} and data["growth"] == {}
    assert data["levels"] == {}


def test_trends_table_markdown_renders_percentages():
    table = trends_table_markdown(_brief(trends=_TRENDS))
    assert "| Period |" in table and "Gross margin" in table
    assert "56.9%" in table                      # margin formatted as a percentage
    assert "26,974" in table                     # revenue formatted with thousands separators
    assert "—" in table                          # missing revenue_growth in period 1 shows a dash


def test_brief_markdown_includes_trend_table_when_present():
    md = brief_markdown(_brief(trends=_TRENDS))
    assert "Financial trend detail" in md and "Gross margin" in md
    assert md.index("Financial trend detail") < md.index("Investment cases")


def test_brief_markdown_omits_trends_when_absent():
    assert "Financial trend detail" not in brief_markdown(_brief())


# --- F-10: the Streamlit shell itself, via streamlit.testing.v1.AppTest ------------------------------
def _fake_app_result():
    """A finished-state dict shaped like `run.py` returns, so `main()` renders the final-brief path."""
    return {
        "brief": _brief(),
        "suspicion": 12.0, "needs_review": False,
        "suspicion_report": {"fired": []}, "log": ["planner: ok", "finalize: brief published"],
    }


class _FakeApp:
    """Stands in for the compiled graph: `.invoke(...)` returns a finished state (no interrupt)."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append(state)
        return self._result


def test_app_runs_and_renders_final_brief():
    # F-10 regression: the app must be importable AND runnable by Streamlit's own harness. We seed a
    # fake compiled app into session_state so no real LLM/tools are built, click Run, and assert the
    # brief renders — proving main() is wired correctly (it used to crash on a relative-import error).
    from streamlit.testing.v1 import AppTest

    def _script():
        import streamlit as st

        from equity_research.ui.app import main
        from tests.test_ui import _fake_app_result, _FakeApp

        if "app" not in st.session_state:
            st.session_state.app = _FakeApp(_fake_app_result())
        main()

    at = AppTest.from_function(_script)
    at.run()
    assert not at.exception                                   # main() executed without error
    # before clicking Run, the app shows the prompt-to-run info banner
    assert any("Run analysis" in i.value for i in at.info)
    # click Run -> the fake app is invoked and the final brief renders in the main column
    at.button[0].click().run()
    assert not at.exception
    assert st_markdown_contains(at, "Equity Research Report")
    assert any("Bull case" in tab.label for tab in at.tabs)


def test_financial_snapshot_and_evidence_appendix_helpers():
    from equity_research.ui import (
        evidence_source_rows,
        financial_snapshot_rows,
        management_commentary,
    )

    brief = _brief(trends=_TRENDS)
    snapshot = financial_snapshot_rows(brief)
    assert snapshot[0]["Metric"] == "Revenue" and snapshot[0]["Value"] == "$60,922M"
    assert snapshot[0]["Citation"] == ""  # no numeric evidence was supplied by this fixture
    assert any(row["Metric"] == "Gross margin" and row["Value"] == "73.0%" for row in snapshot)

    raw_dollar_brief = _brief(trends=[{"period": "2026-04-30", "revenue": 81_615_000_000}])
    assert financial_snapshot_rows(raw_dollar_brief)[0]["Value"] == "$81.6B"

    sources = evidence_source_rows(brief)
    assert len(sources) == 2 and sources[0]["Evidence"]
    assert len(management_commentary(brief)) == 1

    numeric = {"claim": "Gross margin was 73%", "kind": "number", "citation": {
        "source": "financial statements", "period": "FY2025", "snippet": "gross margin = 73%",
        "url": "http://financials",
    }}
    brief["evidence_appendix"].insert(0, numeric)
    cited_snapshot = financial_snapshot_rows(brief)
    assert cited_snapshot[0]["Citation"] == "[S1](http://financials)"
    assert next(row for row in cited_snapshot if row["Metric"] == "Gross margin")["Citation"] == (
        "[S1](http://financials)"
    )


def test_case_content_groups_evidence_and_places_signals_at_branch_level():
    margin = {"claim": "Gross margin expanded", "citation": {"snippet": "profitability rose"}}
    cash = {"claim": "Free cash flow increased", "citation": {"snippet": "cash generation"}}
    groups = grouped_case_content({
        "claims": [margin, cash],
        "catalysts": ["Further gross margin expansion"],
        "watch_items": ["Free cash flow conversion"],
    })
    by_theme = {group["theme"]: group for group in groups}
    assert len(by_theme["Margins & profitability"]["evidence"]) == 1
    assert len(by_theme["Cash generation & balance sheet"]["evidence"]) == 1
    sections = case_evaluation_sections({
        "catalysts": ["Further gross margin expansion"],
        "watch_items": ["Free cash flow conversion"],
        "invalidation_conditions": ["Gross margin contracts materially"],
    })
    assert [section["heading"] for section in sections] == [
        "What could strengthen this thesis",
        "What investors should watch",
        "What would challenge this thesis",
    ]
    assert all(sentence.endswith(".") for section in sections for sentence in section["sentences"])
    assert "because a material change could alter the thesis" in sections[1]["sentences"][0]


def test_trend_change_rows_separates_qoq_and_yoy():
    trends = [
        {"period": "Q1", "revenue": 100.0, "gross_margin": 0.70},
        {"period": "Q2", "revenue": 120.0, "revenue_growth": 0.20,
         "revenue_growth_yoy": 0.50, "gross_margin": 0.72,
         "gross_margin_yoy_change": 0.03},
    ]
    rows = trend_change_rows(_brief(trends=trends))
    revenue = next(row for row in rows if row["Metric"] == "Revenue")
    margin = next(row for row in rows if row["Metric"] == "Gross margin")
    assert revenue["QoQ"] == "▲ 20.0%" and revenue["YoY"] == "▲ 50.0%"
    assert margin["QoQ"] == "▲ 2.0 pp" and margin["YoY"] == "▲ 3.0 pp"


def st_markdown_contains(at, needle: str) -> bool:
    """True if any rendered markdown/caption/title block contains `needle`."""
    blocks = [*at.markdown, *at.title, *at.caption]
    return any(needle in (b.value or "") for b in blocks)
