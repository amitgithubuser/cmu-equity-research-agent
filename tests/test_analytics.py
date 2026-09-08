"""Trend-analytics tests (charts) — `analytics.build_trends`.

These lock the two anti-fabrication rules for visuals: (1) ratios use the SAME `calc` definitions, so
a chart number equals what the brief would cite; (2) a period missing an input is SKIPPED for that
ratio, never guessed. Pure — no network, no LLM.
"""

from __future__ import annotations

from equity_research.analytics import LEVEL_KEYS, RATIO_KEYS, build_trends
from equity_research.tools.calc import gross_margin

# Three fiscal years, newest-first on input (as yfinance returns columns) to prove we re-sort.
_HISTORY = {
    "line_items": {
        "2025-01-31": {"revenue": 130000.0, "cogs": 40000.0, "operating_income": 80000.0,
                       "net_income": 72000.0, "free_cash_flow": 60000.0},
        "2023-01-31": {"revenue": 26974.0, "cogs": 11618.0, "operating_income": 5577.0,
                       "net_income": 4368.0, "free_cash_flow": 3808.0},
        "2024-01-31": {"revenue": 60922.0, "cogs": 16621.0, "operating_income": 32972.0,
                       "net_income": 29760.0, "free_cash_flow": 27021.0},
    }
}


def test_empty_or_missing_history_returns_empty():
    assert build_trends(None) == []
    assert build_trends({}) == []
    assert build_trends({"line_items": {}}) == []


def test_points_are_oldest_first():
    points = build_trends(_HISTORY)
    periods = [p["period"] for p in points]
    assert periods == ["2023-01-31", "2024-01-31", "2025-01-31"]   # re-sorted chronologically


def test_ratios_match_calc_definitions():
    points = build_trends(_HISTORY)
    p0 = points[0]  # 2023
    # gross_margin here must equal the canonical calc definition (one source of truth, D-3)
    assert p0["gross_margin"] == round(gross_margin(revenue=26974.0, cogs=11618.0), 4)
    assert p0["revenue"] == 26974.0
    # margins are 0-1 ratios
    assert 0.0 < p0["net_margin"] < 1.0


def test_growth_appears_from_second_period_only():
    points = build_trends(_HISTORY)
    assert "revenue_growth" not in points[0]              # no prior period
    assert "revenue_growth" in points[1]                  # 2024 vs 2023
    # 2024 revenue more than doubled 2023 -> growth > 1.0
    assert points[1]["revenue_growth"] > 1.0


def test_missing_input_skips_that_ratio_not_the_point():
    hist = {"line_items": {
        "2023-01-31": {"revenue": 100.0},                 # only revenue -> margins can't be computed
        "2024-01-31": {"revenue": 120.0, "cogs": 60.0},   # gross margin computable
    }}
    points = build_trends(hist)
    assert len(points) == 2
    assert "gross_margin" not in points[0]                # skipped, not fabricated
    assert points[0]["revenue"] == 100.0                  # the point still exists (level present)
    assert points[1]["gross_margin"] == 0.5               # computed where inputs exist
    assert points[1]["revenue_growth"] == 0.2             # 120 vs 100


def test_zero_revenue_period_omits_undefined_ratios():
    hist = {"line_items": {"2024-01-31": {"revenue": 0.0, "cogs": 10.0}}}
    points = build_trends(hist)
    assert points[0]["revenue"] == 0.0
    assert "gross_margin" not in points[0]                # divide-by-zero -> omitted, not crashed


def test_key_partition_is_coherent():
    # every ratio key is 0-1-ish; revenue is a level — the UI relies on this split for axes
    assert "gross_margin" in RATIO_KEYS and "revenue_growth" in RATIO_KEYS
    assert "revenue" in LEVEL_KEYS and "revenue" not in RATIO_KEYS


def test_as_of_cutoff_excludes_future_period_before_growth_is_computed():
    hist = {"line_items": {
        "2024-01-31": {"revenue": 100.0, "cogs": 50.0},
        "2025-01-31": {"revenue": 150.0, "cogs": 60.0},
        "2026-01-31": {"revenue": 900.0, "cogs": 90.0},
    }}
    points = build_trends(hist, as_of="FY2025")
    assert [p["period"] for p in points] == ["2024-01-31", "2025-01-31"]
    assert points[-1]["revenue_growth"] == 0.5


def test_yoy_comparison_is_calculated_before_display_window_is_sliced():
    hist = {"line_items": {
        "2025-03-31": {"revenue": 100.0, "cogs": 40.0},
        "2025-06-30": {"revenue": 110.0, "cogs": 44.0},
        "2025-09-30": {"revenue": 120.0, "cogs": 48.0},
        "2025-12-31": {"revenue": 130.0, "cogs": 52.0},
        "2026-03-31": {"revenue": 150.0, "cogs": 52.5},
    }}
    points = build_trends(hist, display_quarters=4)
    assert len(points) == 4 and points[-1]["period"] == "2026-03-31"
    assert points[-1]["revenue_growth_yoy"] == 0.5
    assert points[-1]["gross_margin_yoy_change"] == 0.05
