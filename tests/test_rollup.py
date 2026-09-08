"""F-04 / D-13 unit tests: the quarter roll-up (flow-vs-stock aware TTM).

The accounting subtlety these pin: combining four quarters is NOT one operation. FLOW items (revenue,
income, cash flow) accumulate -> a year is their SUM. STOCK items (debt, equity, current assets) are a
point-in-time balance -> a year is the LATEST quarter's balance, never a sum. Summing balance-sheet
snapshots would quadruple the company's debt — the exact error this module exists to prevent.
"""

import pytest

from equity_research.tools.rollup import (
    classify_item,
    combine_quarters,
    ttm_label,
)


def _quarters():
    """Four quarters, latest-first, with both flow and stock items."""
    return [
        {"period": "2026-06-30", "line_items": {"revenue": 40.0, "cogs": 16.0, "total_debt": 500.0, "total_equity": 900.0}},
        {"period": "2026-03-31", "line_items": {"revenue": 35.0, "cogs": 15.0, "total_debt": 520.0, "total_equity": 850.0}},
        {"period": "2025-12-31", "line_items": {"revenue": 30.0, "cogs": 14.0, "total_debt": 540.0, "total_equity": 800.0}},
        {"period": "2025-09-30", "line_items": {"revenue": 25.0, "cogs": 13.0, "total_debt": 560.0, "total_equity": 750.0}},
    ]


# --- classification --------------------------------------------------------------------------------
@pytest.mark.parametrize("key, kind", [
    ("revenue", "flow"), ("cogs", "flow"), ("net_income", "flow"), ("free_cash_flow", "flow"),
    ("total_debt", "stock"), ("total_equity", "stock"), ("current_assets", "stock"),
    ("mystery_item", "unknown"),
])
def test_classify_item(key, kind):
    assert classify_item(key) == kind


# --- the core roll-up rule -------------------------------------------------------------------------
def test_flows_are_summed_across_quarters():
    rolled = combine_quarters(_quarters(), n=4)
    assert rolled["line_items"]["revenue"] == 40.0 + 35.0 + 30.0 + 25.0   # 130.0 TTM revenue
    assert rolled["line_items"]["cogs"] == 16.0 + 15.0 + 14.0 + 13.0      # 58.0 TTM cogs


def test_stocks_take_the_latest_balance_not_the_sum():
    rolled = combine_quarters(_quarters(), n=4)
    # THE error this prevents: NOT 500+520+540+560. The year's balance is the most recent quarter's.
    assert rolled["line_items"]["total_debt"] == 500.0
    assert rolled["line_items"]["total_equity"] == 900.0


def test_gross_margin_from_ttm_uses_summed_flows():
    # A yearly gross margin must divide summed revenue by summed cogs, i.e. computed on the TTM flows.
    rolled = combine_quarters(_quarters(), n=4)
    li = rolled["line_items"]
    gm = (li["revenue"] - li["cogs"]) / li["revenue"]
    assert gm == pytest.approx((130.0 - 58.0) / 130.0)


def test_records_periods_combined_for_provenance():
    rolled = combine_quarters(_quarters(), n=4)
    assert rolled["periods"] == ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
    assert rolled["n_quarters"] == 4 and rolled["complete"] is True


def test_partial_year_is_flagged_not_hidden():
    # Only two quarters available -> the TTM is partial; `complete` says so, sums only what exists.
    rolled = combine_quarters(_quarters()[:2], n=4)
    assert rolled["complete"] is False and rolled["n_quarters"] == 2
    assert rolled["line_items"]["revenue"] == 40.0 + 35.0


def test_flow_missing_everywhere_is_omitted_not_zero():
    # net_income appears in no quarter -> omitted from the roll-up (never guessed as 0.0).
    rolled = combine_quarters(_quarters(), n=4)
    assert "net_income" not in rolled["line_items"]


def test_ttm_label_names_the_latest_quarter():
    assert ttm_label(_quarters()) == "TTM ending 2026-06-30"
    assert ttm_label([]) == "TTM"
