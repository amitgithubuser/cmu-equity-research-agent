"""F-05 unit tests: the canonical period model.

The bug F-05 caught: EDGAR ingestion stored the SEC *report date* ("2025-01-26") in `period`, while
retrieval filtered on the user's `--as-of` string ("FY2025") with exact equality — so the documented
command retrieved ZERO passages. The market path mirrored it: hard-coded "annual", ignoring --as-of.

These tests pin the ONE representation that fixes both: every spelling normalizes to a canonical label
("FY2025" / "Q1-2025"), and two spellings of the same period compare equal after normalization.
"""

import pytest

from equity_research.periods import (
    canonical_label,
    market_period_kind,
    normalize_period,
    periods_match,
)


# --- normalize_period: every accepted spelling -> one canonical label ------------------------------
@pytest.mark.parametrize(
    "raw, label, kind, year, quarter",
    [
        ("2025-01-26", "FY2025", "annual", 2025, None),   # SEC report date -> fiscal year
        ("FY2025", "FY2025", "annual", 2025, None),
        ("FY-2025", "FY2025", "annual", 2025, None),
        ("2025", "FY2025", "annual", 2025, None),          # bare year
        ("Q1-2025", "Q1-2025", "quarterly", 2025, 1),
        ("Q1 2025", "Q1-2025", "quarterly", 2025, 1),      # space spelling
        ("2025-Q1", "Q1-2025", "quarterly", 2025, 1),      # suffix spelling
        ("Q4-2024", "Q4-2024", "quarterly", 2024, 4),
    ],
)
def test_normalize_period_canonicalizes_every_spelling(raw, label, kind, year, quarter):
    ref = normalize_period(raw)
    assert ref is not None
    assert ref.label == label
    assert ref.kind == kind
    assert ref.fiscal_year == year
    assert ref.quarter == quarter
    assert ref.raw == raw                       # original spelling is retained for provenance
    assert ref.is_quarterly == (kind == "quarterly")


@pytest.mark.parametrize("bad", [None, "", "   ", "not-a-period", "Q5-2025", "2025-13-01"])
def test_normalize_period_returns_none_for_unparseable(bad):
    assert normalize_period(bad) is None
    assert canonical_label(bad) is None


# --- periods_match: THE F-05 equivalence -----------------------------------------------------------
def test_report_date_matches_fy_label():
    # The exact miss F-05 documented: a chunk indexed from report date "2025-01-26" must be found
    # when the user asks for "FY2025". Both normalize to "FY2025".
    assert periods_match("2025-01-26", "FY2025")
    assert periods_match("FY2025", "2025-01-26")


def test_periods_match_across_spellings():
    assert periods_match("Q1 2025", "2025-Q1")   # different quarter spellings, same period
    assert periods_match("2025", "FY-2025")


def test_periods_do_not_match_across_different_periods():
    assert not periods_match("FY2025", "FY2024")
    assert not periods_match("Q1-2025", "Q2-2025")
    assert not periods_match("2024-01-26", "FY2025")   # a competitor-year report must NOT match


def test_periods_match_falls_back_to_exact_when_unparseable():
    # If EITHER side can't be normalized, behave like the old exact filter — never silently over-match.
    assert periods_match("custom-fy-label", "custom-fy-label")
    assert not periods_match("custom-fy-label", "FY2025")


# --- market_period_kind: quarterly question -> quarterly fetch (F-05 market half) -------------------
@pytest.mark.parametrize("raw, kind", [
    ("Q1-2025", "quarterly"),
    ("2025-Q3", "quarterly"),
    ("FY2025", "annual"),
    ("2025", "annual"),
    ("2025-01-26", "annual"),
    (None, "annual"),          # no period -> annual default
    ("garbage", "annual"),     # unparseable -> annual default
])
def test_market_period_kind(raw, kind):
    assert market_period_kind(raw) == kind
