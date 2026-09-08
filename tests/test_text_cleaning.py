"""Regression tests for deterministic cleanup of source/model extraction artifacts."""

from equity_research.text_cleaning import clean_public_text


def test_decodes_entities_and_repairs_joined_finance_text():
    dirty = "1*trilliontoward *&#x35;00*billionBlackwellandRubinopportunity"
    assert clean_public_text(dirty) == (
        "1 trillion toward 500 billion Blackwell and Rubin opportunity"
    )


def test_removes_inline_html_and_zero_width_spacing_artifacts():
    assert clean_public_text("Revenue&nbsp;<sub>QoQ</sub>\u200bgrew") == "Revenue QoQ grew"
