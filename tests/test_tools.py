"""Phase 2 gate (implementation-plan §3.5): numbers are trustworthy BEFORE any LLM touches them.

`calc` is tested hardest (exact values + division guards). The network tools' *pure* helpers
(divergence flag, sentiment) are tested offline; the HTTP paths themselves are exercised in the
`live` tier, not here (no live network in CI).
"""

import math

import pytest

from equity_research.tools import calc
from equity_research.tools.edgar import strip_html
from equity_research.tools.market import divergence
from equity_research.tools.news import is_company_relevant, is_material_headline, score_sentiment
from equity_research.tools.ratings import summarize_distribution


# --- calc: exact values ----------------------------------------------------------------------------
def test_gross_margin():
    assert calc.invoke({"metric": "gross_margin", "inputs": {"revenue": 100.0, "cogs": 60.0}})["value"] == 0.4


def test_debt_to_equity():
    assert calc.invoke({"metric": "debt_to_equity", "inputs": {"total_debt": 50.0, "total_equity": 200.0}})["value"] == 0.25


def test_revenue_growth():
    out = calc.invoke({"metric": "revenue_growth", "inputs": {"current": 120.0, "prior": 100.0}})
    assert out["value"] == 0.2


def test_calc_echoes_inputs_for_provenance():
    out = calc.invoke({"metric": "gross_margin", "inputs": {"revenue": 100.0, "cogs": 60.0}})
    assert out["name"] == "gross_margin" and out["inputs"] == {"revenue": 100.0, "cogs": 60.0}


# --- calc: division guardrails ---------------------------------------------------------------------
def test_calc_zero_revenue_raises():
    with pytest.raises(ValueError):
        calc.invoke({"metric": "gross_margin", "inputs": {"revenue": 0.0, "cogs": 60.0}})


def test_calc_zero_equity_raises():
    with pytest.raises(ValueError):
        calc.invoke({"metric": "debt_to_equity", "inputs": {"total_debt": 50.0, "total_equity": 0.0}})


def test_calc_unknown_metric_raises():
    with pytest.raises(ValueError):
        calc.invoke({"metric": "sharpe_ratio", "inputs": {}})


# --- vendor divergence (OQ-4) ----------------------------------------------------------------------
def test_vendor_divergence_within_threshold():
    rel, flag = divergence(0.400, 0.410, rel_threshold=0.05)  # ~2.4% < 5%
    assert flag is False and rel < 0.05


def test_vendor_divergence_flag_fires():
    _rel, flag = divergence(0.40, 0.30, rel_threshold=0.05)   # 33% > 5%
    assert flag is True


def test_vendor_divergence_zero_vendor_is_flagged():
    rel, flag = divergence(0.40, 0.0)
    assert flag is True and math.isinf(rel)


# --- news sentiment (pure) -------------------------------------------------------------------------
def test_sentiment_positive_and_negative():
    assert score_sentiment("Company beats earnings, shares surge to record") > 0
    assert score_sentiment("Company misses estimates amid lawsuit and probe") < 0
    assert score_sentiment("Company holds annual meeting on Tuesday") == 0.0


def test_news_relevance_requires_ticker_or_issuer_name():
    assert is_company_relevant("Nvidia announces a new platform", "NVDA", ("NVIDIA Corporation", "Nvidia"))
    assert is_company_relevant("NVDA shares rise", "NVDA", ("Nvidia",))
    assert not is_company_relevant("Bill Ackman discusses Pershing Square", "NVDA", ("Nvidia",))


def test_news_developments_exclude_stock_picking_listicles():
    assert is_material_headline("Nvidia launches a new AI infrastructure platform")
    assert not is_material_headline("NVIDIA & 2 Profitable Stocks to Buy for Big Upside")
    assert not is_material_headline("Prediction: What a $1,000 Investment in Nvidia Will Be Worth")


# --- analyst ratings distribution (pure, F-04 / D-13) ----------------------------------------------
def test_ratings_distribution_counts_and_modal():
    d = summarize_distribution({"strongBuy": 10, "buy": 18, "hold": 8, "sell": 3, "strongSell": 1},
                               as_of="2026-08")
    assert d["total"] == 40
    assert d["modal"] == "buy"                 # 18 is the largest bucket
    assert d["distribution"]["buy"] == pytest.approx(18 / 40)


def test_ratings_summary_is_attributed_not_advice():
    # The design rule (D-13): the summary REPORTS the consensus as an attributed observation; it must
    # not read as the agent's own recommendation. It states counts, not "we recommend / you should".
    d = summarize_distribution({"strongBuy": 2, "buy": 3, "hold": 20, "sell": 1, "strongSell": 0})
    s = d["summary"].lower()
    assert "analysts" in s and "rate" in s
    for advice_word in ("you should", "we recommend", "our recommendation", "buy now"):
        assert advice_word not in s


def test_ratings_empty_distribution_is_no_consensus():
    # No ratings -> total 0 and an explicit "no ratings" summary; the caller records an uncertainty
    # rather than inventing a consensus.
    d = summarize_distribution({})
    assert d["total"] == 0 and d["modal"] is None
    assert "no analyst ratings" in d["summary"].lower()


# --- edgar html stripping (pure) -------------------------------------------------------------------
def test_strip_html_removes_tags_and_scripts():
    html = "<html><script>var x=1;</script><body><p>Revenue &amp; growth</p></body></html>"
    text = strip_html(html)
    assert "var x" not in text and "Revenue & growth" in text
