"""Regression coverage for the shared analysis window and mandatory transcript source setup."""

import re
from urllib.parse import quote

from equity_research.nodes_real import source_setup_node
from equity_research.tools.transcript import (
    extract_period,
    parse_bing_links,
    parse_google_links,
    previous_periods,
    select_periods,
    transcript_fetch,
)
from equity_research.window import resolve_analysis_window


class _FakeTool:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, args):
        return self._fn(**args)


def test_window_contract_for_latest_annual_and_unspecified_questions():
    latest = resolve_analysis_window("Can you review the latest qtr?", default_quarters=4)
    annual = resolve_analysis_window("Give me an annual NVDA analysis", default_quarters=4)
    broad = resolve_analysis_window("Can you research NVDA?", default_quarters=4)

    assert (latest.mode, latest.n_quarters) == ("latest_quarter", 1)
    assert (annual.mode, annual.n_quarters) == ("annual", 4)
    assert (broad.mode, broad.n_quarters) == ("recent_quarters", 4)
    assert broad.label == "latest quarter + 3 prior quarters"


def test_explicit_period_has_priority_over_general_wording():
    window = resolve_analysis_window(
        "Give me an annual view, but stop at Q2 2025", default_quarters=4
    )
    assert window.mode == "latest_quarter"
    assert window.n_quarters == 1
    assert window.cutoff == "Q2-2025"


def test_transcript_period_selection_respects_cutoff_and_count():
    rows = [
        {"period": "Q1-2026"},
        {"period": "Q4-2025"},
        {"period": "Q3-2025"},
        {"period": "Q2-2025"},
        {"period": "Q1-2025"},
    ]
    selected = select_periods(rows, 4, "FY2025")
    assert [row["period"] for row in selected] == [
        "Q4-2025", "Q3-2025", "Q2-2025", "Q1-2025"
    ]


def test_transcript_period_selection_prefers_official_source_for_same_quarter():
    rows = [
        {"period": "Q2-2027", "url": "https://www.marketbeat.com/earnings/reports/nvda/"},
        {"period": "Q2-2027", "url": "https://investor.nvidia.com/Q2-2027-transcript.pdf"},
    ]
    selected = select_periods(rows, 1)
    assert selected[0]["url"].startswith("https://investor.nvidia.com/")


def test_source_setup_refetches_transcripts_for_every_run():
    calls = []

    def ingest(ticker, *, n_quarters, as_of="", fetch_tool):
        calls.append((ticker, n_quarters, as_of, fetch_tool is not None))
        return {"transcripts": n_quarters, "chunks": n_quarters * 2,
                "periods": [f"period-{i}" for i in range(n_quarters)]}

    tools = {"transcript_fetch": _FakeTool(lambda **kwargs: [])}
    state = {"ticker": "NVDA", "question": "Can you research NVDA?", "as_of": ""}
    first = source_setup_node(state, tools, transcript_ingest_fn=ingest)
    second = source_setup_node(state, tools, transcript_ingest_fn=ingest)

    assert calls == [("NVDA", 4, "", True), ("NVDA", 4, "", True)]
    assert first["transcript_status"] == second["transcript_status"] == "ingested"


def test_source_setup_automatically_indexes_transcripts_and_filings():
    transcript_calls = []
    filing_calls = []

    def ingest_transcripts(ticker, *, n_quarters, as_of="", fetch_tool):
        transcript_calls.append((ticker, n_quarters, as_of, fetch_tool is not None))
        return {"transcripts": n_quarters, "chunks": 8,
                "periods": [f"Q{i}-2025" for i in range(n_quarters, 0, -1)]}

    def ingest_filings(ticker, *, n_quarters, fetch_tool):
        filing_calls.append((ticker, n_quarters, fetch_tool is not None))
        return {"filings": 3, "chunks": 24, "forms": ["10-K", "10-Q", "10-Q"],
                "periods": ["2025-01-31", "2024-10-31", "2024-07-31"]}

    tools = {
        "transcript_fetch": _FakeTool(lambda **kwargs: []),
        "edgar_fetch": _FakeTool(lambda **kwargs: []),
    }
    out = source_setup_node(
        {"ticker": "NVDA", "question": "Can you research NVDA?", "as_of": ""},
        tools,
        transcript_ingest_fn=ingest_transcripts,
        filing_ingest_fn=ingest_filings,
    )

    assert transcript_calls == [("NVDA", 4, "", True)]
    assert filing_calls == [("NVDA", 4, True)]
    assert out["transcript_status"] == "ingested"
    assert out["filing_status"] == "ingested"
    assert out["filing_forms"] == ["10-K", "10-Q", "10-Q"]
    assert out["filing_chunks"] == 24


def test_source_setup_exposes_missing_transcript_source_as_uncertainty():
    out = source_setup_node(
        {"ticker": "NVDA", "question": "Can you research NVDA?", "as_of": ""},
        {},
    )
    assert out["transcript_status"] == "unavailable"
    assert out.get("open_uncertainties")


def test_source_setup_does_not_serialize_secret_bearing_provider_errors():
    secret = "credential-that-must-stay-out-of-state"

    def ingest(*args, **kwargs):
        raise RuntimeError(f"request failed: https://provider.test?apikey={secret}")

    out = source_setup_node(
        {"ticker": "NVDA", "question": "Can you research NVDA?", "as_of": ""},
        {"transcript_fetch": _FakeTool(lambda **kwargs: [])},
        transcript_ingest_fn=ingest,
    )
    assert out["transcript_status"] == "unavailable"
    assert secret not in str(out)


def test_source_setup_uses_cached_transcript_metadata_after_refresh_failure():
    def ingest(*args, **kwargs):
        raise RuntimeError("public search temporarily unavailable")

    inventory_calls = []

    def inventory(ticker, *, n_quarters, as_of):
        inventory_calls.append((ticker, n_quarters, as_of))
        return {"transcripts": 4, "chunks": 28,
                "periods": ["Q2-2027", "Q1-2027", "Q4-2026", "Q3-2026"]}

    out = source_setup_node(
        {"ticker": "NVDA", "question": "Can you research NVDA?", "as_of": ""},
        {"transcript_fetch": _FakeTool(lambda **kwargs: [])},
        transcript_ingest_fn=ingest,
        transcript_inventory_fn=inventory,
    )
    assert inventory_calls == [("NVDA", 4, "")]
    assert out["transcript_status"] == "cached"
    assert out["transcript_chunks"] == 28
    assert "previously indexed" in " ".join(out["open_uncertainties"])


def test_period_extraction_and_prior_quarter_sequence():
    assert extract_period("NVIDIA Q2 FY27 Earnings Call Corrected Transcript") == "Q2-2027"
    assert extract_period("Fourth Quarter Fiscal 2026 earnings call") == "Q4-2026"
    assert previous_periods("Q2-2027", 4) == [
        "Q2-2027", "Q1-2027", "Q4-2026", "Q3-2026"
    ]


def test_google_link_parser_prefers_public_investor_documents():
    official = "https://investor.example.com/financials/Q4-2025-transcript.html"
    cdn = "https://s1.q4cdn.com/files/Q3-2025-transcript.pdf"
    html = (
        f'<a href="/url?q={quote(official)}&sa=U">Official</a>'
        f'<a href="{cdn}">CDN</a>'
        '<a href="http://127.0.0.1/private">Private</a>'
        '<a href="https://www.google.com/preferences">Google</a>'
    )
    assert parse_google_links(html) == [official, cdn]


def test_bing_link_parser_decodes_redirected_investor_document():
    import base64

    official = "https://investor.example.com/financials/Q4-2025-transcript.pdf"
    encoded = base64.urlsafe_b64encode(official.encode()).decode().rstrip("=")
    html = (
        '<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?'
        f'u=a1{encoded}">Official</a></h2></li>'
    )
    assert parse_bing_links(html) == [official]


class _Response:
    def __init__(self, text, url, content_type="text/html"):
        self.text = text
        self.content = text.encode()
        self.url = url
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        return None


def test_public_search_fetches_exact_annual_transcript_window_without_api_key(monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        if "google.com/search" in url or "bing.com/search" in url:
            target = re.search(r"Q([1-4])\s+2025", params["q"])
            assert target
            quarter = target.group(1)
            transcript_url = (
                f"https://investor.example.com/financials/"
                f"NVDA-Q{quarter}-2025-transcript.html"
            )
            html = f'<a href="/url?q={quote(transcript_url)}&sa=U">Transcript</a>'
            return _Response(html, url)
        quarter = re.search(r"Q([1-4])-2025", url).group(1)
        body = (
            f"NVDA Q{quarter} 2025 Earnings Call Transcript\n"
            "Operator: Welcome to the quarterly conference call.\n"
            "Chief Financial Officer: We will discuss revenue, margins, demand, and guidance.\n"
            "Question-and-Answer Session\n" + ("Management commentary and analyst question. " * 50)
        )
        return _Response(body, url)

    monkeypatch.setattr("requests.get", fake_get)
    result = transcript_fetch.invoke({"ticker": "NVDA", "n_quarters": 4, "as_of": "FY2025"})

    assert [row["period"] for row in result] == [
        "Q4-2025", "Q3-2025", "Q2-2025", "Q1-2025"
    ]
    assert len(calls) == 8  # one search + one document download for each quarter
    assert all("apikey" not in (call[1] or {}) for call in calls)
    assert all(row["url"].startswith("https://investor.example.com/") for row in result)
