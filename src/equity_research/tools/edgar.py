"""`edgar_fetch` — download SEC filings by ticker/type (architecture §7, CP 1.1).

Uses SEC EDGAR's public REST API directly. EDGAR **requires a descriptive User-Agent** (name +
email) or it rejects requests — set `EDGAR_USER_AGENT` in `.env` for real runs. Read-only.

The returned text feeds the RAG ingestion pipeline (§6.4). `requests` is lazy-imported so the
module loads offline; parsing helpers below are pure and unit-testable.
"""

from __future__ import annotations

import re
from html import unescape

from ..config import get_settings
from ..text_cleaning import clean_public_text
from .base import tool

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_FILING_INDEX_URL = "https://www.sec.gov/cgi-bin/browse-edgar"


def strip_html(raw: str) -> str:
    """Crude but dependency-free HTML→text: drop scripts/styles/tags, collapse whitespace.

    Pure function so ingestion/tests don't need a network round-trip.
    """
    raw = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    return clean_public_text(raw)


def _headers() -> dict:
    return {"User-Agent": get_settings().edgar_user_agent, "Accept-Encoding": "gzip, deflate"}


def _cik_for(ticker: str) -> str:
    import requests

    data = requests.get(_TICKER_MAP_URL, headers=_headers(), timeout=30).json()
    t = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == t:
            return f"{int(row['cik_str']):010d}"
    raise ValueError(f"no CIK found for ticker {ticker!r}")


@tool
def edgar_fetch(ticker: str, form_type: str = "10-K", limit: int = 1) -> list[dict]:
    """Download recent SEC filings of a given type for a ticker.

    Args:
        ticker: US-listed symbol.
        form_type: "10-K", "10-Q", or "8-K".
        limit: how many most-recent filings to return.

    Returns:
        A list of {"ticker", "form_type", "period", "url", "text"} dicts (text is cleaned plain text),
        ready to hand to the RAG ingestion pipeline.
    """
    import requests

    cik = _cik_for(ticker)
    subs = requests.get(_SUBMISSIONS_URL.format(cik=int(cik)), headers=_headers(), timeout=30).json()
    recent = subs["filings"]["recent"]

    out: list[dict] = []
    for form, accession, doc, report_date in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["reportDate"]
    ):
        if form != form_type:
            continue
        acc_nodash = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
        html = requests.get(url, headers=_headers(), timeout=60).text
        out.append({
            "ticker": ticker.upper(), "form_type": form, "period": report_date,
            "url": url, "text": strip_html(html),
        })
        if len(out) >= limit:
            break
    return out
