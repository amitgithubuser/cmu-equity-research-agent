"""Public company profile lookup for the report overview."""

from __future__ import annotations

from .base import tool


@tool
def company_profile(ticker: str) -> dict:
    """Fetch a public company description and classification for an exchange-listed ticker.

    Returns a small, source-attributed profile. Missing fields stay empty rather than being guessed.
    """
    import yfinance as yf  # lazy import for the live runtime only

    info = yf.Ticker(ticker).info or {}
    symbol = ticker.upper()
    return {
        "ticker": symbol,
        "name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "business_summary": info.get("longBusinessSummary") or "",
        "website": info.get("website") or "",
        "exchange": info.get("fullExchangeName") or info.get("exchange") or "",
        "market_cap": info.get("marketCap"),
        "source": "Yahoo Finance company profile",
        "url": f"https://finance.yahoo.com/quote/{symbol}/profile/",
    }
