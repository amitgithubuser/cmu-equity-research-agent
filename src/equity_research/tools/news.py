"""`news_search` — recent developments + headline sentiment (architecture §7).

Uses yfinance's news feed (public, no key) at runtime. A dependency-free keyword sentiment scorer
is provided as a pure helper so the sentiment logic is unit-testable offline; on a real run you can
swap in Finnhub/NewsAPI sentiment behind the same shape.
"""

from __future__ import annotations

import re

from .base import tool

_POS = {"beat", "beats", "surge", "surges", "record", "growth", "upgrade", "raises",
        "strong", "gains", "profit", "outperform", "rally", "wins", "approval"}
_NEG = {"miss", "misses", "plunge", "plunges", "lawsuit", "probe", "downgrade", "cuts",
        "weak", "loss", "decline", "warning", "recall", "delay", "investigation", "fraud"}
_LOW_SIGNAL_MARKET_HEADLINE = re.compile(
    r"\b(stocks? to buy|prediction:|will be worth|price target|big upside|best stocks?|"
    r"\$?1,?000 investment|should you buy|buy now)\b",
    re.IGNORECASE,
)


def score_sentiment(headline: str) -> float:
    """Naive lexical sentiment in [-1, 1]: (pos - neg) / (pos + neg), 0.0 if neither fires.

    Pure and deterministic — good enough as a signal and fully testable. Real runs may replace it
    with a vendor sentiment score without changing callers.
    """
    words = {w.strip(".,!?:;\"'()").lower() for w in headline.split()}
    pos = len(words & _POS)
    neg = len(words & _NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def is_company_relevant(text: str, ticker: str, company_names: tuple[str, ...] = ()) -> bool:
    """True when a headline/summary explicitly names the requested issuer.

    Ticker news feeds contain broad market stories and adjacent-company videos. Requiring the symbol
    or an issuer-name token prevents an emotionally worded but unrelated headline from winning just
    because it has the largest absolute sentiment score.
    """
    haystack = (text or "").lower()
    symbol = (ticker or "").strip().lower()
    if symbol and symbol in {w.strip(".,!?:;\"'()[]") for w in haystack.split()}:
        return True
    return any(name and name.lower() in haystack for name in company_names)


def is_material_headline(text: str) -> bool:
    """Exclude generic stock-picking/listicle headlines from company developments."""
    return bool(text and not _LOW_SIGNAL_MARKET_HEADLINE.search(text))


def _company_names(info: dict) -> tuple[str, ...]:
    """Build conservative issuer aliases from yfinance quote metadata."""
    names: list[str] = []
    for key in ("shortName", "longName"):
        raw = str((info or {}).get(key) or "").strip()
        if not raw:
            continue
        names.append(raw)
        first = raw.split()[0].strip(".,")
        if len(first) >= 4:
            names.append(first)
    return tuple(dict.fromkeys(names))


@tool
def news_search(ticker: str, limit: int = 10) -> list[dict]:
    """Fetch recent headlines for a ticker with a lightweight sentiment score.

    Returns a list of {"title", "publisher", "url", "published", "sentiment"} dicts.
    """
    import yfinance as yf  # lazy

    tk = yf.Ticker(ticker)
    items = getattr(tk, "news", None) or []
    try:
        names = _company_names(tk.info or {})
    except Exception:  # noqa: BLE001 - relevance can still fall back to the explicit ticker symbol
        names = ()
    out: list[dict] = []
    for it in items:
        content = it.get("content", it)  # yfinance shapes vary by version
        title = content.get("title") or it.get("title") or ""
        summary = content.get("summary") or content.get("description") or ""
        if not is_company_relevant(f"{title} {summary}", ticker, names) or not is_material_headline(title):
            continue
        publisher = (content.get("provider") or {}).get("displayName") or it.get("publisher") or ""
        url = ((content.get("clickThroughUrl") or {}).get("url")
               or (content.get("canonicalUrl") or {}).get("url") or it.get("link") or "")
        published = content.get("pubDate") or it.get("providerPublishTime") or ""
        out.append({
            "title": title, "publisher": publisher, "url": url,
            "published": published, "sentiment": score_sentiment(title),
        })
        if len(out) >= limit:
            break
    return out
