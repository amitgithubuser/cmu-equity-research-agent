"""Free earnings-call transcript discovery through public web search.

Every real run searches again for the transcript window resolved from the user request. Results are
not treated as evidence until the underlying document is downloaded, parsed, identified as a real
earnings-call transcript, and assigned a reporting quarter. Company investor-relations pages and
their document CDNs are preferred; search-result snippets are never indexed as transcript text.
"""

from __future__ import annotations

import io
import re
from base64 import urlsafe_b64decode
from html import unescape
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from ..config import get_settings
from ..periods import period_at_or_before
from ..text_cleaning import clean_public_text
from .base import tool

_GOOGLE_SEARCH_URL = "https://www.google.com/search"
_BING_SEARCH_URL = "https://www.bing.com/search"
_MAX_SEARCH_BYTES = 2_000_000
_MAX_DOCUMENT_BYTES = 20_000_000
_MIN_TRANSCRIPT_CHARS = 1_200
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 "
    "EquityResearchCapstone/0.1"
)

_PERIOD_PATTERNS = (
    re.compile(r"\bQ([1-4])\s*(?:FY)?\s*[-_/ ]?\s*(20\d{2}|\d{2})\b", re.IGNORECASE),
    re.compile(
        r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter"
        r"(?:\s+(?:of\s+)?(?:fiscal\s+year|fiscal|FY))?\s+(20\d{2}|\d{2})\b",
        re.IGNORECASE,
    ),
)
_QUARTER_WORDS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
}


def select_periods(rows: list[dict], n_quarters: int, as_of: str = "") -> list[dict]:
    """Select newest unique transcript rows within the requested boundary."""
    eligible: dict[str, dict] = {}
    for row in rows or []:
        period = row.get("period")
        if period and period_at_or_before(period, as_of):
            current = eligible.get(period)
            if current is None or _source_priority(row.get("url", "")) > _source_priority(
                current.get("url", "")
            ):
                eligible[period] = row
    ordered = sorted(eligible.values(), key=lambda row: _period_key(row["period"]), reverse=True)
    return ordered[:max(1, int(n_quarters))]


def extract_period(text: str) -> str:
    """Extract a canonical ``Qn-YYYY`` label from transcript text, title, or URL."""
    sample = unescape(text or "")
    for index, pattern in enumerate(_PERIOD_PATTERNS):
        match = pattern.search(sample)
        if not match:
            continue
        raw_quarter, raw_year = match.group(1).lower(), match.group(2)
        quarter = int(raw_quarter) if index == 0 else _QUARTER_WORDS[raw_quarter]
        year = int(raw_year)
        if year < 100:
            year += 2000
        return f"Q{quarter}-{year}"
    return ""


def previous_periods(period: str, count: int) -> list[str]:
    """Return ``period`` followed by its previous fiscal quarters."""
    match = re.fullmatch(r"Q([1-4])-(20\d{2})", period or "")
    if not match:
        return []
    quarter, year = int(match.group(1)), int(match.group(2))
    out = []
    for _ in range(max(0, count)):
        out.append(f"Q{quarter}-{year}")
        quarter -= 1
        if quarter == 0:
            quarter, year = 4, year - 1
    return out


def parse_google_links(html: str) -> list[str]:
    """Extract de-duplicated public document links from Google result HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = unescape(anchor["href"])
        if href.startswith("/url?"):
            params = parse_qs(urlparse(href).query)
            href = (params.get("q") or params.get("url") or [""])[0]
        href = unquote(href)
        if _trusted_document_url(href) and href not in links:
            links.append(href)
    return links


def parse_bing_links(html: str) -> list[str]:
    """Extract Bing result URLs, including its base64-wrapped ``/ck/a`` redirect format."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    links: list[str] = []
    for anchor in soup.select("li.b_algo h2 a[href]"):
        href = unescape(anchor["href"])
        parsed = urlparse(href)
        if parsed.hostname and parsed.hostname.endswith("bing.com") and parsed.path == "/ck/a":
            encoded = (parse_qs(parsed.query).get("u") or [""])[0]
            if encoded.startswith("a1"):
                try:
                    payload = encoded[2:] + "=" * (-len(encoded[2:]) % 4)
                    href = urlsafe_b64decode(payload).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    continue
        if _trusted_document_url(href) and href not in links:
            links.append(href)
    return links


def _trusted_document_url(url: str) -> bool:
    """Bound downloads to public, transcript-like investor-relations documents."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host or host == "localhost":
        return False
    if host.startswith(("127.", "10.", "192.168.", "169.254.")) or host.endswith(".local"):
        return False
    if host.endswith(("google.com", "googleusercontent.com")):
        return False
    haystack = f"{host}{parsed.path}".lower()
    trusted_hint = any(
        hint in haystack
        for hint in ("investor", "investors", "q4cdn.com", "earnings", "financial", "transcript")
    )
    blocked = any(site in host for site in ("reddit.com", "youtube.com", "facebook.com"))
    return trusted_hint and not blocked


def _period_key(period: str) -> tuple[int, int]:
    match = re.fullmatch(r"Q([1-4])-(20\d{2})", period or "")
    return (int(match.group(2)), int(match.group(1))) if match else (0, 0)


def _source_priority(url: str) -> int:
    """Prefer issuer investor-relations hosts, then their common Q4 document CDN."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host.startswith(("investor.", "investors.", "ir.")) or ".investor." in host:
        return 3
    if host.endswith("q4cdn.com"):
        return 2
    return 1


def _targets(n_quarters: int, as_of: str) -> list[str]:
    quarter = re.fullmatch(r"Q([1-4])-(20\d{2})", as_of or "")
    if quarter:
        return [as_of]
    fiscal = re.fullmatch(r"FY(20\d{2})", as_of or "")
    if fiscal:
        year = int(fiscal.group(1))
        return [f"Q{q}-{year}" for q in range(4, 0, -1)][:n_quarters]
    return []


def _safe_response(requests_module, url: str, *, params: dict | None, timeout: int, max_bytes: int):
    """Fetch bounded public content and expose no request URL through raised exceptions."""
    try:
        response = requests_module.get(
            url,
            params=params,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.content
        if len(content) > max_bytes:
            raise ValueError("document exceeds download limit")
        return response
    except Exception as exc:  # noqa: BLE001 - sanitize every network/library failure at this boundary
        raise RuntimeError(f"public transcript request failed ({type(exc).__name__})") from None


def _search_links(requests_module, query: str, timeout: int) -> list[str]:
    try:
        response = _safe_response(
            requests_module,
            _GOOGLE_SEARCH_URL,
            params={"q": query, "num": 10, "hl": "en"},
            timeout=timeout,
            max_bytes=_MAX_SEARCH_BYTES,
        )
        links = parse_google_links(response.text)
        if links:
            return links
    except RuntimeError:
        pass
    response = _safe_response(
        requests_module,
        _BING_SEARCH_URL,
        params={"q": query, "count": 10, "setlang": "en-US"},
        timeout=timeout,
        max_bytes=_MAX_SEARCH_BYTES,
    )
    return parse_bing_links(response.text)


def _document_text(requests_module, url: str, timeout: int) -> tuple[str, str]:
    response = _safe_response(
        requests_module, url, params=None, timeout=timeout, max_bytes=_MAX_DOCUMENT_BYTES
    )
    final_url = str(getattr(response, "url", "") or url)
    if not _trusted_document_url(final_url):
        raise RuntimeError("public transcript redirected to an untrusted location")
    content_type = str((getattr(response, "headers", {}) or {}).get("content-type", "")).lower()
    if "pdf" in content_type or urlparse(final_url).path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return clean_public_text(text), final_url
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(response.text, "html.parser")
    # MarketBeat exposes a full public transcript supplied by Quartr. Its page also contains a
    # generated "Key Takeaways" block. Start at the prepared remarks marker so only speaker dialogue
    # and Q&A enter RAG, while retaining the public page as the citation URL.
    if (urlparse(final_url).hostname or "").lower() == "www.marketbeat.com":
        presentation = soup.select_one("#transcript #presentation")
        if presentation is not None and presentation.parent is not None:
            return clean_public_text(presentation.parent.get_text("\n", strip=True)), final_url
    return clean_public_text(soup.get_text("\n", strip=True)), final_url


def _looks_like_transcript(text: str) -> bool:
    if len(text or "") < _MIN_TRANSCRIPT_CHARS:
        return False
    lowered = text.lower()
    signals = (
        "earnings call",
        "conference call",
        "question-and-answer",
        "questions and answers",
        "operator",
        "chief financial officer",
    )
    return sum(signal in lowered for signal in signals) >= 2


def _search_query(ticker: str, target: str = "") -> str:
    period = target.replace("-", " ") if target else "latest"
    return (
        f'{ticker} {period} earnings call "transcript" '
        "(investor relations OR filetype:pdf)"
    )


def _discover(requests_module, ticker: str, target: str, timeout: int) -> list[dict]:
    rows = []
    links = _search_links(requests_module, _search_query(ticker, target), timeout)
    for url in sorted(links, key=_source_priority, reverse=True):
        try:
            text, final_url = _document_text(requests_module, url, timeout)
        except RuntimeError:
            continue
        period = extract_period(f"{final_url}\n{text[:5_000]}")
        if not period or (target and period != target) or not _looks_like_transcript(text):
            continue
        host = urlparse(final_url).hostname or "public source"
        rows.append({
            "ticker": ticker,
            "period": period,
            "date": "",
            "text": text,
            "source": f"Earnings call transcript {period} ({host})",
            "url": final_url,
        })
    return rows


def _marketbeat_history(requests_module, ticker: str, seed_url: str, timeout: int) -> list[dict]:
    """Follow a public MarketBeat transcript to that ticker's consecutive earnings history."""
    if (urlparse(seed_url).hostname or "").lower() != "www.marketbeat.com":
        return []
    seed = _safe_response(
        requests_module, seed_url, params=None, timeout=timeout, max_bytes=_MAX_DOCUMENT_BYTES
    )
    from bs4 import BeautifulSoup

    history_url = ""
    ticker_path = f"/{ticker.upper()}/earnings/"
    for anchor in BeautifulSoup(seed.text, "html.parser").find_all("a", href=True):
        candidate = urljoin(seed.url, anchor["href"])
        parsed = urlparse(candidate)
        if "/stocks/" in parsed.path and ticker_path.lower() in parsed.path.lower():
            history_url = candidate
            break
    if not history_url:
        return []
    history = _safe_response(
        requests_module, history_url, params=None, timeout=timeout, max_bytes=_MAX_DOCUMENT_BYTES
    )
    report_links: list[str] = []
    for anchor in BeautifulSoup(history.text, "html.parser").find_all("a", href=True):
        candidate = urljoin(history.url, anchor["href"]).split("#", 1)[0]
        if "/earnings/reports/" in urlparse(candidate).path and candidate not in report_links:
            report_links.append(candidate)

    rows = []
    for url in report_links[:12]:
        try:
            text, final_url = _document_text(requests_module, url, timeout)
        except RuntimeError:
            continue
        period = extract_period(f"{final_url}\n{text[:5_000]}")
        if not period or not _looks_like_transcript(text):
            continue
        rows.append({
            "ticker": ticker,
            "period": period,
            "date": "",
            "text": text,
            "source": f"Earnings call transcript {period} (www.marketbeat.com)",
            "url": final_url,
        })
    return rows


def _expand_public_history(requests_module, ticker: str, rows: list[dict], timeout: int) -> list[dict]:
    """Expand one discovered public transcript into a consecutive history when the host supports it."""
    expanded = list(rows)
    for row in rows:
        if (urlparse(row.get("url", "")).hostname or "").lower() == "www.marketbeat.com":
            expanded.extend(
                _marketbeat_history(requests_module, ticker, row["url"], timeout)
            )
            break
    return expanded


@tool
def transcript_fetch(ticker: str, n_quarters: int = 1, as_of: str = "") -> list[dict]:
    """Find and download the requested earnings-call transcript window through public web search."""
    import requests

    settings = get_settings()
    symbol = ticker.strip().upper()
    needed = max(1, int(n_quarters))
    targets = _targets(needed, as_of)
    found: list[dict] = []

    if targets:
        for target in targets:
            found.extend(_discover(requests, symbol, target, settings.source_timeout_seconds))
        if len(select_periods(found, needed, as_of)) < needed:
            found.extend(_discover(requests, symbol, "", settings.source_timeout_seconds))
    else:
        found.extend(_discover(requests, symbol, "", settings.source_timeout_seconds))
    found = _expand_public_history(requests, symbol, found, settings.source_timeout_seconds)

    if not targets:
        selected = select_periods(found, needed)
        if selected:
            # Re-search every detected target explicitly. This often promotes an issuer-hosted PDF
            # that ranked below an aggregator on the broad search, while still retaining the public
            # copy as a fallback when the company does not publish a transcript.
            for target in previous_periods(selected[0]["period"], needed):
                found.extend(_discover(requests, symbol, target, settings.source_timeout_seconds))

    selected = select_periods(found, needed, as_of)
    if len(selected) < needed:
        raise RuntimeError(
            f"only {len(selected)} of {needed} required public transcripts were found"
        )
    return selected


__all__ = [
    "extract_period",
    "parse_bing_links",
    "parse_google_links",
    "previous_periods",
    "select_periods",
    "transcript_fetch",
]
