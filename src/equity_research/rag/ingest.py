"""RAG ingestion — offline, in advance (architecture §6.4, CP 3.1).

Pipeline: download+clean (via `edgar_fetch`) → **section-aware chunk** → embed → index with
metadata. Every chunk carries `{company, period, section, source, url}` — used BOTH to cite and to
filter (§6.4). Numbers do NOT come through here; they're looked up directly (D-4).
"""

from __future__ import annotations

from ..config import get_settings
from .chunking import chunk_document
from .store import get_store


def ingest_filing(text: str, meta: dict, settings=None, store=None) -> int:
    """Chunk one filing/transcript and add it to the vector store. Returns the chunk count.

    `meta` MUST carry at least {company, period, source, url}. `section` and `chunk` are added by
    the chunker. Passing an explicit `store` lets tests use a shared in-memory index.
    """
    settings = settings or get_settings()
    required = {"company", "period", "source"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"ingest meta is missing required keys: {sorted(missing)}")

    chunks = chunk_document(
        text, meta, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    store = store or get_store(settings)
    store.add(chunks)
    return len(chunks)


def ingest_edgar(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 1,
    settings=None,
    store=None,
    fetch_tool=None,
) -> int:
    """Fetch filings via `edgar_fetch` and ingest them. Real run only (hits EDGAR)."""
    from ..tools.edgar import edgar_fetch

    settings = settings or get_settings()
    store = store or get_store(settings)
    total = 0
    from ..periods import canonical_label

    fetch_tool = fetch_tool or edgar_fetch
    for filing in fetch_tool.invoke({"ticker": ticker, "form_type": form_type, "limit": limit}):
        report_date = filing["period"]                       # SEC reportDate, e.g. "2025-01-26"
        period = canonical_label(report_date) or report_date  # canonical "FY2025" for filter equivalence
        meta = {
            "company": filing["ticker"], "period": period, "report_date": report_date,
            "source": f"{filing['form_type']} {period}", "url": filing["url"],
            "document_type": filing["form_type"].lower().replace("-", ""),
        }
        total += ingest_filing(filing["text"], meta, settings=settings, store=store)
    return total


def ingest_filing_window(
    ticker: str,
    *,
    n_quarters: int,
    settings=None,
    store=None,
    fetch_tool=None,
) -> dict:
    """Fetch and index the current annual filing plus recent quarterly filings.

    Stable chunk ids make repeated runs idempotent. The helper returns a source-coverage report so
    `source_setup` can expose exactly what entered the Vector DB.
    """
    from ..tools.edgar import edgar_fetch

    settings = settings or get_settings()
    store = store or get_store(settings)
    fetch_tool = fetch_tool or edgar_fetch
    forms: list[str] = []
    periods: list[str] = []
    chunks = 0
    filings = 0

    for form_type, limit in (("10-K", 1), ("10-Q", max(1, min(int(n_quarters), 4)))):
        rows = fetch_tool.invoke({"ticker": ticker, "form_type": form_type, "limit": limit})
        for filing in rows:
            from ..periods import canonical_label

            report_date = filing["period"]
            period = canonical_label(report_date) or report_date
            meta = {
                "company": filing.get("ticker", ticker).upper(),
                "period": period,
                "report_date": report_date,
                "source": f"{filing.get('form_type', form_type)} {period}",
                "url": filing.get("url", ""),
                "document_type": filing.get("form_type", form_type).lower().replace("-", ""),
            }
            chunks += ingest_filing(filing["text"], meta, settings=settings, store=store)
            filings += 1
            forms.append(filing.get("form_type", form_type))
            periods.append(period)
    return {"filings": filings, "chunks": chunks, "forms": forms, "periods": periods}


def ingest_transcript_window(
    ticker: str,
    *,
    n_quarters: int,
    as_of: str = "",
    settings=None,
    store=None,
    fetch_tool=None,
) -> dict:
    """Fetch and index the transcript window required for this run.

    The provider is called every time. For an annual cutoff, all four calls share the FY retrieval
    anchor while retaining the exact transcript quarter in metadata and source labels.
    """
    from ..periods import canonical_label, normalize_period
    from ..tools.transcript import transcript_fetch

    settings = settings or get_settings()
    store = store or get_store(settings)
    fetch_tool = fetch_tool or transcript_fetch
    rows = fetch_tool.invoke({"ticker": ticker, "n_quarters": n_quarters, "as_of": as_of})
    total = 0
    periods: list[str] = []
    cutoff_ref = normalize_period(as_of)
    annual_anchor = canonical_label(as_of) if cutoff_ref and not cutoff_ref.is_quarterly else ""
    for row in rows:
        actual_period = row["period"]
        index_period = annual_anchor or actual_period
        meta = {
            "company": ticker.upper(),
            "period": index_period,
            "transcript_period": actual_period,
            "report_date": row.get("date", ""),
            "source": row.get("source") or f"Earnings call transcript {actual_period}",
            "url": row.get("url", ""),
            "document_type": "earnings_call_transcript",
        }
        total += ingest_filing(row["text"], meta, settings=settings, store=store)
        periods.append(actual_period)
    return {"transcripts": len(rows), "chunks": total, "periods": periods}


def transcript_inventory(
    ticker: str,
    *,
    n_quarters: int | None = None,
    as_of: str = "",
    settings=None,
    store=None,
) -> dict:
    """Report cached transcript coverage from metadata only; never loads transcript bodies."""
    from ..periods import normalize_period, period_at_or_before

    settings = settings or get_settings()
    store = store or get_store(settings)
    metadata = store.inventory({
        "$and": [
            {"company": ticker.upper()},
            {"document_type": "earnings_call_transcript"},
        ]
    })
    periods = {
        str(item.get("transcript_period") or item.get("period"))
        for item in metadata
        if item.get("transcript_period") or item.get("period")
    }
    periods = [period for period in periods if period_at_or_before(period, as_of)]
    periods.sort(
        key=lambda period: (
            (normalize_period(period).fiscal_year if normalize_period(period) else 0),
            (normalize_period(period).quarter if normalize_period(period) else 0) or 0,
        ),
        reverse=True,
    )
    if n_quarters is not None:
        periods = periods[:max(1, int(n_quarters))]
    return {"transcripts": len(periods), "chunks": len(metadata), "periods": periods}


def filing_inventory(
    ticker: str,
    *,
    n_quarters: int | None = None,
    as_of: str = "",
    settings=None,
    store=None,
) -> dict:
    """Report unique cached 10-K/10-Q coverage without loading filing bodies."""
    from ..periods import period_at_or_before

    settings = settings or get_settings()
    store = store or get_store(settings)
    metadata = store.inventory({"company": ticker.upper()})
    filings: dict[tuple, dict] = {}
    for item in metadata:
        document_type = str(item.get("document_type", "")).lower()
        if document_type not in {"10k", "10q"}:
            continue
        period = str(item.get("period", ""))
        report_date = str(item.get("report_date", ""))
        if not period_at_or_before(report_date or period, as_of):
            continue
        key = (document_type, item.get("url") or item.get("source") or period)
        filings[key] = item

    rows = sorted(
        filings.values(),
        key=lambda item: str(item.get("report_date") or item.get("period")),
        reverse=True,
    )
    k_rows = [item for item in rows if str(item.get("document_type", "")).lower() == "10k"][:1]
    q_limit = max(1, min(int(n_quarters or 4), 4))
    q_rows = [item for item in rows if str(item.get("document_type", "")).lower() == "10q"][:q_limit]
    selected = [*k_rows, *q_rows]
    return {
        "filings": len(selected),
        "chunks": len(metadata),
        "forms": ["10-K" for _ in k_rows] + ["10-Q" for _ in q_rows],
        "periods": [str(item.get("period", "")) for item in selected],
        "form_counts": {"10-K": len(k_rows), "10-Q": len(q_rows)},
    }
