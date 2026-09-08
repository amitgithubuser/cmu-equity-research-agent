"""Canonical period model (fix F-05) — one representation for fiscal year / quarter / report date.

The bug F-05 caught: EDGAR ingestion stored the SEC *report date* (e.g. ``2025-01-26``) in the
``period`` metadata, while retrieval filtered on the user's ``--as-of`` string (e.g. ``FY2025``) with
exact equality — so the documented command retrieved **zero** passages. The market path had the mirror
problem: it hard-coded ``period="annual"`` and ignored the requested period entirely.

This module gives every layer (CLI, EDGAR, RAG, market data, memory) ONE way to talk about periods:

  * ``normalize_period(x)`` — turn any of ``"2025-01-26"`` / ``"FY2025"`` / ``"2025"`` / ``"Q1 2025"``
    / ``"2025-Q1"`` into a canonical label: ``"FY2025"`` for an annual period, ``"Q1-2025"`` for a
    quarter. Idempotent, pure, offline.
  * ``PeriodRef`` — the normalized (label, kind, fiscal_year, quarter, raw) tuple a caller can reason
    about (e.g. to decide annual-vs-quarterly market fetch).
  * ``periods_match(a, b)`` — equivalence after normalization, so ``"FY2025"`` matches a chunk stored
    from report date ``"2025-01-26"`` (both normalize to ``"FY2025"``).

Deliberately dependency-free and heuristic (no calendar/fiscal-calendar library): a report date's YEAR
is used as the fiscal year. That is correct for the common case and, crucially, *consistent* across
ingest and retrieve — which is all the filter needs. A company with an off-calendar fiscal year can
override by ingesting with an explicit fiscal label; the raw ``report_date`` is always retained too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ISO date with a validity check on month (01-12) and day (01-31) so a malformed date like
# "2025-13-01" is rejected outright rather than silently normalized to a plausible fiscal year.
_ISO_DATE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
_FY = re.compile(r"^FY[-\s]?(\d{4})$", re.IGNORECASE)
_YEAR = re.compile(r"^(\d{4})$")
# "Q1-2025", "Q1 2025", "2025-Q1", "2025Q1", "1Q2025"
_Q_PREFIX = re.compile(r"^Q([1-4])[-\s]?(\d{4})$", re.IGNORECASE)
_Q_SUFFIX = re.compile(r"^(\d{4})[-\s]?Q([1-4])$", re.IGNORECASE)


@dataclass(frozen=True)
class PeriodRef:
    """A normalized period. `label` is canonical ("FY2025" | "Q1-2025"); `raw` is what was passed in."""

    label: str
    kind: str            # "annual" | "quarterly"
    fiscal_year: int
    quarter: int | None  # 1..4 for quarterly, None for annual
    raw: str

    @property
    def is_quarterly(self) -> bool:
        return self.kind == "quarterly"


def normalize_period(value: str | None) -> PeriodRef | None:
    """Normalize any accepted period spelling to a canonical `PeriodRef`. Returns None if unparseable.

    Accepted: ISO report date ("2025-01-26"), "FY2025", "2025", "Q1-2025"/"Q1 2025"/"2025-Q1".
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    m = _Q_PREFIX.match(raw)
    if m:
        q, yr = int(m.group(1)), int(m.group(2))
        return PeriodRef(f"Q{q}-{yr}", "quarterly", yr, q, raw)
    m = _Q_SUFFIX.match(raw)
    if m:
        yr, q = int(m.group(1)), int(m.group(2))
        return PeriodRef(f"Q{q}-{yr}", "quarterly", yr, q, raw)
    m = _FY.match(raw)
    if m:
        yr = int(m.group(1))
        return PeriodRef(f"FY{yr}", "annual", yr, None, raw)
    m = _YEAR.match(raw)
    if m:
        yr = int(m.group(1))
        return PeriodRef(f"FY{yr}", "annual", yr, None, raw)
    m = _ISO_DATE.match(raw)
    if m:
        yr = int(m.group(1))
        # A report date anchors a fiscal YEAR (annual filing). Consistent across ingest+retrieve.
        return PeriodRef(f"FY{yr}", "annual", yr, None, raw)
    return None


def canonical_label(value: str | None) -> str | None:
    """The canonical label string ("FY2025"/"Q1-2025") for a period, or None if unparseable."""
    ref = normalize_period(value)
    return ref.label if ref else None


def periods_match(a: str | None, b: str | None) -> bool:
    """True if two period spellings refer to the same canonical period (F-05 filter equivalence).

    If EITHER side is unparseable, fall back to a case-insensitive exact match so a nonstandard label
    still behaves like the old exact filter (never silently over-matches).
    """
    la, lb = canonical_label(a), canonical_label(b)
    if la is not None and lb is not None:
        return la == lb
    return (a or "").strip().lower() == (b or "").strip().lower()


def period_at_or_before(candidate: str | None, cutoff: str | None) -> bool:
    """Whether a data period is on or before the requested as-of boundary.

    Annual cutoffs compare fiscal years. Quarterly cutoffs compare year+quarter; an ISO statement
    date is mapped to its calendar quarter only for that quarterly comparison. Unparseable values
    are rejected when a real cutoff is present so an unknown/future period cannot leak into a brief.
    """
    if not cutoff:
        return True
    candidate_ref = normalize_period(candidate)
    cutoff_ref = normalize_period(cutoff)
    if cutoff_ref is None:
        return periods_match(candidate, cutoff)
    if candidate_ref is None:
        return False
    if cutoff_ref.kind == "annual":
        return candidate_ref.fiscal_year <= cutoff_ref.fiscal_year

    candidate_quarter = candidate_ref.quarter
    if candidate_quarter is None:
        m = _ISO_DATE.match(str(candidate or "").strip())
        if m:
            candidate_quarter = (int(m.group(2)) - 1) // 3 + 1
        else:
            candidate_quarter = 4
    return (candidate_ref.fiscal_year, candidate_quarter) <= (
        cutoff_ref.fiscal_year,
        cutoff_ref.quarter or 4,
    )


def market_period_kind(value: str | None) -> str:
    """Map a requested period to the yfinance fetch granularity: 'quarterly' or 'annual'.

    Used by `market_lookup`/`market_history` so a quarterly question actually pulls quarterly
    statements instead of the hard-coded annual (F-05 market half).
    """
    ref = normalize_period(value)
    return "quarterly" if (ref and ref.is_quarterly) else "annual"
