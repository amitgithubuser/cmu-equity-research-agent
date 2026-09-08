"""Resolve one analysis window from the user's wording and optional explicit cutoff.

The same result drives transcript acquisition, market-data selection, trend construction, and the
Planner prompt. Resolving once prevents each layer from interpreting "annual" or "latest quarter"
differently.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from .config import get_settings
from .periods import canonical_label, normalize_period

WindowMode = Literal["latest_quarter", "annual", "recent_quarters"]

_QUARTER_IN_TEXT = re.compile(
    r"\b(?:Q([1-4])[-\s]?(?:FY)?(20\d{2})|(?:FY)?(20\d{2})[-\s]?Q([1-4]))\b",
    re.IGNORECASE,
)
_FISCAL_YEAR_IN_TEXT = re.compile(r"\bFY[-\s]?(20\d{2})\b", re.IGNORECASE)
_LATEST_QUARTER = re.compile(
    r"\b(latest|current|this|most recent)\s+(quarter|qtr)\b|\blast\s+(quarter|qtr)\b",
    re.IGNORECASE,
)
_ANNUAL = re.compile(
    r"\b(annual|annually|full[-\s]?year|fiscal year|yearly|year over year|yoy|ttm)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnalysisWindow:
    """Normalized scope shared by every source and calculation step."""

    mode: WindowMode
    n_quarters: int
    cutoff: str
    label: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _period_from_question(question: str) -> str:
    match = _QUARTER_IN_TEXT.search(question or "")
    if match:
        quarter = match.group(1) or match.group(4)
        year = match.group(2) or match.group(3)
        return f"Q{quarter}-{year}"
    match = _FISCAL_YEAR_IN_TEXT.search(question or "")
    return f"FY{match.group(1)}" if match else ""


def resolve_analysis_window(
    question: str = "",
    as_of: str = "",
    *,
    default_quarters: int | None = None,
) -> AnalysisWindow:
    """Apply the user-visible window contract.

    Priority: explicit ``as_of`` → explicit period in the question → period-intent wording → the
    default recent-quarter view. A fiscal-year cutoff means four quarters; a quarter cutoff means one.
    """
    default_quarters = default_quarters or get_settings().reason_quarters
    explicit = canonical_label(as_of) or _period_from_question(question)
    ref = normalize_period(explicit)
    if ref and ref.is_quarterly:
        return AnalysisWindow(
            mode="latest_quarter",
            n_quarters=1,
            cutoff=ref.label,
            label=ref.label,
            reason="specific quarter requested",
        )
    if ref and not ref.is_quarterly:
        return AnalysisWindow(
            mode="annual",
            n_quarters=4,
            cutoff=ref.label,
            label=f"{ref.label} from four quarters",
            reason="fiscal-year boundary requested",
        )
    if _LATEST_QUARTER.search(question or ""):
        return AnalysisWindow(
            mode="latest_quarter",
            n_quarters=1,
            cutoff="",
            label="latest quarter",
            reason="latest-quarter wording detected",
        )
    if _ANNUAL.search(question or ""):
        return AnalysisWindow(
            mode="annual",
            n_quarters=4,
            cutoff="",
            label="latest four quarters (TTM)",
            reason="annual/TTM wording detected",
        )
    n_quarters = max(2, int(default_quarters))
    return AnalysisWindow(
        mode="recent_quarters",
        n_quarters=n_quarters,
        cutoff="",
        label=f"latest quarter + {n_quarters - 1} prior quarters",
        reason="no period specified",
    )


__all__ = ["AnalysisWindow", "WindowMode", "resolve_analysis_window"]
