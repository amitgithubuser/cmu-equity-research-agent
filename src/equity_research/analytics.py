"""Trend analytics (charts) — derive per-period ratio series from multi-period line items.

Why this exists: `market_lookup` fetches only the *latest* period's line items (one number per
ratio). A **trend** — the thing a research brief should actually show — needs the *series* across
periods. yfinance already returns every period as a column of the statement frames; `market_history`
(tools/market.py) surfaces that series, and `build_trends` turns it into chartable points.

Two design rules carried over from `calc` (D-3):
  * ratios are computed by the SAME `calc` functions — one definition of "margin"/"leverage" across
    the whole system, never a second definition here;
  * a period missing an input is **skipped for that ratio**, never guessed — the chart shows only
    numbers that trace to real line items (the anti-fabrication rule, applied to visuals).

Pure module: no network, no LLM — so the trend logic is unit-tested offline with hand-fed history.
"""

from __future__ import annotations

from .tools.calc import (
    fcf_margin,
    gross_margin,
    net_margin,
    operating_margin,
    revenue_growth,
)

# Ratios shown as trend lines, each mapped to (calc fn, required line-item keys). Margins are the
# core "quality" story; revenue is a level (plotted separately from the 0-1 margins).
_RATIO_SPECS = {
    "gross_margin": (gross_margin, ("revenue", "cogs")),
    "operating_margin": (operating_margin, ("operating_income", "revenue")),
    "net_margin": (net_margin, ("net_income", "revenue")),
    "fcf_margin": (fcf_margin, ("free_cash_flow", "revenue")),
}


def _period_key(p: str) -> str:
    """Sort key: ISO-ish date strings sort chronologically as plain strings."""
    return str(p)


def build_trends(
    history: dict | None,
    as_of: str | None = None,
    display_quarters: int | None = None,
) -> list[dict]:
    """Turn multi-period line items into an oldest→newest list of chartable trend points.

    Args:
        history: ``{"line_items": {period: {revenue, cogs, ...}, ...}, ...}`` as returned by the
            `market_history` tool. Periods may arrive newest-first; output is always oldest-first so
            a line chart reads left→right in time.
        as_of: optional hard cutoff. Periods after it are excluded before ratios/growth are computed.
        display_quarters: optional number of most recent points to return. YoY values are calculated
            before this display slice so a four-quarter comparator can remain hidden from the chart.

    Returns:
        ``[{"period": "2023-01-31", "revenue": 26974.0, "gross_margin": 0.564, ...}, ...]`` — each
        point carries only the ratios whose inputs were present that period (missing → omitted, not
        guessed). ``revenue_growth`` is included from the second period onward and YoY values from
        the fifth point onward.
    """
    if not history:
        return []
    from .periods import period_at_or_before

    by_period: dict = history.get("line_items", {}) or {}
    periods = sorted(
        (p for p in by_period if period_at_or_before(str(p), as_of)),
        key=_period_key,
    )   # oldest → newest, never beyond the requested as-of boundary
    points: list[dict] = []
    prev_revenue: float | None = None
    prior_points: list[dict] = []

    for period in periods:
        li = by_period.get(period, {}) or {}
        point: dict = {"period": str(period)}

        revenue = li.get("revenue")
        if revenue is not None:
            point["revenue"] = float(revenue)

        for name, (fn, needed) in _RATIO_SPECS.items():
            if all(li.get(k) is not None for k in needed):
                try:
                    point[name] = round(float(fn(**{k: float(li[k]) for k in needed})), 4)
                except (ValueError, ZeroDivisionError):
                    pass   # undefined ratio (e.g. zero revenue) → omit, don't fabricate

        # QoQ/YoY growth needs an adjacent prior period; only add once we have one.
        if revenue is not None and prev_revenue not in (None, 0):
            try:
                point["revenue_growth"] = round(revenue_growth(float(revenue), float(prev_revenue)), 4)
            except (ValueError, ZeroDivisionError):
                pass
        if len(prior_points) >= 4:
            year_ago = prior_points[-4]
            year_ago_revenue = year_ago.get("revenue")
            if revenue is not None and year_ago_revenue not in (None, 0):
                try:
                    point["revenue_growth_yoy"] = round(
                        revenue_growth(float(revenue), float(year_ago_revenue)), 4
                    )
                except (ValueError, ZeroDivisionError):
                    pass
            for name in _RATIO_SPECS:
                if name in point and name in year_ago:
                    point[f"{name}_yoy_change"] = round(point[name] - year_ago[name], 4)
        if revenue is not None:
            prev_revenue = float(revenue)

        points.append(point)
        prior_points.append(point)

    if display_quarters is not None:
        return points[-max(1, int(display_quarters)):]
    return points


# Which trend keys are 0-1 ratios (one axis) vs. absolute levels (their own axis) — used by the
# presenter/UI so margins and revenue aren't crammed onto one scale.
RATIO_KEYS = tuple(_RATIO_SPECS) + ("revenue_growth", "revenue_growth_yoy")
LEVEL_KEYS = ("revenue",)
