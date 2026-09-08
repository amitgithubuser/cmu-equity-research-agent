"""Quarter roll-up — combine trailing quarters into an annual (TTM) view (D-13, fix F-04).

The data strategy (D-13): the agent pulls **quarterly** statements starting from the latest available
quarter and moving backward. When a question is *yearly* but only quarterly data is on hand, it combines
the trailing four quarters (TTM) to answer — and cites every quarter it summed.

The subtlety this module exists for: "combine four quarters" is NOT one operation. Line items are of
two kinds, and mixing them is a real accounting error:

  * FLOW items accumulate over a period (revenue, COGS, income, cash flow). A year = the SUM of its four
    quarters. Summing is correct.
  * STOCK items are a balance at a point in time (total debt/equity, current assets/liabilities). A year's
    balance is the LATEST quarter's balance, not a sum — summing four balance-sheet snapshots is nonsense
    (you'd quadruple the company's debt).

So the roll-up sums flows and takes the latest balance for stocks. Pure, dependency-free, offline-testable
— same discipline as `market.divergence` and `news.score_sentiment`.
"""

from __future__ import annotations

# Line items that ACCUMULATE over a reporting period -> sum across quarters for a TTM figure.
FLOW_ITEMS = frozenset({
    "revenue", "cogs", "operating_income", "net_income", "free_cash_flow",
})
# Line items that are a point-in-time BALANCE -> take the most recent quarter, never sum.
STOCK_ITEMS = frozenset({
    "total_debt", "total_equity", "current_assets", "current_liabilities",
})


def classify_item(key: str) -> str:
    """'flow' | 'stock' | 'unknown' for a line-item key (unknown items are treated conservatively)."""
    if key in FLOW_ITEMS:
        return "flow"
    if key in STOCK_ITEMS:
        return "stock"
    return "unknown"


def combine_quarters(quarters: list[dict], n: int = 4) -> dict:
    """Roll up the trailing `n` quarters into one annual (TTM) line-item dict.

    Args:
        quarters: quarterly statements, **latest first** — each `{"period": str, "line_items": {..}}`.
        n: how many trailing quarters make a year (4).

    Returns:
        `{"line_items": {..}, "periods": [<period labels summed/used>], "n_quarters": k,
          "complete": bool}`. FLOW items are summed across the available quarters; STOCK items take the
        latest quarter's value. `complete` is False when fewer than `n` quarters were available, so the
        caller can disclose that the TTM is partial rather than presenting it as a full year.

    An item is summed only over the quarters that actually report it; a flow missing from every quarter is
    omitted (never guessed as 0), mirroring the rest of the pipeline's "omit, don't invent" rule.
    """
    window = quarters[:n]
    periods = [q.get("period", "") for q in window]

    flows: dict[str, float] = {}
    for q in window:
        for key, val in (q.get("line_items", {}) or {}).items():
            if classify_item(key) == "flow" and isinstance(val, (int, float)):
                flows[key] = flows.get(key, 0.0) + float(val)

    stocks: dict[str, float] = {}
    # latest-first, so the FIRST quarter that reports a stock item is the most recent balance.
    for q in window:
        for key, val in (q.get("line_items", {}) or {}).items():
            if classify_item(key) == "stock" and key not in stocks and isinstance(val, (int, float)):
                stocks[key] = float(val)

    return {
        "line_items": {**flows, **stocks},
        "periods": periods,
        "n_quarters": len(window),
        "complete": len(window) >= n,
    }


def ttm_label(quarters: list[dict]) -> str:
    """Human/citation label for a trailing roll-up: 'TTM ending <latest quarter>' (D-13 decision).

    Makes explicit that the annual figure is a trailing-twelve-month SUM of quarters, not the filed annual
    number — so a reader knows exactly what was combined.
    """
    if not quarters:
        return "TTM"
    latest = quarters[0].get("period", "")
    return f"TTM ending {latest}" if latest else "TTM"
