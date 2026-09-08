"""Market data — `market_lookup` (raw line items) and `vendor_ratio` (cross-check).  ⟶ §7.1, D-3

The deliberate split (CP 2.1 §3): the data source exposes *both* pre-computed ratios and raw line
items. We fetch **raw line items** here and compute ratios ourselves in `calc` (provenance +
definitional consistency). `vendor_ratio` pulls the vendor's *pre-computed* ratio and is used
**only as a cross-check** — a sharp divergence gets flagged rather than silently trusted.

Runtime = real services only: `market_lookup`/`vendor_ratio` call yfinance (lazy-imported so the
pure divergence helper stays offline-testable).
"""

from __future__ import annotations

from ..config import get_settings
from .base import tool


# --------------------------------------------------------------------------- pure, testable helper
def divergence(our_value: float, vendor_value: float, rel_threshold: float | None = None) -> tuple[float, bool]:
    """Relative divergence |our - vendor| / |vendor| and whether it exceeds the threshold (OQ-4).

    Pure function — no network — so the flag logic is unit-testable offline. A zero vendor value is
    treated as "cannot compare" (rel = inf, flagged) so we never divide by zero or hide a mismatch.
    """
    if rel_threshold is None:
        rel_threshold = get_settings().vendor_divergence_rel
    if vendor_value == 0:
        return (float("inf"), True)
    rel = abs(our_value - vendor_value) / abs(vendor_value)
    return (rel, rel > rel_threshold)


# --------------------------------------------------------------------------- network tools
@tool
def market_lookup(ticker: str, period: str = "annual") -> dict:
    """Fetch current price and RAW statement line items for a ticker (not ratios).

    Args:
        ticker: US-listed symbol, e.g. "NVDA".
        period: "annual" or "quarterly".

    Returns:
        {"ticker", "period", "price", "line_items": {revenue, cogs, operating_income, net_income,
        total_debt, total_equity, current_assets, current_liabilities, free_cash_flow}, "as_of"}.
        Missing line items are omitted rather than guessed.
    """
    import yfinance as yf  # lazy — only needed on a real run

    tk = yf.Ticker(ticker)
    quarterly = period.lower().startswith("q")
    income = tk.quarterly_income_stmt if quarterly else tk.income_stmt
    balance = tk.quarterly_balance_sheet if quarterly else tk.balance_sheet
    cash = tk.quarterly_cashflow if quarterly else tk.cashflow

    def _first(df, *names):
        """Most recent value for the first matching row label, or None."""
        if df is None or getattr(df, "empty", True):
            return None
        for name in names:
            if name in df.index:
                series = df.loc[name].dropna()
                if not series.empty:
                    return float(series.iloc[0])
        return None

    line_items = {
        "revenue": _first(income, "Total Revenue", "Operating Revenue"),
        "cogs": _first(income, "Cost Of Revenue", "Cost Of Goods Sold"),
        "operating_income": _first(income, "Operating Income", "Total Operating Income As Reported"),
        "net_income": _first(income, "Net Income", "Net Income Common Stockholders"),
        "total_debt": _first(balance, "Total Debt"),
        "total_equity": _first(balance, "Stockholders Equity", "Total Equity Gross Minority Interest"),
        "current_assets": _first(balance, "Current Assets"),
        "current_liabilities": _first(balance, "Current Liabilities"),
        "free_cash_flow": _first(cash, "Free Cash Flow"),
    }
    line_items = {k: v for k, v in line_items.items() if v is not None}

    price = None
    try:
        fast = tk.fast_info
        price = float(fast.get("last_price") or fast.get("lastPrice"))
    except Exception:  # noqa: BLE001 - price is best-effort; the brief must not depend on it silently
        price = None

    as_of = None
    if income is not None and not getattr(income, "empty", True):
        as_of = str(income.columns[0])[:10]

    return {"ticker": ticker.upper(), "period": period, "price": price,
            "line_items": line_items, "as_of": as_of}


@tool
def market_history(ticker: str, period: str = "annual") -> dict:
    """Fetch RAW statement line items for ALL available periods (for trend charts). ⟶ analytics.py

    `market_lookup` returns only the latest period; this returns the full series so `build_trends`
    can compute a per-period ratio history. Same raw-line-item discipline (D-3): we return line items,
    never vendor ratios, and omit missing items rather than guessing.

    Args:
        ticker: US-listed symbol, e.g. "NVDA".
        period: "annual" or "quarterly".

    Returns:
        {"ticker", "period", "line_items": {period_str: {revenue, cogs, operating_income, net_income,
        free_cash_flow, ...}, ...}} — one inner dict per reporting period (yfinance column).
    """
    import yfinance as yf  # lazy — only needed on a real run

    tk = yf.Ticker(ticker)
    quarterly = period.lower().startswith("q")
    income = tk.quarterly_income_stmt if quarterly else tk.income_stmt
    balance = tk.quarterly_balance_sheet if quarterly else tk.balance_sheet
    cash = tk.quarterly_cashflow if quarterly else tk.cashflow

    # label -> (statement frame, alternative row names yfinance may use)
    fields = {
        "revenue": (income, ("Total Revenue", "Operating Revenue")),
        "cogs": (income, ("Cost Of Revenue", "Cost Of Goods Sold")),
        "operating_income": (income, ("Operating Income", "Total Operating Income As Reported")),
        "net_income": (income, ("Net Income", "Net Income Common Stockholders")),
        "total_debt": (balance, ("Total Debt",)),
        "total_equity": (balance, ("Stockholders Equity", "Total Equity Gross Minority Interest")),
        "current_assets": (balance, ("Current Assets",)),
        "current_liabilities": (balance, ("Current Liabilities",)),
        "free_cash_flow": (cash, ("Free Cash Flow",)),
    }

    def _row(df, names):
        """The matching row Series (indexed by period), or None."""
        if df is None or getattr(df, "empty", True):
            return None
        for name in names:
            if name in df.index:
                return df.loc[name]
        return None

    line_items: dict[str, dict] = {}
    for key, (df, names) in fields.items():
        row = _row(df, names)
        if row is None:
            continue
        for col, val in row.dropna().items():
            period_str = str(col)[:10]
            line_items.setdefault(period_str, {})[key] = float(val)

    return {"ticker": ticker.upper(), "period": period, "line_items": line_items}


@tool
def market_quarters(ticker: str, n: int = 4) -> dict:
    """Fetch RAW line items for the latest `n` QUARTERS, ordered latest-first (D-13, fix F-04).

    This is the quarterly-first data path: `market_lookup` returns a single period and `market_history`
    returns everything unordered; this returns exactly the trailing `n` quarters in recency order so the
    Researcher can (a) answer a quarterly question from the latest quarter and (b) roll up four quarters
    into a TTM view for a yearly question (`tools.rollup.combine_quarters`). Same raw-line-item discipline
    (D-3): line items only, missing items omitted rather than guessed.

    Args:
        ticker: US-listed symbol, e.g. "NVDA".
        n: how many trailing quarters to return (default 4 = one year).

    Returns:
        {"ticker", "quarters": [{"period": "<report date>", "line_items": {...}}, ...]} — latest first.
    """
    import yfinance as yf  # lazy — only needed on a real run

    tk = yf.Ticker(ticker)
    income = tk.quarterly_income_stmt
    balance = tk.quarterly_balance_sheet
    cash = tk.quarterly_cashflow

    fields = {
        "revenue": (income, ("Total Revenue", "Operating Revenue")),
        "cogs": (income, ("Cost Of Revenue", "Cost Of Goods Sold")),
        "operating_income": (income, ("Operating Income", "Total Operating Income As Reported")),
        "net_income": (income, ("Net Income", "Net Income Common Stockholders")),
        "total_debt": (balance, ("Total Debt",)),
        "total_equity": (balance, ("Stockholders Equity", "Total Equity Gross Minority Interest")),
        "current_assets": (balance, ("Current Assets",)),
        "current_liabilities": (balance, ("Current Liabilities",)),
        "free_cash_flow": (cash, ("Free Cash Flow",)),
    }

    def _row(df, names):
        if df is None or getattr(df, "empty", True):
            return None
        for name in names:
            if name in df.index:
                return df.loc[name]
        return None

    # gather per-period line items, keyed by report date; yfinance columns are already newest-first.
    by_period: dict[str, dict] = {}
    order: list[str] = []
    if income is not None and not getattr(income, "empty", True):
        order = [str(c)[:10] for c in income.columns]
    for key, (df, names) in fields.items():
        row = _row(df, names)
        if row is None:
            continue
        for col, val in row.dropna().items():
            period_str = str(col)[:10]
            by_period.setdefault(period_str, {})[key] = float(val)
            if period_str not in order:
                order.append(period_str)

    # newest-first; if we couldn't read the income columns, fall back to sorted-desc by date string.
    if not order:
        order = sorted(by_period, reverse=True)
    quarters = [{"period": p, "line_items": by_period[p]} for p in order if p in by_period][:n]
    return {"ticker": ticker.upper(), "quarters": quarters}


@tool
def vendor_ratio(ticker: str, metric: str) -> dict:
    """Pull the vendor's PRE-COMPUTED ratio for cross-check only (calc is the source of truth).

    Returns {"ticker", "metric", "vendor_value"} — vendor_value may be None if the vendor lacks it.
    """
    import yfinance as yf  # lazy

    info = yf.Ticker(ticker).info or {}
    vendor_map = {
        "gross_margin": "grossMargins",
        "operating_margin": "operatingMargins",
        "net_margin": "profitMargins",
        "revenue_growth": "revenueGrowth",
        "debt_to_equity": "debtToEquity",
        "current_ratio": "currentRatio",
    }
    raw = info.get(vendor_map.get(metric, ""))
    value = None
    if raw is not None:
        value = float(raw)
        # yfinance reports debtToEquity as a percentage (e.g. 45.2 for 0.452).
        if metric == "debt_to_equity" and value > 5:
            value = value / 100.0
    return {"ticker": ticker.upper(), "metric": metric, "vendor_value": value}
