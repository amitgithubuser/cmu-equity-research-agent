"""`calc` — deterministic financial ratios computed in code, never by an LLM (architecture §7.1).

This is the reason numbers are trustworthy: computing ratios ourselves buys **provenance** (each
number ties back to specific line items) and **definitional consistency** (one definition of
"margin"/"debt" across every company, not the vendor's undocumented TTM/annual choice). It is the
LLM-arithmetic risk from CP 2.1 designed out. Build it first, test it hardest.

Each ratio is a plain function with ONE documented definition. `calc(metric, inputs)` dispatches to
it and returns `{name, value, inputs}` so the number carries its provenance.
"""

from __future__ import annotations

from .base import tool


# --------------------------------------------------------------------------- ratio definitions
def gross_margin(revenue: float, cogs: float) -> float:
    """(revenue - cost of goods sold) / revenue. Higher = more pricing power."""
    if revenue == 0:
        raise ValueError("revenue is zero — gross_margin undefined")
    return (revenue - cogs) / revenue


def operating_margin(operating_income: float, revenue: float) -> float:
    """operating income / revenue."""
    if revenue == 0:
        raise ValueError("revenue is zero — operating_margin undefined")
    return operating_income / revenue


def net_margin(net_income: float, revenue: float) -> float:
    """net income / revenue."""
    if revenue == 0:
        raise ValueError("revenue is zero — net_margin undefined")
    return net_income / revenue


def revenue_growth(current: float, prior: float) -> float:
    """(current - prior) / prior. YoY or QoQ depending on which periods are passed."""
    if prior == 0:
        raise ValueError("prior revenue is zero — growth undefined")
    return (current - prior) / prior


def debt_to_equity(total_debt: float, total_equity: float) -> float:
    """total debt / total shareholders' equity. Leverage."""
    if total_equity == 0:
        raise ValueError("equity is zero — debt_to_equity undefined")
    return total_debt / total_equity


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """current assets / current liabilities. Short-term liquidity."""
    if current_liabilities == 0:
        raise ValueError("current liabilities are zero — current_ratio undefined")
    return current_assets / current_liabilities


def fcf_margin(free_cash_flow: float, revenue: float) -> float:
    """free cash flow / revenue. Cash-generation quality."""
    if revenue == 0:
        raise ValueError("revenue is zero — fcf_margin undefined")
    return free_cash_flow / revenue


# The registry maps metric name -> (function, required input keys). One definition each.
_METRICS = {
    "gross_margin": gross_margin,
    "operating_margin": operating_margin,
    "net_margin": net_margin,
    "revenue_growth": revenue_growth,
    "debt_to_equity": debt_to_equity,
    "current_ratio": current_ratio,
    "fcf_margin": fcf_margin,
}

SUPPORTED_METRICS = tuple(_METRICS)


# --------------------------------------------------------------------------- the tool
@tool
def calc(metric: str, inputs: dict) -> dict:
    """Compute a derived financial ratio in code from raw line items.

    Args:
        metric: one of the supported ratio names (see SUPPORTED_METRICS).
        inputs: the raw line items the ratio needs, e.g. {"revenue": 100.0, "cogs": 60.0}.

    Returns:
        {"name", "value", "inputs"} — value rounded to 4 dp; inputs echoed for provenance (§7.1).
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; supported: {', '.join(SUPPORTED_METRICS)}")
    fn = _METRICS[metric]
    value = fn(**inputs)
    return {"name": metric, "value": round(float(value), 4), "inputs": inputs}
