"""Read-only, allow-listed tools (architecture §7). The worst case is a bad *brief*, never a bad
*action* — no tool writes to the outside world.

The `TOOLS` registry is the single allow-list; the graph and MCP server both consume it.
"""

from __future__ import annotations

from .calc import calc
from .edgar import edgar_fetch
from .market import market_history, market_lookup, market_quarters, vendor_ratio
from .news import news_search
from .profile import company_profile
from .ratings import analyst_ratings
from .transcript import transcript_fetch

# The allow-list. Nothing outside this dict is callable by an agent node.
TOOLS = {
    "calc": calc,
    "market_lookup": market_lookup,
    "market_history": market_history,
    "market_quarters": market_quarters,
    "vendor_ratio": vendor_ratio,
    "edgar_fetch": edgar_fetch,
    "news_search": news_search,
    "company_profile": company_profile,
    "analyst_ratings": analyst_ratings,
    "transcript_fetch": transcript_fetch,
}

__all__ = [
    "TOOLS",
    "analyst_ratings",
    "calc",
    "company_profile",
    "edgar_fetch",
    "market_history",
    "market_lookup",
    "market_quarters",
    "news_search",
    "transcript_fetch",
    "vendor_ratio",
]
