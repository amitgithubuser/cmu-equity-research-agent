"""Stage 1 — pre-generation `input_guard` (architecture §9.1, Lab 6.1 `check_query_specificity`).

Runs BEFORE the Planner. Two jobs, adapting the lab's medical pattern-bank to an equity domain bank:
  * Ticker validation — a real US-listed-looking symbol (regex + optional known-symbol check).
  * Advice re-routing — detect "should I buy/sell?" phrasing and re-route to an evidence-only framing
    (or refuse), because the system is NOT financial advice.

Returns `{proceed, issues, risk_level, question}`; `proceed=False` blocks the run (fail-safe).
Pure/deterministic, so it's unit-testable and cheap (runs on Haiku in production only if we later add
an LLM classifier; the pattern bank alone is enough for the gate).
"""

from __future__ import annotations

import re

# A US ticker: 1-5 letters, optional class suffix like ".B". Deliberately permissive but not free text.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:[.\-][A-Z]{1,2})?$")

# Advice-seeking phrasing (equity-domain pattern bank).
_ADVICE_PATTERNS = [
    r"\bshould i (buy|sell|hold|short|invest)\b",
    r"\bis (it|this|now) a good time to (buy|sell)\b",
    r"\bwhat should i (do|invest)\b",
    r"\b(buy|sell) (it|now|this stock)\b",
    r"\bwill (it|the stock|shares) (go up|go down|rise|drop|crash|moon)\b",
    r"\bprice target\b.*\?",
    r"\bhow much should i (buy|invest)\b",
]
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), re.IGNORECASE)


def check_ticker(ticker: str) -> bool:
    """True if the string looks like a valid US-listed symbol (shape check)."""
    return bool(ticker) and bool(_TICKER_RE.match(ticker.strip().upper()))


def detect_advice(question: str) -> bool:
    """True if the question is asking for a buy/sell recommendation rather than research."""
    return bool(question) and bool(_ADVICE_RE.search(question))


def reframe_advice(question: str) -> str:
    """Rewrite an advice question into an evidence-only research question (re-route, not refuse)."""
    return ("What does the evidence (filings, financials, and recent news) say about the "
            "investment case — bull, bear, and key risks — for this company?")


def input_guard_node(state: dict, known_symbols: set[str] | None = None) -> dict:
    """Pre-gen gate. Blocks invalid tickers (fail-safe); reroutes advice queries to evidence framing."""
    ticker = (state.get("ticker") or "").strip().upper()
    question = state.get("question", "") or ""
    issues: list[str] = []

    if not check_ticker(ticker):
        issues.append(f"'{ticker}' is not a valid US ticker symbol")
        return {"proceed": False, "issues": issues, "risk_level": "high",
                "escalation_reason": "invalid_ticker",
                "trajectory": [{"action": "input_guard", "args": {"ticker": ticker},
                                "observation": "invalid ticker", "status": "refused"}],
                "log": [f"input_guard: REJECT invalid ticker {ticker!r}"]}

    if known_symbols is not None and ticker not in known_symbols:
        issues.append(f"'{ticker}' is not in the known-symbol list")
        return {"proceed": False, "issues": issues, "risk_level": "high",
                "escalation_reason": "unknown_ticker",
                "trajectory": [{"action": "input_guard", "args": {"ticker": ticker},
                                "observation": "unknown ticker", "status": "refused"}],
                "log": [f"input_guard: REJECT unknown ticker {ticker!r}"]}

    if detect_advice(question):
        issues.append("advice-seeking query re-routed to evidence-only framing")
        return {"proceed": True, "ticker": ticker, "question": reframe_advice(question),
                "issues": issues, "risk_level": "medium", "advice_rerouted": True,
                "escalation_reason": "advice_query",
                "trajectory": [{"action": "input_guard", "args": {"ticker": ticker},
                                "observation": "advice request reframed", "status": "ok"}],
                "log": ["input_guard: advice query re-routed to research framing"]}

    return {"proceed": True, "ticker": ticker, "issues": issues, "risk_level": "low",
            "trajectory": [{"action": "input_guard", "args": {"ticker": ticker},
                            "observation": "request accepted", "status": "ok"}],
            "log": ["input_guard: ok"]}
