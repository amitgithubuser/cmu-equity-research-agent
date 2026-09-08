"""`analyst_ratings` — the consensus recommendation DISTRIBUTION, as cited evidence (D-13, F-04).

The design decision (CP 2.1/6.1, recorded in D-13): the agent reports the third-party analyst consensus
as **cited evidence it attributes**, and NEVER adopts it as its own verdict. Concretely:

  * We surface the recommendation DISTRIBUTION only — how many analysts rate strong-buy / buy / hold /
    sell / strong-sell — not price targets, not earnings-surprise history (explicitly out of scope).
  * The agent does not emit its own buy/sell call. A rating is a citation ("18 of 40 analysts rate Buy,
    per <vendor> as of <date>"), the same as any other piece of evidence — so it stays subject to the
    evidence gate and the bull/bear support logic, and the tool stays read-only / fail-safe (D-8).

`summarize_distribution` is a pure, dependency-free helper (offline-testable, like `market.divergence`
and `news.score_sentiment`); `analyst_ratings` is the network tool that pulls the raw counts.
"""

from __future__ import annotations

from .base import tool

# Canonical rating buckets, best -> worst. yfinance's recommendation frame uses these column names.
RATING_BUCKETS = ("strongBuy", "buy", "hold", "sell", "strongSell")
_LABELS = {
    "strongBuy": "Strong Buy", "buy": "Buy", "hold": "Hold",
    "sell": "Sell", "strongSell": "Strong Sell",
}


def summarize_distribution(counts: dict, as_of: str = "") -> dict:
    """Summarize a recommendation-count distribution into a cited, attributable fact (pure).

    Args:
        counts: {"strongBuy": int, "buy": int, "hold": int, "sell": int, "strongSell": int} — missing
            buckets are treated as 0.
        as_of: the period/date the vendor's snapshot is from (for the citation).

    Returns:
        {"total", "counts", "distribution" (share per bucket), "modal" (most-common bucket label),
         "summary"} — or `{"total": 0, ...}` when there are no ratings (the caller then records an
        uncertainty rather than inventing a consensus). The `summary` is phrased as an ATTRIBUTED
        observation of what analysts say, never as advice ("N of M analysts rate X").
    """
    norm = {b: int(counts.get(b, 0) or 0) for b in RATING_BUCKETS}
    total = sum(norm.values())
    if total == 0:
        return {"total": 0, "counts": norm, "distribution": {}, "modal": None,
                "summary": "no analyst ratings available"}

    distribution = {b: round(norm[b] / total, 4) for b in RATING_BUCKETS}
    modal = max(RATING_BUCKETS, key=lambda b: norm[b])
    as_of_txt = f" (as of {as_of})" if as_of else ""
    # attributed observation, not a recommendation: describe the consensus, don't make a call.
    summary = (f"{total} analysts covering; {norm[modal]} rate {_LABELS[modal]}"
               f" ({round(distribution[modal] * 100)}%){as_of_txt}. "
               f"Distribution — " + ", ".join(f"{_LABELS[b]}: {norm[b]}" for b in RATING_BUCKETS))
    return {"total": total, "counts": norm, "distribution": distribution,
            "modal": modal, "summary": summary}


@tool
def analyst_ratings(ticker: str) -> dict:
    """Fetch the analyst consensus recommendation DISTRIBUTION for a ticker (counts only, no targets).

    Read-only cross-check evidence (D-13): the counts of strong-buy/buy/hold/sell/strong-sell analysts,
    which the agent reports as an attributed observation — never its own buy/sell verdict. Price targets
    and earnings surprises are intentionally NOT fetched.

    Returns:
        {"ticker", "as_of", **summarize_distribution(...)} — `total` is 0 when the vendor has no ratings.
    """
    import yfinance as yf  # lazy — only needed on a real run

    tk = yf.Ticker(ticker)
    counts: dict = {}
    as_of = ""
    try:
        rec = tk.recommendations  # a DataFrame; most recent period is the first row
        if rec is not None and not getattr(rec, "empty", True):
            row = rec.iloc[0]
            counts = {b: row.get(b) for b in RATING_BUCKETS if b in row.index}
            if "period" in row.index:
                as_of = str(row.get("period"))
                if as_of.strip().lower() in {"0m", "current", "latest"}:
                    as_of = "current vendor snapshot"
    except Exception:  # noqa: BLE001 - a missing/renamed feed must degrade to "no ratings", never crash
        counts = {}

    return {"ticker": ticker.upper(), "as_of": as_of, **summarize_distribution(counts, as_of)}
