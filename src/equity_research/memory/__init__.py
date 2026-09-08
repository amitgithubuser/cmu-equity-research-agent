"""Two memory tiers (architecture §6.1, §6.2, CP 2.1).

* **Working (short-term) memory** = context management over the current run's *narrative*: prune stale
  tool output, compact older turns near the budget — while the durable findings are never discarded.
  See `working.py`.
* **Long-term memory** = a cross-run record of each company's prior-quarter numbers + management tone,
  which is what powers the quarter-over-quarter comparison. See `longterm.py`. (Filings persist in the
  Chroma vector index — that's `rag/`.)

The one rule tying them together (§6.2): compressing the narrative is fine; the saved findings —
facts, sources, metrics — are never dropped. Working memory compresses; long-term memory persists.
"""

from .longterm import (
    InMemoryLongTermStore,
    JSONLongTermStore,
    PeriodRecord,
    compare_to_prior,
    get_longterm_store,
)
from .working import (
    FINDINGS_KEYS,
    compact_log,
    compact_state,
    prune_observation,
    would_overflow,
)

__all__ = [
    "FINDINGS_KEYS",
    "InMemoryLongTermStore",
    "JSONLongTermStore",
    "PeriodRecord",
    "compact_log",
    "compact_state",
    "compare_to_prior",
    "get_longterm_store",
    "prune_observation",
    "would_overflow",
]
