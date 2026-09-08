"""Working (short-term) memory — context management for one run (architecture §6.2, CP 2.1 §2).

A single run can overflow the context window: a 100-page 10-K + a full transcript + a dozen news
items. §6.2 defines **three moves** to stay under budget, governed by **one rule**:

  1. **Prune stale tool output** (*context editing*) — once a number/quote is extracted, drop the raw
     filing text; keep a short snippet + a citation pointer that can be re-fetched.
  2. **Summarize older turns** (*compaction*) near the limit — roll up early Thought/Observation steps
     into one summary line; keep the last few turns verbatim.
  3. **Write findings to durable memory as found** — each fact/source/metric persists immediately (that
     lives in `longterm.py`; here we just make sure compaction never touches those findings).

> **The rule that ties them together:** compressing the *narrative* is fine, but the saved **findings —
> facts, sources, metrics — are never discarded.** Whatever overflows working memory is exactly what
> gets persisted. So everything in this module operates on the *narrative* (scratchpad / trajectory
> text), never on `evidence` / `metrics` / `open_uncertainties`.

Pure functions over plain data — no LLM, no network — so context management is deterministic and
unit-testable. A real run can pass an LLM summarizer for step 2; the default is an extractive rollup.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import get_settings

# Keys on AgentState that hold FINDINGS — compaction/pruning must never drop or rewrite these.
# (They are the durable output; the narrative is disposable, the findings are not.)
FINDINGS_KEYS = ("evidence", "metrics", "open_uncertainties", "history", "qoq")


def prune_observation(observation: str, *, keep_chars: int = 300) -> str:
    """Context-editing (move 1): shrink a bulky raw tool observation to a short retained snippet.

    Once the Researcher has extracted the number/claim it needed, the full raw filing text is dead
    weight. We keep a leading snippet (enough to recognize + re-fetch) and mark the rest elided. The
    *finding* itself already lives in `evidence`/`metrics` with a citation, so nothing verifiable is
    lost — only the redundant bulk.
    """
    if observation is None:
        return ""
    text = str(observation)
    if len(text) <= keep_chars:
        return text
    return text[:keep_chars].rstrip() + f" …[+{len(text) - keep_chars} chars elided]"


def _extractive_summary(lines: list[str]) -> str:
    """Default compaction summarizer: a single rolled-up line naming what the early turns did.

    Deliberately lossy on *prose* but factual on *count* — the verifiable findings are untouched in
    state, so this only needs to preserve the shape of the narrative ("did N things"), not its detail.
    """
    n = len(lines)
    head = lines[0] if lines else ""
    return f"[compacted {n} earlier step(s); first: {head[:120]}]"


def compact_log(
    log: list[str],
    *,
    char_budget: int | None = None,
    keep_recent: int | None = None,
    summarizer: Callable[[list[str]], str] | None = None,
) -> list[str]:
    """Compaction (move 2): if the narrative log exceeds the char budget, summarize the older turns
    and keep the most recent `keep_recent` verbatim.

    Args:
        log: the human-readable trace lines (state["log"]).
        char_budget: soft ceiling on total chars before compaction fires (defaults to settings).
        keep_recent: how many trailing lines to keep verbatim (defaults to settings).
        summarizer: optional `(older_lines) -> str` (e.g. an LLM call on a real run); defaults to the
            deterministic extractive rollup so this is testable offline.

    Returns:
        A new log list: `[summary_line, *recent_lines]` when compacted, else the log unchanged. The
        return is idempotent-friendly — an already-small log passes through untouched.
    """
    s = get_settings()
    char_budget = s.context_char_budget if char_budget is None else char_budget
    keep_recent = s.keep_recent_turns if keep_recent is None else keep_recent
    summarizer = summarizer or _extractive_summary

    log = list(log or [])
    if sum(len(line) for line in log) <= char_budget or len(log) <= keep_recent:
        return log

    older, recent = log[:-keep_recent], log[-keep_recent:]
    return [summarizer(older), *recent]


def compact_state(
    state: dict,
    *,
    char_budget: int | None = None,
    keep_recent: int | None = None,
    summarizer: Callable[[list[str]], str] | None = None,
) -> dict:
    """Apply context management to a state snapshot WITHOUT touching any findings (the rule).

    Returns a partial-state update containing ONLY the narrative keys it rewrote (the `log`). It never
    emits a findings key (`FINDINGS_KEYS`), so merging this update back into state cannot drop or
    rewrite a fact, source, or metric — that is the §6.2 invariant, enforced by construction.
    """
    # The one invariant that makes compaction safe: findings are carried, never compressed.
    compacted = compact_log(state.get("log", []), char_budget=char_budget,
                            keep_recent=keep_recent, summarizer=summarizer)
    return {"log": compacted}


def would_overflow(state: dict, *, char_budget: int | None = None) -> bool:
    """True if the current narrative log is over budget (i.e. compaction would do something)."""
    s = get_settings()
    char_budget = s.context_char_budget if char_budget is None else char_budget
    return sum(len(line) for line in state.get("log", []) or []) > char_budget
