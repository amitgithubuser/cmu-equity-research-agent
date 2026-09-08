"""Phase 1: verify the blackboard's `Annotated[list, add]` reducers are declared (architecture §5.4).

We don't need LangGraph to check this — the reducer is attached as `Annotated` metadata, so we can
introspect it directly and confirm the append-not-clobber contract the graph relies on.
"""

from operator import add
from typing import get_args, get_type_hints

from equity_research.state import AgentState, merge_evidence


def _reducer_of(field: str):
    hints = get_type_hints(AgentState, include_extras=True)
    ann = hints[field]
    # Annotated[list[...], add] -> metadata is the trailing args
    return get_args(ann)[1] if get_args(ann) else None


def test_evidence_reducer_appends_different_tasks_and_replaces_a_repaired_task():
    assert _reducer_of("evidence") is merge_evidence
    current = [{"claim": "old", "source_task": "t1"}, {"claim": "keep", "source_task": "t2"}]
    incoming = [{"claim": "repaired a", "source_task": "t1"},
                {"claim": "repaired b", "source_task": "t1"}]
    assert merge_evidence(current, incoming) == [current[1], *incoming]


def test_log_and_uncertainties_use_add_reducer():
    assert _reducer_of("log") is add
    assert _reducer_of("open_uncertainties") is add


def test_scalar_fields_have_no_reducer():
    # confidence / route are last-write-wins; they must NOT carry the `add` reducer.
    assert _reducer_of("confidence") is not add
    assert _reducer_of("route") is not add
