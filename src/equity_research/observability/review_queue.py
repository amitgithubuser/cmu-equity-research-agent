"""ReviewQueue + HITL `interrupt()` (architecture §10.3, Lab 6.2).

Lab 6.2's ReviewQueue was plain Python — a list a human polled. The architecture upgrades it to a
**LangGraph graph interrupt**: `human_review_node` calls `interrupt(payload)`, which pauses the graph
and *saves state via the checkpointer*. The reviewer is surfaced the `payload` (brief + flags +
suspicion), and the driver resumes with `graph.invoke(Command(resume={"decision": ...}))`. The resumed
run picks up inside the same node, `interrupt()` returns the decision, and the graph continues.

`ReviewQueue` is retained as the record of what was escalated (so a UI can list pending/decided
reviews); the *pausing* is the graph interrupt.

Decisions: `"allow"` (finalize as-is), `"revise"` (loop back for another pass), `"block"` (refuse to
publish the brief). The mapping to control state lives in `apply_decision`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..graph_engine import interrupt


@dataclass
class ReviewItem:
    trajectory_id: str
    brief: dict
    suspicion: float
    reason: str
    decision: str = ""          # "" until a human decides
    note: str = ""


@dataclass
class ReviewQueue:
    """In-memory record of escalated runs (a real deployment would back this with a DB/queue)."""

    items: list[ReviewItem] = field(default_factory=list)

    def enqueue(self, item: ReviewItem) -> ReviewItem:
        self.items.append(item)
        return item

    def pending(self) -> list[ReviewItem]:
        return [i for i in self.items if not i.decision]

    def resolve(self, trajectory_id: str, decision: str, note: str = "") -> None:
        for i in self.items:
            if i.trajectory_id == trajectory_id and not i.decision:
                i.decision, i.note = decision, note
                return


def review_payload(state: dict) -> dict:
    """What the reviewer sees when the graph pauses (§10.3)."""
    return {
        "trajectory_id": state.get("trajectory_id", "run"),
        "ticker": state.get("ticker", ""),
        "brief": state.get("draft_brief", {}),
        "suspicion": state.get("suspicion", 0.0),
        "reason": state.get("escalation_reason", ""),
        "confidence": state.get("confidence"),
        "flags": list(dict.fromkeys(
            (state.get("suspicion_report") or {}).get("fired", [])
            + (state.get("review_triggers") or [])
        )),
    }


def human_review_node(state: dict) -> dict:
    """HITL before finalize. `interrupt()` pauses; `Command(resume={"decision": ...})` continues.

    On a live LangGraph run the checkpointer persists state at the interrupt; here (offline engine or
    real) the resume value flows back through `interrupt()` as the reviewer's decision dict.
    """
    decision = interrupt(review_payload(state))
    # decision is whatever the driver passed to Command(resume=...): a str or {"decision", "note"}
    if isinstance(decision, dict):
        verdict = decision.get("decision", "allow")
        note = decision.get("note", "")
    else:
        verdict = str(decision or "allow")
        note = ""
    out = {"human_decision": verdict, "human_note": note,
           "trajectory": [{
               "action": "human_review",
               "args": {"decision": verdict},
               "observation": note or "reviewer supplied no note",
               "status": "refused" if verdict == "block" else "ok",
           }],
           "log": [f"human_review: decision={verdict!r}"]}
    if verdict == "revise":
        out["human_revision_rounds"] = state.get("human_revision_rounds", 0) + 1
    return out


def apply_decision(state: dict) -> dict:
    """Translate a human decision into control state for the finalize step.

    allow  → publish as-is.
    revise → the graph normally returns to the Planner. If the revision cap has already been
             reached, finalize publishes the draft with a revision-requested flag.
    block  → withhold the brief (fail-safe: refuse rather than assert unsupported claims, §10.4).
    """
    verdict = state.get("human_decision", "allow")
    if verdict == "block":
        return {"blocked": True, "log": ["decision: BLOCK — brief withheld"]}
    if verdict == "revise":
        return {"revision_requested": True, "log": ["decision: REVISE — flagged for another pass"]}
    return {"blocked": False, "log": ["decision: ALLOW — finalize"]}
