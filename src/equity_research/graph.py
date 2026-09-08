"""The compiled LangGraph workflow for the equity research analyst.

The top-level graph deliberately follows LangGraph's standard agent/tool pattern:

    researcher -> tools -> researcher

`tools` is one LangGraph ToolNode. The individual financial, retrieval, news, and ratings
operations are tools inside that node; they are not top-level graph nodes. The remaining nodes are
meaningful checkpoints for mandatory source preparation, evidence completeness, independent thesis
review, report synthesis, release checks, and human oversight.

Five agent roles remain explicit: Planner, Researcher, Thesis Analyst, Critic, and Editor. The other
nodes are deterministic workflow, safety, persistence, or human-intervention steps.
"""

from __future__ import annotations

from collections.abc import Callable

from .graph_engine import END, START, StateGraph
from .state import AgentState

# The public topology. Keeping this list small makes traces, diagrams, and failure ownership clear.
NODE_NAMES = [
    "input_guard",
    "source_setup",
    "planner",
    "researcher",
    "tools",
    "evidence_check",
    "thesis_analyst",
    "critic",
    "editor",
    "quality_gate",
    "human_review",
    "finalize",
]


def _route_after_input(state: dict) -> str:
    return "reject" if not state.get("proceed", True) else "plan"


def _route_after_researcher(state: dict) -> str:
    """A tool call enters ToolNode; a normal response means research is ready to check."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    return "tools" if getattr(last, "tool_calls", None) else "evidence_check"


def _route_after_evidence_check(state: dict) -> str:
    if state.get("need_more") and state.get("retries", 0) < state.get("max_retries", 0):
        return "researcher"
    return "thesis"


def _route_after_critic(state: dict) -> str:
    return "expand" if state.get("depth", 0) < state.get("max_depth", 3) else "editor"


def _route_after_quality(state: dict) -> str:
    """One release decision after confidence, source, monitor, and completeness checks."""
    decision = state.get("quality_decision")
    if decision == "repair":
        return "repair"
    if decision == "human_review":
        return "human"
    if decision == "pass":
        return "finalize"
    repairable = bool(state.get("unsourced") or state.get("unsupported"))
    # source_guard increments the counter when it requests a repair, so equality still represents
    # the final allowed repair pass (for max=1, round 1 must route back once).
    under_cap = state.get("editor_rounds", 0) <= state.get("editor_reretrieve_max", 1)
    if repairable and under_cap:
        return "repair"
    if state.get("needs_review"):
        return "human"
    return "finalize"


def _route_after_human(state: dict) -> str:
    if (
        state.get("human_decision") == "revise"
        and state.get("human_revision_rounds", 0) <= state.get("max_human_revisions", 1)
    ):
        return "revise"
    return "finalize"


def build_graph(nodes: dict[str, Callable], checkpointer=None):
    """Build and compile the twelve-node application graph from injected implementations."""
    missing = [name for name in NODE_NAMES if name not in nodes]
    if missing:
        raise ValueError(f"build_graph missing node implementations: {missing}")

    graph = StateGraph(AgentState)
    for name in NODE_NAMES:
        graph.add_node(name, nodes[name])

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", _route_after_input, {"reject": END, "plan": "source_setup"}
    )
    graph.add_edge("source_setup", "planner")
    graph.add_edge("planner", "researcher")

    # Standard LangGraph ReAct shape: model selects a tool, ToolNode executes it, observation returns.
    graph.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {"tools": "tools", "evidence_check": "evidence_check"},
    )
    graph.add_edge("tools", "researcher")
    graph.add_conditional_edges(
        "evidence_check",
        _route_after_evidence_check,
        {"researcher": "researcher", "thesis": "thesis_analyst"},
    )

    # The Critic owns branch evidence gating, scoring, pruning, and BFS depth advancement.
    graph.add_edge("thesis_analyst", "critic")
    graph.add_conditional_edges(
        "critic", _route_after_critic, {"expand": "thesis_analyst", "editor": "editor"}
    )

    graph.add_edge("editor", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        _route_after_quality,
        {"repair": "researcher", "human": "human_review", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "human_review", _route_after_human, {"revise": "planner", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
