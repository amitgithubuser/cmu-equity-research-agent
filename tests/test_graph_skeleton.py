"""Phase 4 gate (implementation-plan §5.2): the graph compiles, a stubbed run reaches END with a
brief, and every loop is provably bounded (no infinite cycles).
"""

from equity_research.graph import NODE_NAMES, build_graph
from equity_research.nodes_stub import stub_nodes


def _run(initial, max_depth=3):
    graph = build_graph(stub_nodes())
    state = {"ticker": "TEST", "question": "how are margins?", "as_of": "FY2025",
             "max_depth": max_depth, **initial}
    return graph.invoke(state)


def test_graph_compiles():
    assert build_graph(stub_nodes()) is not None


def test_public_graph_has_exactly_twelve_meaningful_nodes():
    assert NODE_NAMES == [
        "input_guard", "source_setup", "planner", "researcher", "tools",
        "evidence_check", "thesis_analyst", "critic", "editor", "quality_gate",
        "human_review", "finalize",
    ]
    rendered = build_graph(stub_nodes()).get_graph()
    app_nodes = set(rendered.nodes) - {"__start__", "__end__"}
    assert app_nodes == set(NODE_NAMES)
    assert not app_nodes.intersection({
        "researcher_dispatch", "calc_tool", "rag_retrieve", "evidence_gate",
        "keep_topk", "confidence_guard", "source_guard", "monitor",
    })


def test_graph_visualization_shows_one_agent_tool_loop():
    mermaid = build_graph(stub_nodes()).get_graph().draw_mermaid()
    assert "researcher" in mermaid and "tools" in mermaid
    assert "calc_tool" not in mermaid and "rag_retrieve" not in mermaid


def test_end_to_end_stub_returns_brief():
    out = _run({})
    assert "brief" in out
    brief = out["brief"]
    assert set(brief) >= {"ticker", "bull_case", "bear_case", "key_risks", "confidence", "disclaimer"}
    assert "not financial advice" in brief["disclaimer"].lower()


def test_tot_loop_terminates_at_depth_cap():
    out = _run({}, max_depth=3)
    assert out["depth"] == 3  # the Critic owns pruning/depth and routes to Editor at the cap


def test_input_guard_reject_short_circuits():
    # force a fail-safe reject: proceed=False should end the run before planner
    nodes = stub_nodes()
    nodes["input_guard"] = lambda s: {"proceed": False, "log": ["blocked"]}
    graph = build_graph(nodes)
    out = graph.invoke({"ticker": "TEST", "max_depth": 3})
    assert "brief" not in out  # never reached the editor/finalize


def test_editor_backedge_bounded():
    # Force the Editor to ALWAYS report unsourced -> back-edge to researcher. The back-edge must be
    # bounded by editor_reretrieve_max, after which the run proceeds to a brief instead of looping.
    nodes = stub_nodes()

    def editor_always_unsourced(s):
        return {"draft_brief": {"ticker": s.get("ticker", "TEST"), "as_of": "FY2025",
                                "bull_case": [], "bear_case": [], "key_risks": [],
                                "confidence": 0.5, "confidence_rationale": "stub",
                                "disclaimer": "This is research, not financial advice."},
                "confidence": 0.5, "unsourced": True,
                "editor_rounds": s.get("editor_rounds", 0) + 1,
                "log": ["editor: unsourced"]}

    nodes["editor"] = editor_always_unsourced
    graph = build_graph(nodes)
    out = graph.invoke({"ticker": "TEST", "max_depth": 2, "editor_reretrieve_max": 2})
    # bounded: editor_rounds stops incrementing once the cap is hit, and a brief is produced
    assert out.get("editor_rounds", 0) <= 3
    assert "brief" in out


def test_no_unbounded_loops_step_guard():
    # the engine raises if any loop runs away; a normal stub run must complete well under the cap
    graph = build_graph(stub_nodes())
    out = graph.invoke({"ticker": "TEST", "max_depth": 3})
    assert out is not None
