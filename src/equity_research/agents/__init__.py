"""The five agent nodes (architecture §4). Each is `(state, llm, ...) -> partial state` so tests can
inject a fake `llm`/`tools` and the graph runs without spending tokens.
"""

from .editor import editor_node
from .planner import planner_node
from .researcher import (
    build_research_tool_node,
    classify_route,
    researcher_agent_node,
    researcher_node,
)
from .thesis import analyst_node, critic_node

__all__ = [
    "analyst_node",
    "build_research_tool_node",
    "classify_route",
    "critic_node",
    "editor_node",
    "planner_node",
    "researcher_agent_node",
    "researcher_node",
]
