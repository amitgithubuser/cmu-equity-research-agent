"""Thin MCP exposure of the read-only tools (architecture §7.2).

MCP is the standard connector for the data-source tools and carries the shared retrieved-evidence
cache so a passage is fetched once and reused across ToT branches. The tool *logic* lives in the
sibling modules; this file only registers them — kept intentionally thin (implementation-plan §3.4).

If the MCP SDK isn't installed, `build_server()` raises a clear message rather than failing at
import time, so the rest of the package (and the tests) load fine without MCP. Per the plan, MCP is
"connector polish, not a blocker": the same tools work as plain LangChain tools inside the graph.
"""

from __future__ import annotations

from . import TOOLS


def build_server():  # pragma: no cover - exercised only with the MCP SDK + a real run
    """Register the allow-listed tools on a FastMCP server and return it. Run with `.run()`."""
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "MCP SDK not installed. Install `mcp` to expose tools over MCP, or use the tools "
            "directly as LangChain tools inside the graph (they work either way)."
        ) from exc

    server = FastMCP("equity-research-tools")
    for name, tool_obj in TOOLS.items():
        fn = getattr(tool_obj, "func", None) or getattr(tool_obj, "_fn", tool_obj)
        server.add_tool(fn, name=name, description=(getattr(tool_obj, "description", "") or fn.__doc__ or ""))
    return server


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
