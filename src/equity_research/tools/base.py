"""Adaptive `@tool` decorator (architecture §7).

At runtime (`.[real]` installed) this is LangChain's real `@tool`, so the tools are genuine
LangChain tools that can be exposed over MCP and bound to a model. Offline (lean install, tests)
it falls back to a tiny local shim that provides the same `.invoke(dict)` / `.name` / `.description`
surface the rest of the code and the tests rely on.

This is the "mockable boundary" from the plan applied to the tool layer: identical call surface,
real or shim, so `calc` and the divergence logic are unit-testable without the heavy deps.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

try:  # pragma: no cover - exercised only when .[real] is installed
    from langchain_core.tools import tool as _lc_tool

    HAVE_LANGCHAIN = True
except Exception:  # noqa: BLE001 - any import failure means offline mode
    HAVE_LANGCHAIN = False


class _ShimTool:
    """Minimal stand-in for a LangChain tool: `.invoke(dict)`, `.name`, `.description`."""

    def __init__(self, fn: Callable[..., Any]):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self.name = fn.__name__
        self.description = (fn.__doc__ or "").strip()
        self._params = list(inspect.signature(fn).parameters)

    def invoke(self, args: dict | None = None, /, **kwargs) -> Any:
        """Match LangChain's contract: a single dict of named args (or kwargs)."""
        payload = dict(args) if isinstance(args, dict) else {}
        payload.update(kwargs)
        return self._fn(**payload)

    def __call__(self, *args, **kwargs) -> Any:
        return self._fn(*args, **kwargs)


def tool(fn: Callable[..., Any]):
    """Wrap `fn` as a LangChain tool if available, else as the offline shim."""
    if HAVE_LANGCHAIN:
        return _lc_tool(fn)
    return _ShimTool(fn)
