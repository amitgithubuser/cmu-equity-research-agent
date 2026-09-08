"""LLM boundary (implementation-plan guiding principle #4: "mockable LLM boundary").

Every agent node takes its `llm` as a parameter, so tests inject a fake and the graph runs
end-to-end without spending tokens. This module provides:

  * `get_llm(role)` — a real `ChatAnthropic` (Opus for reasoning, Haiku for routing) on a real run;
  * `FakeLLM` — a scripted stand-in with the SAME surface (`.invoke`, `.with_structured_output`,
    `.bind_tools`) for tests.

The surface we depend on is deliberately small: `.invoke(prompt) -> str | AIMessage`, and
`.with_structured_output(Model).invoke(prompt) -> Model` (Anthropic tool-use structured output).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import get_settings


def message_text(result: Any) -> str:
    """Normalize plain strings and LangChain message content to one text value."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


# --------------------------------------------------------------------------- real client
def get_llm(role: str = "reasoning", **overrides) -> Any:  # pragma: no cover - needs .[real] + key
    """Return a real ChatAnthropic bound to the role's model. `role` in {"reasoning", "routing"}."""
    from langchain_anthropic import ChatAnthropic

    s = get_settings()
    model = s.routing_model if role == "routing" else s.reasoning_model
    kwargs = {
        "model": model,
        "max_tokens": s.max_tokens,
        "api_key": s.anthropic_api_key or None,
        "default_request_timeout": s.llm_timeout_seconds,
        "max_retries": s.llm_max_retries,
    }
    # Claude Opus 4.8 rejects the legacy `temperature` parameter. Keep the setting for models that
    # accept it (including the routing model), but omit it for the incompatible Opus family so the
    # repository's default configuration works without a runtime-only model override.
    if not model.lower().startswith("claude-opus-4-8"):
        kwargs["temperature"] = s.temperature
    kwargs.update(overrides)
    return ChatAnthropic(**kwargs)


# --------------------------------------------------------------------------- fake client (tests)
class _StructuredRunner:
    """`.with_structured_output(Model)` result: `.invoke(prompt)` returns a Model instance."""

    def __init__(self, model_cls, handler: Callable[[str, type], Any]):
        self._model_cls = model_cls
        self._handler = handler

    def invoke(self, prompt: Any) -> Any:
        result = self._handler(str(prompt), self._model_cls)
        if isinstance(result, self._model_cls):
            return result
        return self._model_cls.model_validate(result)


class FakeLLM:
    """A scripted LLM for tests.

    Args:
        text_handler: `(prompt) -> str` for plain `.invoke` calls.
        structured_handler: `(prompt, model_cls) -> model instance | dict` for
            `.with_structured_output(model_cls).invoke(prompt)` calls.
    """

    def __init__(self, text_handler: Callable[[str], str] | None = None,
                 structured_handler: Callable[[str, type], Any] | None = None):
        self._text = text_handler or (lambda p: "")
        self._structured = structured_handler or (lambda p, m: m())
        self.calls: list[str] = []

    def invoke(self, prompt: Any) -> Any:
        self.calls.append(str(prompt))
        return self._text(str(prompt))

    def with_structured_output(self, model_cls) -> _StructuredRunner:
        def handler(prompt: str, cls: type):
            self.calls.append(prompt)
            return self._structured(prompt, cls)
        return _StructuredRunner(model_cls, handler)

    def bind_tools(self, tools):
        return self
