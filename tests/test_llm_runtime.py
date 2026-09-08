"""Regression tests for the real-client construction boundary (no network calls)."""

import sys
from types import SimpleNamespace

from equity_research import llm as llm_module


class _Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _settings():
    return SimpleNamespace(
        reasoning_model="claude-opus-4-8",
        routing_model="claude-haiku-4-5",
        temperature=0.1,
        max_tokens=4096,
        llm_timeout_seconds=120,
        llm_max_retries=2,
        anthropic_api_key="test",
    )


def test_opus_48_client_omits_deprecated_temperature(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_anthropic",
                        SimpleNamespace(ChatAnthropic=_Client))
    monkeypatch.setattr(llm_module, "get_settings", _settings)
    client = llm_module.get_llm("reasoning")
    assert "temperature" not in client.kwargs
    assert client.kwargs["default_request_timeout"] == 120
    assert client.kwargs["max_retries"] == 2


def test_routing_client_keeps_configured_temperature(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_anthropic",
                        SimpleNamespace(ChatAnthropic=_Client))
    monkeypatch.setattr(llm_module, "get_settings", _settings)
    client = llm_module.get_llm("routing")
    assert client.kwargs["temperature"] == 0.1
