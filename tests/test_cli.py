"""Phase 11/12 gate: the CLI imports offline, formats a brief, and drives a full run when real
services are injected (monkeypatched to fakes here — no tokens, no network).
"""

from equity_research import cli
from equity_research.llm import FakeLLM
from equity_research.rag import build_test_store
from equity_research.rag.retrieve import retrieve
from equity_research.schemas import EvidencePlan, EvidenceTask
from equity_research.tools import calc


class _FakeTool:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, args):
        return self._fn(**args)


def _services():
    def market_lookup(ticker, period="annual"):
        return {"ticker": ticker, "period": period, "as_of": "FY2025",
                "line_items": {"revenue": 100.0, "cogs": 40.0}}

    plan = EvidencePlan(tasks=[
        EvidenceTask(claim="gross margin", kind="number", metric="gross_margin"),
        EvidenceTask(claim="data center revenue growth", kind="text"),
    ], stopping_note="cited")
    llm = FakeLLM(text_handler=lambda p: "Data center revenue grew.",
                  structured_handler=lambda p, m: plan if m is EvidencePlan else m())
    store = build_test_store([
        {"text": "Data center revenue grew rapidly as GPU demand increased across cloud customers.",
         "metadata": {"company": "NVDA", "period": "FY2025", "section": "Item 7",
                      "source": "10-K FY2025 Item 7", "url": "http://x"}},
    ])
    return {"reasoning_llm": llm, "routing_llm": llm,
            "tools": {"market_lookup": _FakeTool(market_lookup), "calc": calc},
            "retrieve_fn": lambda q, company, period=None: retrieve(q, company, period, store=store),
            "embeddings": None}


def test_format_brief_renders_sections():
    brief = {"ticker": "NVDA", "as_of": "FY2025",
             "bull_case": [{"claim": "growth", "citation": {"source": "10-K", "company": "NVDA", "period": "FY2025"}}],
             "bear_case": [], "key_risks": [], "confidence": 0.6,
             "confidence_rationale": "gap", "disclaimer": "This is research, not financial advice."}
    text = cli._format_brief(brief)
    assert "Bull case" in text and "not financial advice" in text


def test_format_withheld_brief():
    brief = {"ticker": "NVDA", "withheld": True, "reason": "blocked",
             "disclaimer": "This is research, not financial advice."}
    assert "withheld" in cli._format_brief(brief).lower()


def test_cli_main_runs_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_build_real_services", lambda: _services())
    rc = cli.main([
        "NVDA", "-q", "How are margins?", "--as-of", "FY2025", "--no-hitl",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NVDA — Equity Research Report" in out
    assert "[performance] total=" in out and "Building the research plan" in out


def test_cli_main_rejects_bad_ticker(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_build_real_services", lambda: _services())
    rc = cli.main(["notaticker!!", "--no-hitl"])
    assert rc == 1
