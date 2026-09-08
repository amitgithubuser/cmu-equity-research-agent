"""Token-free fixtures for testing the final twelve-node topology.

These functions mirror the public graph checkpoints without representing tools or folded quality
checks as extra nodes. Production runs use ``nodes_real.build_real_nodes``.
"""

from __future__ import annotations

from .schemas import Citation, Evidence


def _fake_evidence(claim: str, kind: str = "text") -> dict:
    return Evidence(
        claim=claim, kind=kind, value=(1.0 if kind == "number" else None),
        citation=Citation(source="10-K FY2025", company="TEST", period="FY2025", snippet=claim),
    ).model_dump()


def input_guard(s: dict) -> dict:
    return {"proceed": True, "log": ["input_guard: ok"]}


def planner(s: dict) -> dict:
    return {"plan": [{"claim": "revenue", "kind": "number"}], "max_retries": 3,
            "retries": 0, "editor_rounds": 0, "depth": 0,
            "max_depth": s.get("max_depth", 3), "editor_reretrieve_max": 2,
            "log": ["planner: 1 task"]}


def source_setup(s: dict) -> dict:
    return {"analysis_window": {"mode": "recent_quarters", "n_quarters": 4, "cutoff": "",
                                "label": "latest quarter + 3 prior quarters",
                                "reason": "no period specified"},
            "transcript_status": "ingested", "transcript_periods": ["Q2-2026"],
            "log": ["source_setup: transcript window ready"]}


def researcher(s: dict) -> dict:
    return {"evidence": [_fake_evidence("stub finding")], "retries": 1, "need_more": False,
            "log": ["researcher: plan complete"]}


def tools(s: dict) -> dict:
    return {"log": ["tools"]}


def evidence_check(s: dict) -> dict:
    return {"need_more": False, "log": ["evidence_check"]}


def thesis_analyst(s: dict) -> dict:
    return {"branches": [{"side": "bull", "claims": []}, {"side": "bear", "claims": []}],
            "log": ["thesis_analyst"]}


def critic(s: dict) -> dict:
    depth = s.get("depth", 0) + 1
    return {"scored_branches": s.get("branches", []), "depth": depth,
            "log": [f"critic depth={depth}"]}


def editor(s: dict) -> dict:
    draft = {
        "ticker": s.get("ticker", "TEST"), "as_of": s.get("as_of", "FY2025"),
        "bull_case": [_fake_evidence("bull point")], "bear_case": [_fake_evidence("bear point")],
        "key_risks": [_fake_evidence("risk point")], "confidence": 0.6,
        "confidence_rationale": "stub", "disclaimer": "This is research, not financial advice.",
    }
    return {"draft_brief": draft, "confidence": 0.6, "unsourced": False, "log": ["editor"]}


def quality_gate(s: dict) -> dict:
    return {"suspicion": 0.0, "needs_review": False, "unsupported": False,
            "quality_decision": "pass", "log": ["quality_gate"]}


def human_review(s: dict) -> dict:
    return {"human_decision": "allow", "log": ["human_review"]}


def finalize(s: dict) -> dict:
    return {"brief": s.get("draft_brief", {}), "log": ["finalize"]}


def stub_nodes() -> dict:
    """The full `{node_name: fn}` map `build_graph` expects."""
    return {
        "input_guard": input_guard, "source_setup": source_setup,
        "planner": planner, "researcher": researcher,
        "tools": tools, "evidence_check": evidence_check,
        "thesis_analyst": thesis_analyst, "critic": critic,
        "editor": editor, "quality_gate": quality_gate,
        "human_review": human_review, "finalize": finalize,
    }
