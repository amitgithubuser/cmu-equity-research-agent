"""Regression coverage for the one-sided NVDA report captured on 2026-09-07.

The numeric values below are frozen from that local output. Qualitative transcript/filing passages
are explicitly test-only fixtures; they validate workflow balance and presentation, not market facts.
"""

from equity_research.agents.editor import editor_node
from equity_research.nodes_real import evidence_check_node
from equity_research.ui.presenter import brief_markdown


def _evidence(claim, source, period, snippet, *, kind="text", value=None, task=""):
    item = {
        "claim": claim,
        "kind": kind,
        "citation": {
            "source": source,
            "company": "NVDA",
            "period": period,
            "snippet": snippet,
        },
        "value": value,
    }
    if task:
        item["source_task"] = task
    return item


def test_required_downside_evidence_is_retried_before_thesis_generation():
    upside = _evidence(
        "Management described strong demand.", "Synthetic earnings-call fixture (test only)",
        "quarter ended 2026-07-31", "Management described strong demand.", task="up",
    )
    state = {
        "plan": [
            {"id": "up", "claim": "upside", "kind": "text", "required": True,
             "evidence_role": "management_upside"},
            {"id": "down", "claim": "downside", "kind": "text", "required": True,
             "evidence_role": "management_downside"},
        ],
        "covered": ["up", "down"],
        "evidence": [upside],
        "retries": 2,
        "max_retries": 20,
        "balance_repair_rounds": 0,
        "balance_repair_max": 1,
    }
    out = evidence_check_node(state)
    assert out["need_more"] is True
    assert out["reopen"] == ["down"]
    assert out["balance_gaps"] == ["management_downside"]


def test_nvda_report_renders_rich_balanced_cases_and_separate_sources():
    period = "quarter ended 2026-07-31"
    gross = _evidence(
        "Gross margin was 75.0% in the latest eligible quarter.",
        "latest eligible quarter from a 4-quarter statement window", period,
        "gross_margin=0.7498 from cited statement line items", kind="number", value=0.7498,
        task="gm",
    )
    operating = _evidence(
        "Operating margin was 66.2% in the latest eligible quarter.",
        "latest eligible quarter from a 4-quarter statement window", period,
        "operating_margin=0.6624 from cited statement line items", kind="number", value=0.6624,
        task="op",
    )
    fcf = _evidence(
        "Free cash flow margin was 22.2%, down 37.3 percentage points from the prior period.",
        "latest eligible quarter from a 4-quarter statement window", period,
        "fcf_margin=0.2224; prior-period fcf_margin=0.5953; change=-0.3729",
        kind="number", value=0.2224, task="fcf",
    )
    upside = _evidence(
        "Management described sustained demand as an upside driver.",
        "Synthetic NVDA earnings-call fixture (test only)", period,
        "Management described sustained demand as an upside driver.", task="up",
    )
    downside = _evidence(
        "Management identified execution and supply transitions as downside uncertainties.",
        "Synthetic NVDA earnings-call fixture (test only)", period,
        "Management identified execution and supply transitions as downside uncertainties.", task="down",
    )
    filing = _evidence(
        "The test filing passage identifies customer concentration as a material risk.",
        "Synthetic NVDA filing fixture (test only)", "FY2026",
        "The test filing passage identifies customer concentration as a material risk.", task="risk",
    )

    branches = [
        {
            "side": "bull", "survived": True, "support_score": 0.83,
            "thesis": "Broad profitability and cited demand support a constructive operating case.",
            "mechanism": "Stable gross and operating margins could preserve operating leverage.",
            "time_horizon": "Next four reported quarters",
            "catalysts": ["Sustained reported demand", "Stable reported gross margin"],
            "watch_items": ["Revenue growth", "Gross margin", "Operating margin"],
            "invalidation_conditions": ["A sustained contraction in reported margins"],
            "claims": [gross, operating, upside],
            "evidence_rationales": [
                "Gross margin measures pricing and cost performance.",
                "Operating margin indicates whether gross profit converts into operating earnings.",
                "Management commentary supplies the demand mechanism.",
            ],
        },
        {
            "side": "bear", "survived": True, "support_score": 0.72,
            "thesis": "Cash-conversion pressure and execution uncertainty create a credible downside case.",
            "mechanism": "Lower cash conversion could weaken the quality of reported profitability.",
            "time_horizon": "Next four reported quarters",
            "catalysts": ["Another reported decline in free cash flow margin"],
            "watch_items": ["Free cash flow margin", "Management execution commentary"],
            "invalidation_conditions": ["Free cash flow margin recovers while margins remain stable"],
            "claims": [fcf, downside],
            "evidence_rationales": [
                "The observed decline is the quantitative downside premise.",
                "Management uncertainty identifies how the downside could develop.",
            ],
        },
        {
            "side": "risk", "survived": True, "support_score": 0.68,
            "thesis": "Customer concentration remains a material monitoring risk.",
            "mechanism": "Concentration can amplify the impact of changes in a major customer's demand.",
            "time_horizon": "Ongoing",
            "watch_items": ["Customer concentration disclosures"],
            "invalidation_conditions": ["Disclosed customer concentration declines materially"],
            "claims": [filing],
            "evidence_rationales": ["The filing fixture directly identifies the risk."],
        },
    ]
    state = {
        "ticker": "NVDA",
        "analysis_window": {"label": "latest quarter + 3 prior quarters"},
        "confidence": 0.57,
        "scored_branches": branches,
        "insufficient": {"bull": False, "bear": False},
        "evidence": [gross, operating, fcf, upside, downside, filing],
        "metrics": {
            "gross_margin": {"name": "gross_margin", "value": 0.7498, "inputs": {},
                             "period": period, "source": gross["citation"]["source"]},
            "operating_margin": {"name": "operating_margin", "value": 0.6624, "inputs": {},
                                 "period": period, "source": operating["citation"]["source"]},
            "fcf_margin": {"name": "fcf_margin", "value": 0.2224, "inputs": {},
                           "period": period, "source": fcf["citation"]["source"]},
        },
        "qoq": {"summary": "free cash flow margin down 37.3pp vs the prior period"},
        "transcript_status": "cached",
        "transcript_periods": ["Q2-2027", "Q1-2027", "Q4-2026", "Q3-2026"],
        "filing_status": "ingested",
        "filing_forms": ["10-K"],
        "filing_periods": ["FY2026"],
    }
    brief = editor_node(state)["draft_brief"]
    report = brief_markdown(brief)

    assert "#### Bull case" in report and "#### Bear case" in report
    assert "**How it works:**" in report and "How to evaluate this thesis" in report
    assert "What investors should watch" in report and "What would challenge this thesis" in report
    assert "**Monitor:**" not in report and "**Would weaken:**" not in report
    assert "Evidence score" in report and "not a probability or price target" in report
    assert "##### Margins & profitability" in report
    assert "##### Cash generation & balance sheet" in report
    assert "Sources and evidence" in report and "[S1]" in report
    assert "<sub>" not in report
