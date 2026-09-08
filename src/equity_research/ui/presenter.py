"""Pure presentation helpers for the UI (Phase 12).

Kept Streamlit-free so they're unit-testable offline: the Streamlit app (`app.py`) is a thin shell
that calls these to turn the final state into display-ready structures. This mirrors the CLI's
`_format_brief` split — rendering logic is tested; the framework glue is not.
"""

from __future__ import annotations

from ..analytics import LEVEL_KEYS, RATIO_KEYS
from ..text_cleaning import clean_public_text

# Human labels for the trend keys the chart/table shows.
_TREND_LABELS = {
    "revenue": "Revenue",
    "gross_margin": "Gross margin",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "fcf_margin": "FCF margin",
    "revenue_growth": "Revenue growth (QoQ)",
    "revenue_growth_yoy": "Revenue growth (YoY)",
}

_METRIC_LABELS = {
    "gross_margin": "Gross margin",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "fcf_margin": "Free cash flow margin",
    "revenue_growth": "Revenue growth",
    "debt_to_equity": "Debt to equity",
    "current_ratio": "Current ratio",
}

_PERCENT_METRICS = {
    "gross_margin", "operating_margin", "net_margin", "fcf_margin", "revenue_growth",
}

_PROGRESS_LABELS = {
    "input_guard": "Checking the request",
    "source_setup": "Preparing filings and earnings-call transcripts",
    "planner": "Building the research plan",
    "researcher": "Choosing the next research action",
    "tools": "Gathering evidence and calculating metrics",
    "evidence_check": "Checking whether the evidence plan is complete",
    "thesis_analyst": "Building competing bull and bear explanations",
    "critic": "Testing and ranking thesis branches",
    "editor": "Writing the detailed cited report",
    "quality_gate": "Running release checks",
    "human_review": "Waiting for human review",
    "finalize": "Finalizing the report",
}


def progress_label(node_name: str, update: dict | None = None) -> str:
    """Return a safe UI status line without exposing prompts, source text, or secrets."""
    update = update or {}
    label = _PROGRESS_LABELS.get(node_name, node_name.replace("_", " ").title())
    if node_name == "source_setup":
        t_count = len(update.get("transcript_periods", []) or [])
        f_count = len(update.get("filing_periods", []) or [])
        return f"{label} · {t_count} transcript period(s), {f_count} filing period(s) ready"
    if node_name == "planner":
        return f"{label} · {len(update.get('plan', []) or [])} evidence task(s)"
    if node_name == "critic":
        return f"{label} · reasoning depth {update.get('depth', 0)}"
    if node_name == "quality_gate":
        return f"{label} · {update.get('quality_decision', 'checked')}"
    return label


def _format_metric(name: str, value) -> str:
    if not isinstance(value, (int, float)):
        return "Not available"
    if name in _PERCENT_METRICS:
        return f"{float(value):.1%}"
    if name in {"debt_to_equity", "current_ratio"}:
        return f"{float(value):.2f}x"
    return f"{float(value):,.2f}"


def _format_currency_level(value: float) -> str:
    """Format either raw-dollar levels or legacy fixtures expressed in millions."""
    amount = float(value)
    magnitude = abs(amount)
    if magnitude >= 1_000_000_000:
        return f"${amount / 1_000_000_000:,.1f}B"
    if magnitude >= 1_000_000:
        return f"${amount / 1_000_000:,.1f}M"
    if magnitude >= 1:
        return f"${amount:,.0f}M"
    return f"${amount:,.0f}"


def financial_snapshot_rows(brief: dict) -> list[dict]:
    """Return display-ready financial metrics with provenance and vendor comparison status."""
    rows: list[dict] = []
    trends = brief.get("trends", []) or []
    if trends and isinstance(trends[-1].get("revenue"), (int, float)):
        latest = trends[-1]
        rows.append({
            "Metric": "Revenue",
            "Value": _format_currency_level(latest["revenue"]),
            "Period": latest.get("period", ""),
            "Cross-check": "",
            "Citation": _first_numeric_reference(brief),
        })

    for name, metric in (brief.get("metrics", {}) or {}).items():
        if not isinstance(metric, dict):
            continue
        vendor = metric.get("vendor_value")
        rows.append({
            "Metric": _METRIC_LABELS.get(name, name.replace("_", " ").title()),
            "Value": _format_metric(name, metric.get("value")),
            "Period": metric.get("period") or brief.get("as_of", ""),
            "Cross-check": (
                f"Vendor {_format_metric(name, vendor)}"
                + ("; review difference" if metric.get("divergence_flag") else "")
                if vendor is not None else "Calculated from statement line items"
            ),
            "Citation": _metric_reference(brief, name),
        })
    return rows


def evidence_source_rows(brief: dict) -> list[dict]:
    """Return one de-duplicated row per cited evidence item for the report appendix."""
    items = list(brief.get("evidence_appendix", []) or [])
    if not items:
        for group in ("bull_case", "bear_case", "key_risks"):
            items.extend(brief.get(group, []) or [])

    rows: list[dict] = []
    seen: set[tuple] = set()
    for ev in items:
        citation = ev.get("citation", {}) or {}
        key = (
            ev.get("claim", ""), citation.get("source", ""), citation.get("period", ""),
            citation.get("snippet", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "Claim": clean_public_text(ev.get("claim", "")),
            "Source": clean_public_text(citation.get("source", "")),
            "Period": clean_public_text(citation.get("period", "")),
            "Evidence": clean_public_text(citation.get("snippet", "")),
            "URL": citation.get("url") or "",
        })
    return rows


def _evidence_key(ev: dict) -> tuple:
    citation = ev.get("citation", {}) or {}
    return (
        clean_public_text(ev.get("claim", "")),
        clean_public_text(citation.get("source", "")),
        clean_public_text(citation.get("period", "")),
        clean_public_text(citation.get("snippet", "")),
    )


def _source_numbers(brief: dict) -> dict[tuple, tuple[int, str]]:
    return {
        (row["Claim"], row["Source"], row["Period"], row["Evidence"]): (number, row["URL"])
        for number, row in enumerate(evidence_source_rows(brief), 1)
    }


def _source_marker(ev: dict, numbers: dict[tuple, tuple[int, str]]) -> str:
    match = numbers.get(_evidence_key(ev))
    if not match:
        return ""
    number, url = match
    return f"[S{number}]({url})" if url else f"[S{number}]"


def evidence_reference(brief: dict, evidence: dict) -> str:
    """Return the report's clickable reader-facing source marker for one Evidence object."""
    return _source_marker(evidence, _source_numbers(brief))


def _first_numeric_reference(brief: dict) -> str:
    for evidence in brief.get("evidence_appendix", []) or []:
        if evidence.get("kind") == "number":
            return evidence_reference(brief, evidence)
    return ""


def _metric_reference(brief: dict, metric_name: str) -> str:
    """Locate the cited calculation that produced a displayed metric."""
    readable = _METRIC_LABELS.get(metric_name, metric_name.replace("_", " ")).lower()
    aliases = {readable, metric_name.replace("_", " ").lower()}
    for evidence in brief.get("evidence_appendix", []) or []:
        if evidence.get("kind") != "number":
            continue
        citation = evidence.get("citation", {}) or {}
        text = " ".join((evidence.get("claim", ""), citation.get("snippet", ""))).lower()
        if any(alias in text for alias in aliases):
            return evidence_reference(brief, evidence)
    return ""


def _strength_label(score: float) -> str:
    if score >= 0.80:
        return "Strong"
    if score >= 0.70:
        return "Well supported"
    if score >= 0.60:
        return "Supported"
    return "Limited"


def evidence_strength_label(score: float) -> str:
    """Public display label for the Critic's numeric evidence score."""
    return _strength_label(float(score))


def management_commentary(brief: dict) -> list[dict]:
    """Select transcript-derived evidence for a dedicated management-commentary section."""
    if brief.get("management_commentary"):
        rows = []
        for ev in brief["management_commentary"]:
            citation = ev.get("citation", {}) or {}
            rows.append({
                "Claim": clean_public_text(ev.get("claim", "")),
                "Source": clean_public_text(citation.get("source", "")),
                "Period": clean_public_text(citation.get("period", "")),
                "Evidence": clean_public_text(citation.get("snippet", "")),
                "URL": citation.get("url") or "",
            })
        return rows
    terms = ("transcript", "earnings call", "conference call", "investor relations")
    return [
        row for row in evidence_source_rows(brief)
        if any(term in row["Source"].lower() for term in terms)
    ]


def trends_chart_data(brief: dict) -> dict:
    """Split a brief's `trends` into chart-ready series (Streamlit `st.line_chart`).

    Returns ``{"periods": [...], "margins": {label: [values...]}, "levels": {label: [...]}}`` — margins
    (0-1 ratios + growth) share one axis; revenue (a level) gets its own, so the two aren't crammed onto
    one scale. A metric missing in a given period contributes ``None`` (a gap in the line, not a zero).
    """
    trends = brief.get("trends", []) or []
    periods = [p.get("period", "") for p in trends]
    margins: dict[str, list] = {}
    growth: dict[str, list] = {}
    levels: dict[str, list] = {}
    for key in (item for item in RATIO_KEYS if item not in {"revenue_growth", "revenue_growth_yoy"}):
        if any(key in p for p in trends):
            margins[_TREND_LABELS.get(key, key)] = [p.get(key) for p in trends]
    for key in ("revenue_growth", "revenue_growth_yoy"):
        if any(key in p for p in trends):
            growth[_TREND_LABELS.get(key, key)] = [p.get(key) for p in trends]
    for key in LEVEL_KEYS:
        if any(key in p for p in trends):
            levels[_TREND_LABELS.get(key, key)] = [p.get(key) for p in trends]
    return {"periods": periods, "margins": margins, "growth": growth, "levels": levels}


def trend_change_rows(brief: dict) -> list[dict]:
    """Return a compact latest/QoQ/YoY comparison table for the financial section."""
    trends = brief.get("trends", []) or []
    if not trends:
        return []
    latest = trends[-1]
    prior = trends[-2] if len(trends) >= 2 else {}
    rows: list[dict] = []
    if isinstance(latest.get("revenue"), (int, float)):
        rows.append({
            "Metric": "Revenue",
            "Latest": _format_currency_level(latest["revenue"]),
            "QoQ": _format_change(latest.get("revenue_growth"), percent=True),
            "YoY": _format_change(latest.get("revenue_growth_yoy"), percent=True),
        })
    for key in ("gross_margin", "operating_margin", "net_margin", "fcf_margin"):
        value = latest.get(key)
        if not isinstance(value, (int, float)):
            continue
        qoq = value - prior[key] if isinstance(prior.get(key), (int, float)) else None
        rows.append({
            "Metric": _TREND_LABELS[key],
            "Latest": _format_metric(key, value),
            "QoQ": _format_change(qoq, percentage_points=True),
            "YoY": _format_change(latest.get(f"{key}_yoy_change"), percentage_points=True),
        })
    return rows


def _format_change(value, *, percent: bool = False, percentage_points: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    arrow = "▲" if value > 0 else ("▼" if value < 0 else "•")
    if percentage_points:
        return f"{arrow} {abs(float(value)) * 100:.1f} pp"
    if percent:
        return f"{arrow} {abs(float(value)):.1%}"
    return f"{arrow} {abs(float(value)):,.2f}"


_THEME_ORDER = (
    "Margins & profitability",
    "Growth & demand",
    "Cash generation & balance sheet",
    "Guidance & outlook",
    "Competition & positioning",
    "Supply & execution",
    "Regulatory & structural risks",
    "Other evidence",
)


def _content_theme(text: str) -> str:
    value = (text or "").lower().replace("-", " ")
    rules = (
        ("Cash generation & balance sheet", (
            "cash flow", "cash generation", "liquidity", "free cash", "debt", "leverage",
            "current ratio", "balance sheet", "buyback", "repurchase", "dividend",
        )),
        ("Competition & positioning", ("compet", "market share", "position", "platform")),
        ("Regulatory & structural risks", (
            "regulat", "tax", "litigation", "cyber", "export", "concentration", "risk factor",
        )),
        ("Supply & execution", ("supply", "inventory", "commitment", "execution", "capacity")),
        ("Margins & profitability", ("margin", "profit", "operating income", "net income")),
        ("Guidance & outlook", ("guidance", "outlook", "forecast", "expects", "expected")),
        ("Growth & demand", (
            "revenue", "growth", "demand", "pipeline", "order", "blackwell", "rubin", "network",
        )),
    )
    for theme, terms in rules:
        if any(term in value for term in terms):
            return theme
    return "Other evidence"


def grouped_case_content(branch: dict) -> list[dict]:
    """Group a branch's supporting evidence into stable, related business themes."""
    groups = {
        name: {"theme": name, "evidence": []}
        for name in _THEME_ORDER
    }
    claims = branch.get("claims", []) or []
    rationales = branch.get("evidence_rationales", []) or []
    for index, evidence in enumerate(claims):
        rationale = rationales[index] if index < len(rationales) else ""
        citation = evidence.get("citation", {}) or {}
        theme = _content_theme(
            " ".join((evidence.get("claim", ""), citation.get("snippet", ""), rationale))
        )
        groups[theme]["evidence"].append({"evidence": evidence, "rationale": rationale})
    return [groups[name] for name in _THEME_ORDER if groups[name]["evidence"]]


def case_evaluation_sections(branch: dict) -> list[dict]:
    """Turn terse model signals into three clear, branch-level evaluation explanations."""
    specs = (
        ("catalysts", "What could strengthen this thesis", _strengthen_sentence),
        ("watch_items", "What investors should watch", _watch_sentence),
        ("invalidation_conditions", "What would challenge this thesis", _challenge_sentence),
    )
    sections = []
    for key, heading, formatter in specs:
        items = list(dict.fromkeys(clean_public_text(item) for item in branch.get(key, []) or []))
        if items:
            sections.append({"heading": heading, "sentences": [formatter(item) for item in items]})
    return sections


def _lower_lead(text: str) -> str:
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[:1].lower() + text[1:]


def _strengthen_sentence(item: str) -> str:
    return f"A development such as {_lower_lead(item).rstrip('.')} would strengthen this thesis."


def _watch_sentence(item: str) -> str:
    return (
        f"Investors should monitor {_lower_lead(item).rstrip('.')} because a material change could "
        "alter the thesis."
    )


def _challenge_sentence(item: str) -> str:
    return f"A development such as {_lower_lead(item).rstrip('.')} would weaken this thesis."


def trends_table_markdown(brief: dict) -> str:
    """Render the trend series as a Markdown table (for the CLI / non-graphical view)."""
    trends = brief.get("trends", []) or []
    if not trends:
        return ""
    # union of keys present, in a stable display order
    ordered = ["revenue", *[k for k in RATIO_KEYS]]
    cols = [k for k in ordered if any(k in p for p in trends)]
    header = "| Period | " + " | ".join(_TREND_LABELS.get(k, k) for k in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [header, sep]
    for p in trends:
        cells = []
        for k in cols:
            v = p.get(k)
            if v is None:
                cells.append("—")
            elif k in ("revenue",):
                cells.append(f"{v:,.0f}")
            else:
                cells.append(f"{v:.1%}")   # margins/growth as percentages
        rows.append(f"| {p.get('period', '')} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def brief_markdown(
    brief: dict,
    *,
    include_header: bool = True,
    include_trends: bool = True,
    include_sources: bool = True,
    part: str = "all",
    include_financial_text: bool = True,
) -> str:
    """Render the report narrative as Markdown.

    Charts and tables stay separate in the Streamlit shell, while CLI callers can request the full
    text report. Every factual bullet still comes directly from an Evidence object with a citation.
    """
    if not brief:
        return "_No brief produced._"
    if brief.get("withheld"):
        return (f"### ⛔ Brief withheld — {brief.get('ticker', '')}\n\n"
                f"**Reason:** {brief.get('reason', 'human review: blocked')}\n\n"
                f"> {brief.get('disclaimer', '')}")

    if part not in {"all", "overview", "closing"}:
        raise ValueError("part must be 'all', 'overview', or 'closing'")

    out: list[str] = []
    source_numbers = _source_numbers(brief)
    if part in {"all", "overview"}:
        if include_header:
            out.append(
                f"## {brief.get('ticker', '')} — Equity Research Report  "
                f"\n*Analysis window: {brief.get('as_of', 'latest')}*"
            )

        evidence_count = len(evidence_source_rows(brief))
        thesis_count = len(brief.get("theses", []) or [])
        out.append("\n### Executive summary")
        if brief.get("executive_summary"):
            out.append(clean_public_text(brief["executive_summary"]))
        out.append(
            f"**Confidence:** {brief.get('confidence', 0.0):.2f}  "
            f"\n**Evidence base:** {evidence_count} cited findings  "
            f"\n**Surviving thesis branches:** {thesis_count}"
        )
        if brief.get("confidence_rationale"):
            out.append(f"\n_{brief['confidence_rationale']}_")
        qoq = brief.get("qoq", {}) or {}
        if qoq.get("summary"):
            out.append(f"\n**Change from the prior period:** {qoq['summary']}")

        overview = brief.get("company_overview") or {}
        if overview:
            out.append("\n### Company and business overview")
            out.append(
                f"{clean_public_text(overview.get('claim', ''))} "
                f"{_source_marker(overview, source_numbers)}".strip()
            )

        developments = brief.get("recent_developments", []) or []
        if developments:
            out.append("\n### Recent developments")
            for ev in developments:
                out.append(
                    f"- {clean_public_text(ev.get('claim', ''))} "
                    f"{_source_marker(ev, source_numbers)}".strip()
                )

        financial = brief.get("financial_analysis", []) or []
        if financial or brief.get("metrics") or brief.get("trends"):
            out.append("\n### Financial analysis")
        if financial and include_financial_text:
            numeric_sources = [
                _source_marker(ev, source_numbers)
                for ev in (brief.get("evidence_appendix", []) or [])
                if ev.get("kind") == "number"
            ]
            for index, line in enumerate(financial):
                marker = numeric_sources[index] if index < len(numeric_sources) else ""
                out.append(f"- {line} {marker}".strip())
        if part == "all" and include_trends:
            table = trends_table_markdown(brief)
            if table:
                out.append("\n#### Financial trend detail\n" + table)

    if part == "all":
        theses = brief.get("theses", []) or []
        out.append("\n### Investment cases")
        out.append(
            "> **How to read the evidence score:** Each thesis is independently graded from 0 to 1 "
            "on source support, internal consistency, materiality, and resilience. The four checks "
            "are equally weighted; 0.60 is the minimum passing score. It is not a probability or "
            "price target."
        )

        def case_section(title: str, side: str, fallback_key: str) -> None:
            branches = [branch for branch in theses if branch.get("side") == side]
            out.append(f"\n#### {title}")
            if not branches:
                fallback = brief.get(fallback_key, []) or []
                if not fallback:
                    out.append("_Insufficient cited evidence — this case was not asserted._")
                    return
                for ev in fallback:
                    out.append(f"{ev.get('claim', '')} {_source_marker(ev, source_numbers)}".strip())
                return

            for branch_number, branch in enumerate(branches, 1):
                thesis = clean_public_text(branch.get("thesis") or "Grounded thesis")
                score = float(branch.get("support_score", 0.0) or 0.0)
                angle = f"Angle {branch_number}: " if len(branches) > 1 else ""
                out.append(f"\n**{angle}Thesis — {thesis}**")
                out.append(
                    f"> **Evidence score: {score:.2f} / 1.00 — {_strength_label(score)}.** "
                    "Scores at or above 0.60 survived the independent Critic."
                )
                if branch.get("mechanism"):
                    out.append(f"**How it works:** {clean_public_text(branch['mechanism'])}")
                if branch.get("time_horizon"):
                    out.append(f"**Evaluation horizon:** {branch['time_horizon']}")
                for group in grouped_case_content(branch):
                    out.append(f"\n##### {group['theme']}")
                    for item in group["evidence"]:
                        evidence = item["evidence"]
                        out.append(
                            f"**Evidence:** {clean_public_text(evidence.get('claim', ''))} "
                            f"{_source_marker(evidence, source_numbers)}".strip()
                        )
                        if item["rationale"]:
                            out.append(
                                f"_Why it matters: {clean_public_text(item['rationale'])}_"
                            )
                evaluation = case_evaluation_sections(branch)
                if evaluation:
                    out.append("\n##### How to evaluate this thesis")
                    out.append(
                        "These conditions apply to the thesis as a whole, rather than to any one "
                        "evidence category."
                    )
                    for section in evaluation:
                        out.append(f"**{section['heading']}.** " + " ".join(section["sentences"]))

        case_section("Bull case", "bull", "bull_case")
        case_section("Bear case", "bear", "bear_case")
        case_section("Key risks", "risk", "key_risks")

    if part in {"all", "closing"}:
        commentary = management_commentary(brief)
        if commentary:
            out.append("\n### Management commentary")
            for row in commentary:
                key = (row["Claim"], row["Source"], row["Period"], row["Evidence"])
                match = source_numbers.get(key)
                marker = ""
                if match:
                    number, url = match
                    marker = f"[S{number}]({url})" if url else f"[S{number}]"
                out.append(f"- {row['Claim']} {marker}".strip())

        open_questions = brief.get("open_questions", []) or []
        if open_questions:
            out.append("\n### Open questions and evidence gaps")
            out.extend(f"- {item}" for item in open_questions)

    if part == "all" and include_sources:
        sources = evidence_source_rows(brief)
        if sources:
            out.append("\n### Sources and evidence")
            for i, row in enumerate(sources, 1):
                source = f"{row['Source']} · {row['Period']}".strip(" ·")
                cite = f"[{source}]({row['URL']})" if row["URL"] else source
                marker = f"[S{i}]({row['URL']})" if row["URL"] else f"[S{i}]"
                out.append(f"**{marker} {row['Claim']}**  \n{cite}  \n> {row['Evidence']}")

    if part in {"all", "closing"}:
        out.append(f"\n---\n> {brief.get('disclaimer', 'This is research, not financial advice.')}")
    return "\n".join(out)


def monitor_summary(state: dict) -> dict:
    """Compact monitor/HITL status for the sidebar."""
    report = state.get("suspicion_report", {}) or {}
    window = state.get("analysis_window", {}) or {}
    return {
        "analysis_window": window.get("label", "not resolved"),
        "transcript_status": state.get("transcript_status", "not checked"),
        "transcript_periods": state.get("transcript_periods", []),
        "filing_status": state.get("filing_status", "not checked"),
        "filing_forms": state.get("filing_forms", []),
        "quality_decision": state.get("quality_decision", "not checked"),
        "suspicion": state.get("suspicion", 0.0),
        "needs_review": bool(state.get("needs_review")),
        "fired": report.get("fired", []),
        "review_triggers": state.get("review_triggers", []),
        "escalation_reason": state.get("escalation_reason", ""),
        "blocked": bool(state.get("blocked")),
        "human_decision": state.get("human_decision", ""),
    }


def trajectory_lines(state: dict, limit: int = 200) -> list[str]:
    """The human-readable log trail for an expandable 'trace' panel."""
    return list(state.get("log", []) or [])[:limit]


def runtime_profile_rows(profile: dict, service_setup_seconds: float = 0.0) -> list[dict]:
    """Convert raw node timings into reader-friendly stage names and shares."""
    graph_total = float(profile.get("total_seconds", 0.0) or 0.0)
    total = graph_total + max(0.0, float(service_setup_seconds or 0.0))
    raw_rows = list(profile.get("stages", []) or [])
    if service_setup_seconds:
        raw_rows.append({"stage": "service_setup", "seconds": service_setup_seconds})
    rows = []
    for row in sorted(raw_rows, key=lambda item: float(item.get("seconds", 0.0)), reverse=True):
        name = str(row.get("stage", "unknown"))
        seconds = float(row.get("seconds", 0.0) or 0.0)
        label = (
            "Starting models and local services"
            if name == "service_setup" else _PROGRESS_LABELS.get(
                name, name.replace("_", " ").title()
            )
        )
        rows.append({
            "Stage": label,
            "Seconds": round(seconds, 1),
            "Share": f"{seconds / total:.0%}" if total > 0 else "—",
        })
    return rows
