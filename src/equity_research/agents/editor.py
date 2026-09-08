"""Editor node (architecture §4 agent #5, §9.4) — synthesize the brief.

Design decision (faithful to §9.4 "copy citations from the evidence; do not fabricate"): the Editor
assembles the brief **deterministically** from the scored branches, so every claim carries the exact
`Citation` that was gathered — the LLM is NOT allowed to mint Citation objects (that would reopen the
fabrication risk the whole system is built to prevent). The LLM's only role here is to polish the
citation-free `confidence_rationale` prose; the numbers, claims, and citations are structural.

The Editor:
  * keeps the strongest bull + strongest bear (plus key risks) from the scored branches,
  * uses the confidence the ToT engine computed from the support gap (it does NOT invent one — D-6),
  * emits a schema-valid `ResearchBrief` where EVERY claim carries a `Citation` and the disclaimer
    is always present (schema-level guardrail, §9.4).

`unsourced` is recomputed by inspecting the produced brief — a claim missing a citation flips the
back-edge to the Researcher (bounded by `editor_reretrieve_max`).
"""

from __future__ import annotations

from ..analytics import build_trends
from ..schemas import Evidence, Metric, ResearchBrief, ThesisBranch


def editor_node(state: dict, llm=None, synthesizer=None, rationale_writer=None) -> dict:
    """Produce the draft brief; recompute `unsourced` from what was actually produced.

    Args:
        synthesizer: optional `(state, llm) -> ResearchBrief` override (used in tests to force cases).
        rationale_writer: optional `(brief_context) -> str` to polish the confidence rationale prose
            on a real run; defaults to a deterministic rationale.
    """
    ticker = state.get("ticker", "")
    window = state.get("analysis_window") or {}
    # Show the resolved scope—not merely the raw CLI field—so a reader can tell that an annual answer
    # is a four-quarter roll-up and an unspecified question used the recent-quarter default.
    as_of = window.get("label") or state.get("as_of") or window.get("cutoff") or "latest"
    confidence = float(state.get("confidence", 0.3))
    scored = state.get("scored_branches", []) or []
    insufficient = state.get("insufficient", {}) or {}

    if synthesizer is not None:
        brief: ResearchBrief = synthesizer(state, llm)
    else:
        brief = _synthesize(ticker, as_of, confidence, scored, insufficient, llm, rationale_writer)

    # Preserve the detailed, already-grounded material from the blackboard. The Editor does not
    # generate new facts here: it validates and copies the surviving branches, calculated metrics,
    # cited evidence, cross-run comparison, ratings distribution, and explicit evidence gaps.
    if not brief.theses:
        brief.theses = _surviving_theses(scored)
    if not brief.metrics:
        brief.metrics = _validated_metrics(state.get("metrics"))
    if not brief.qoq:
        brief.qoq = dict(state.get("qoq") or {})
    if not brief.ratings:
        brief.ratings = dict(state.get("ratings") or {})
    if not brief.evidence_appendix:
        brief.evidence_appendix = _unique_evidence(state.get("evidence"))
    if not brief.open_questions:
        brief.open_questions = list(dict.fromkeys(state.get("open_uncertainties") or []))

    all_evidence = state.get("evidence") or []
    if brief.company_overview is None:
        brief.company_overview = _first_evidence(all_evidence, source_terms=("company profile",))
    if not brief.recent_developments:
        raw_news = state.get("recent_developments") or _evidence_matching(
            all_evidence, source_terms=("news:",)
        )
        brief.recent_developments = _validated_evidence(raw_news)
    if not brief.management_commentary:
        brief.management_commentary = _validated_evidence(_evidence_matching(
            all_evidence,
            source_terms=("transcript", "earnings call", "conference call", "investor relations"),
        ))
    if not brief.financial_analysis:
        brief.financial_analysis = _financial_analysis(brief.metrics, brief.qoq, as_of)
    if not brief.executive_summary:
        brief.executive_summary = _executive_summary(scored, confidence, insufficient)
    if not brief.source_coverage:
        brief.source_coverage = {
            "transcript_status": state.get("transcript_status", "not checked"),
            "transcript_periods": list(state.get("transcript_periods") or []),
            "transcript_chunks": int(state.get("transcript_chunks", 0) or 0),
            "filing_status": state.get("filing_status", "not checked"),
            "filing_forms": list(state.get("filing_forms") or []),
            "filing_periods": list(state.get("filing_periods") or []),
            "filing_chunks": int(state.get("filing_chunks", 0) or 0),
            "cited_findings": len(brief.evidence_appendix),
            "required_evidence_gaps": list(state.get("balance_gaps") or []),
        }

    # Attach the per-period trend series (revenue + margins) from the captured history, unless the
    # synthesizer already set it. Pure derivation via the same `calc` definitions (analytics.py) —
    # no LLM, no fabrication: a chart point exists only where real line items back it.
    if not brief.trends:
        brief.trends = build_trends(
            state.get("history"),
            as_of=window.get("cutoff"),
            display_quarters=window.get("n_quarters"),
        )

    # Recompute unsourced from the brief itself — the authoritative check (never trust by assumption).
    unsourced = _any_unsourced(brief)
    editor_rounds = state.get("editor_rounds", 0) + (1 if unsourced else 0)
    return {"draft_brief": brief.model_dump(), "confidence": brief.confidence,
            "unsourced": unsourced, "editor_rounds": editor_rounds,
            "trajectory": [{
                "action": "editor",
                "args": {"evidence_count": len(brief.evidence_appendix)},
                "observation": (
                    f"report sections prepared; bull={len(brief.bull_case)}; "
                    f"bear={len(brief.bear_case)}; risks={len(brief.key_risks)}"
                ),
                "status": "ok" if not unsourced else "refused",
            }],
            "log": [(f"editor: bull={len(brief.bull_case)} bear={len(brief.bear_case)} "
                     f"risks={len(brief.key_risks)} unsourced={unsourced}")]}


def _synthesize(ticker, as_of, confidence, scored, insufficient, llm, rationale_writer) -> ResearchBrief:
    """Deterministic assembly from scored branches. A starved side becomes an explicit
    'insufficient evidence' note rather than an invented case (CP 3.1 / §8.3).
    """
    def claims_for(side: str) -> list[Evidence]:
        out: list[Evidence] = []
        seen: set[tuple] = set()
        for b in sorted([b for b in scored if b.get("side") == side],
                        key=lambda b: b.get("support_score", 0.0), reverse=True):
            for c in b.get("claims", []):
                ev = Evidence.model_validate(c)
                key = (
                    " ".join(ev.claim.lower().split()),
                    ev.citation.source,
                    ev.citation.company,
                    ev.citation.period,
                    ev.citation.snippet,
                )
                if key not in seen:
                    seen.add(key)
                    out.append(ev)
        return out

    bull = claims_for("bull")
    bear = claims_for("bear")
    risks = claims_for("risk")

    rationale_bits = [f"confidence {confidence} set from the bull/bear support gap"]
    # `insufficient` describes the most recent Critic pass, while `scored` accumulates survivors
    # across all BFS depths. Report insufficiency from the FINAL assembled brief so the rationale can
    # never claim a side is missing while that same side is visibly present.
    if not bull:
        rationale_bits.append("bull side had insufficient cited evidence")
    if not bear:
        rationale_bits.append("bear side had insufficient cited evidence")
    rationale = "; ".join(rationale_bits)
    if rationale_writer is not None:
        rationale = rationale_writer({"ticker": ticker, "confidence": confidence,
                                      "bull": len(bull), "bear": len(bear), "base": rationale}) or rationale

    return ResearchBrief(
        ticker=ticker, as_of=as_of, bull_case=bull, bear_case=bear, key_risks=risks,
        confidence=confidence, confidence_rationale=rationale,
    )


def _any_unsourced(brief: ResearchBrief) -> bool:
    """True if any claim in the brief lacks a citation. By schema construction this should be False;
    we still check, because the back-edge (§5.1 edge 3) depends on catching it if it ever isn't.
    """
    groups = [
        brief.bull_case,
        brief.bear_case,
        brief.key_risks,
        brief.recent_developments,
        brief.management_commentary,
    ]
    if brief.company_overview is not None:
        groups.append([brief.company_overview])
    for group in groups:
        for ev in group:
            cit = ev.citation
            if not cit or not cit.source or not cit.snippet:
                return True
    return False


def _validated_evidence(raw: list[dict] | None) -> list[Evidence]:
    out: list[Evidence] = []
    for item in raw or []:
        try:
            out.append(Evidence.model_validate(item))
        except (TypeError, ValueError):
            continue
    return out


def _evidence_matching(raw: list[dict] | None, *, source_terms: tuple[str, ...]) -> list[dict]:
    out: list[dict] = []
    for item in raw or []:
        source = str((item.get("citation") or {}).get("source", "")).lower()
        if any(term in source for term in source_terms):
            out.append(item)
    return out


def _first_evidence(raw: list[dict] | None, *, source_terms: tuple[str, ...]) -> Evidence | None:
    items = _validated_evidence(_evidence_matching(raw, source_terms=source_terms))
    return items[0] if items else None


def _financial_analysis(metrics: dict[str, Metric], qoq: dict, as_of: str) -> list[str]:
    """Convert verified calculated metrics into readable report sentences without an LLM."""
    labels = {
        "gross_margin": "Gross margin",
        "operating_margin": "Operating margin",
        "net_margin": "Net margin",
        "fcf_margin": "Free cash flow margin",
        "revenue_growth": "Revenue growth",
        "debt_to_equity": "Debt to equity",
        "current_ratio": "Current ratio",
    }
    percent = {"gross_margin", "operating_margin", "net_margin", "fcf_margin", "revenue_growth"}
    lines: list[str] = []
    for name, metric in metrics.items():
        value = metric.value
        shown = f"{value:.1%}" if name in percent else f"{value:.2f}x"
        metric_period = metric.period or as_of
        line = f"{labels.get(name, name.replace('_', ' ').title())} is {shown} for {metric_period}."
        if metric.vendor_value is not None:
            vendor = f"{metric.vendor_value:.1%}" if name in percent else f"{metric.vendor_value:.2f}x"
            line += f" The external cross-check reports {vendor}."
            if metric.divergence_flag:
                line += " The difference requires review."
        lines.append(line)
    if qoq.get("summary"):
        lines.append(str(qoq["summary"]))
    return lines


def _executive_summary(scored: list[dict], confidence: float, insufficient: dict) -> str:
    """Summarize the accepted thesis structure without introducing a new factual claim."""
    def strongest(side: str) -> str:
        candidates = [item for item in scored if item.get("side") == side]
        if not candidates:
            return "insufficient cited evidence"
        top = max(candidates, key=lambda item: item.get("support_score", 0.0))
        return str(top.get("thesis") or "a grounded evidence branch").rstrip(".!? ")

    return (
        f"The strongest supported bull interpretation is {strongest('bull')}. "
        f"The strongest supported bear interpretation is {strongest('bear')}. "
        f"Overall confidence is {confidence:.2f}; the report keeps evidence gaps visible."
    )


def _surviving_theses(scored: list[dict]) -> list[ThesisBranch]:
    """Return the Critic-approved thesis branches with their cited claims intact."""
    out: list[ThesisBranch] = []
    for branch in sorted(scored or [], key=lambda b: b.get("support_score", 0.0), reverse=True):
        if not branch.get("survived", True):
            continue
        try:
            out.append(ThesisBranch.model_validate(branch))
        except (TypeError, ValueError):
            continue
    return out


def _validated_metrics(raw: dict | None) -> dict[str, Metric]:
    """Keep only schema-valid calculated metrics; malformed tool output stays out of the report."""
    out: dict[str, Metric] = {}
    for name, value in (raw or {}).items():
        try:
            out[str(name)] = Metric.model_validate(value)
        except (TypeError, ValueError):
            continue
    return out


def _unique_evidence(raw: list[dict] | None) -> list[Evidence]:
    """Build a de-duplicated source appendix from cited evidence gathered during research."""
    out: list[Evidence] = []
    seen: set[tuple] = set()
    for item in raw or []:
        try:
            ev = Evidence.model_validate(item)
        except (TypeError, ValueError):
            continue
        key = (
            " ".join(ev.claim.lower().split()), ev.citation.source, ev.citation.period,
            ev.citation.snippet,
        )
        if key not in seen:
            seen.add(key)
            out.append(ev)
    return out
