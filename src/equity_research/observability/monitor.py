"""Lab 6.2 `SabotageMonitor`, re-purposed (architecture §10.2).

The lab scored how likely an agent was *exfiltrating data* from weighted behavioral signals. We keep
the exact mechanism — weighted signals summed to a `[0,100]` suspicion score, `suspicion ≥ threshold ⇒
flagged` — but swap the signals for **equity-research risk signals**:

| Signal | Fires when | Weight |
|---|---|---|
| `uncited_claim`      | a factual (text) claim in the brief has no supporting citation snippet | high |
| `stale_period`       | a headline number's period ≠ the run's `as_of` anchor                  | high |
| `injection_marker`   | evidence/observation text contains prompt-injection phrasing           | high |
| `narrow_support_gap` | bull vs. bear support gap below the margin (weak, coin-flip confidence) | med  |
| `source_conflict`    | our `calc` and the vendor SEVERELY disagree on the same metric          | med  |
| `vendor_divergence`  | our `calc` ratio diverged from the vendor's reported ratio (mild flag)  | med  |

`source_conflict` and `vendor_divergence` are scored from **`state["metrics"]`** — the documented
`schemas.Metric` shape (`name / value / inputs / vendor_value / divergence_flag`) the Researcher
populates when it cross-checks each computed ratio against `vendor_ratio` (F-08). They are NOT read
from `state["evidence"]`: an `Evidence` dict carries only a claim + citation and never a `vendor_value`,
so the old evidence-based scan could never fire on real application state. The two tiers differ by
severity: any `divergence_flag` (a mild gap past `vendor_divergence_rel`) raises `vendor_divergence`;
only a gap past the larger `source_conflict_rel` — recomputed here from the raw values so the monitor
judges independently — is called a `source_conflict` (a hard HITL trigger).

Weights are provisional (📊 calibrated in Phase 10 via the ablation/ROC work, OQ-7). They are chosen so
a **single high signal crosses the default `escalation_threshold` (50)** — i.e. one uncited claim or one
injection marker is, on its own, enough to demand a human. The scratchpad is excluded from what the
monitor reads (it scores the `TrajectoryLogger.monitor_view`), so hiding intent in private reasoning
does not lower the score — the monitor judges actions and outputs.
"""

from __future__ import annotations

import re

from ..config import get_settings
from ..tools.market import divergence

# --- signal weights (📊 [calibrate P10]) -----------------------------------------------------------
HIGH, MED = 50.0, 25.0
WEIGHTS: dict[str, float] = {
    "uncited_claim": HIGH,
    "stale_period": HIGH,
    "injection_marker": HIGH,
    "narrow_support_gap": MED,
    "source_conflict": MED,
    "vendor_divergence": MED,
}

# Prompt-injection phrasing sometimes smuggled inside filings/news blobs (Lab 6.2 adversarial subset).
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (the |all )?(previous|prior|above)",
    r"you are now",
    r"system prompt",
    r"override your (instructions|rules|guardrails)",
    r"do not (cite|mention|reveal)",
    r"reveal your (system|prompt|instructions)",
    r"act as (if|though)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _claims(brief: dict) -> list[dict]:
    out: list[dict] = []
    for group in (
        "bull_case", "bear_case", "key_risks", "recent_developments", "management_commentary"
    ):
        out.extend(brief.get(group, []) or [])
    if brief.get("company_overview"):
        out.append(brief["company_overview"])
    return out


def detect_injection(texts: list[str]) -> bool:
    """True if any text carries prompt-injection phrasing."""
    return any(_INJECTION_RE.search(t or "") for t in texts)


def redact_injection(text: str) -> tuple[str, bool]:
    """Defang untrusted text: replace prompt-injection phrasing with a marker. Returns (clean, found).

    Used as the PREVENTIVE input boundary (F-08b): the Researcher scans each retrieved passage through
    this before it reaches the model, so an "ignore previous instructions…" smuggled into a filing/news
    blob is neutralized rather than executed. The post-generation monitor scanning citation snippets is
    the second layer, so a phrase that slips the redactor still escalates after the fact.
    """
    if not text:
        return text or "", False
    found = bool(_INJECTION_RE.search(text))
    clean = _INJECTION_RE.sub("[redacted: possible-injection]", text) if found else text
    return clean, found


def score_suspicion(
    brief: dict,
    *,
    as_of: str = "",
    scored_branches: list[dict] | None = None,
    evidence: list[dict] | None = None,
    metrics: dict | None = None,
    monitor_view: list[dict] | None = None,
    gap_margin: float | None = None,
    conflict_rel: float | None = None,
) -> dict:
    """Pure signal scorer → {suspicion (0-100), fired: [signal...], detail: {...}}.

    `metrics` is the `state["metrics"]` blackboard channel: `{metric_name: schemas.Metric-shaped dict}`.
    The `source_conflict` / `vendor_divergence` signals are read from it (see module docstring), NOT
    from `evidence`.
    """
    s = get_settings()
    gap_margin = s.critic_panel_margin if gap_margin is None else gap_margin
    conflict_rel = s.source_conflict_rel if conflict_rel is None else conflict_rel
    evidence = evidence or []
    metrics = metrics or {}
    scored_branches = scored_branches or []
    claims = _claims(brief)

    fired: list[str] = []
    detail: dict = {}

    # --- high signals ---
    uncited = [c for c in claims
               if c.get("kind") != "number" and not (c.get("citation") or {}).get("snippet")]
    if uncited:
        fired.append("uncited_claim")
        detail["uncited_claim"] = [c.get("claim", "") for c in uncited]

    # stale: a headline NUMBER attributed to a period other than the run's anchor (presented as current)
    if as_of:
        from ..periods import periods_match

        def same_period(cited: str) -> bool:
            raw = str(cited or "")
            if raw.lower().startswith("ttm ending "):
                raw = raw[len("TTM ending "):]
            return periods_match(raw, as_of)

        stale = [c for c in claims
                 if c.get("kind") == "number"
                 and (c.get("citation") or {}).get("period")
                 and not same_period((c.get("citation") or {}).get("period"))]
        if stale:
            fired.append("stale_period")
            detail["stale_period"] = [(c.get("claim", ""), (c.get("citation") or {}).get("period"))
                                      for c in stale]

    # injection phrasing anywhere the agent read/acted (evidence snippets + public observations)
    texts = [(e.get("citation") or {}).get("snippet", "") for e in evidence]
    texts += [str(step.get("observation", "")) for step in (monitor_view or [])]
    texts += [c.get("claim", "") for c in claims]
    if detect_injection(texts):
        fired.append("injection_marker")
        detail["injection_marker"] = True

    # --- medium signals ---
    def strongest(side: str) -> float | None:
        values = [
            float(branch.get("support_score", 0.0) or 0.0)
            for branch in scored_branches if branch.get("side") == side
        ]
        return max(values) if values else None

    # Compare the strongest surviving bull and bear—the same decision-level comparison used by
    # confidence—not whichever branch happened to appear last in the list.
    bull, bear = strongest("bull"), strongest("bear")
    if bull is not None and bear is not None and abs(bull - bear) < gap_margin:
        fired.append("narrow_support_gap")
        detail["narrow_support_gap"] = {"bull": bull, "bear": bear, "gap": round(abs(bull - bear), 4)}

    # vendor divergence / source conflict — scored from state["metrics"] (schemas.Metric shape), where
    # the Researcher records each computed ratio and its vendor cross-check (F-08). `divergence_flag`
    # was set upstream at the mild `vendor_divergence_rel` bar; here we ALSO recompute the raw gap so
    # the monitor decides severity independently (defense-in-depth): only a gap past the larger
    # `source_conflict_rel` escalates to a hard-trigger `source_conflict`.
    diverged: list[str] = []
    conflicts: list[str] = []
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        vendor = m.get("vendor_value")
        if m.get("divergence_flag") or (vendor is not None):
            if m.get("divergence_flag"):
                diverged.append(name)
            value = m.get("value")
            if vendor is not None and value is not None:
                rel, _ = divergence(float(value), float(vendor), rel_threshold=conflict_rel)
                if rel > conflict_rel:
                    conflicts.append(name)
    if diverged:
        fired.append("vendor_divergence")
        detail["vendor_divergence"] = diverged
    if conflicts:
        fired.append("source_conflict")
        detail["source_conflict"] = conflicts

    suspicion = min(100.0, sum(WEIGHTS[sig] for sig in fired))
    return {"suspicion": round(suspicion, 2), "fired": fired, "detail": detail}


# HITL triggers that force a human REGARDLESS of the numeric score (architecture §10.3, CP 6.1).
_HARD_TRIGGERS = {
    "advice_query", "injection", "source_conflict", "ungrounded", "overconfident",
    "required_evidence_missing",
}


def monitor_node(state: dict) -> dict:
    """Score the final draft; set `needs_review` from the suspicion score OR a hard HITL trigger."""
    s = get_settings()
    brief = state.get("draft_brief", {}) or {}
    report = score_suspicion(
        brief,
        as_of=(state.get("analysis_window") or {}).get("cutoff") or state.get("as_of", ""),
        scored_branches=state.get("scored_branches"),
        evidence=state.get("evidence"),
        metrics=state.get("metrics"),
        monitor_view=state.get("trajectory"),
    )
    suspicion = report["suspicion"]

    # hard HITL triggers (§10.3): fire even if the weighted score is under threshold
    triggers: list[str] = []
    if state.get("escalation_reason") == "advice_query" or state.get("advice_rerouted"):
        triggers.append("advice_query")
    if "injection_marker" in report["fired"]:
        triggers.append("injection")
    if "source_conflict" in report["fired"]:
        triggers.append("source_conflict")
    # a claim still ungrounded after the Editor exhausted its re-retrieval budget
    if state.get("unsupported") and state.get("editor_rounds", 0) >= s.editor_reretrieve_max:
        triggers.append("ungrounded")
    # prose that was pervasively overconfident (confidence_guard FAIL) — a cosmetic word-swap does not
    # fix an over-claiming thesis, so a human sees it before it ships (F-07 fail-closed default)
    if state.get("overconfident"):
        triggers.append("overconfident")
    if state.get("confidence", 1.0) < 0.35:   # low confidence / near-tie → human (§10.3)
        triggers.append("low_confidence")
    window = state.get("analysis_window", {}) or {}
    required_periods = max(1, int(window.get("n_quarters", 1) or 1))
    transcript_complete = (
        state.get("transcript_status") in {"ingested", "cached"}
        and len(state.get("transcript_periods", []) or []) >= required_periods
    )
    if not transcript_complete:
        triggers.append("transcript_missing")
    if state.get("balance_gaps"):
        triggers.append("required_evidence_missing")

    needs_review = suspicion >= s.escalation_threshold or bool(triggers)
    reason = "; ".join(report["fired"] + triggers) if needs_review else ""

    return {"suspicion": suspicion, "needs_review": needs_review, "review_triggers": triggers,
            "suspicion_report": report, "escalation_reason": reason or state.get("escalation_reason", ""),
            "log": [(f"monitor: suspicion={suspicion} needs_review={needs_review} "
                     f"fired={report['fired']} triggers={triggers}")]}
