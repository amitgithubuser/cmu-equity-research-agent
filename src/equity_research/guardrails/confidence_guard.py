"""Stage 2 — during-generation `confidence_guard` (architecture §9.2, Lab 6.1
`check_confidence_calibration`).

Reuses the overconfidence-marker scan: count sentences containing absolute/overconfident phrasing and
score = flagged / total. Verdict: PASS < warn ≤ WARN < fail ≤ FAIL.

Crucially, this does NOT set the confidence NUMBER — that comes from the bull/bear support gap
(§8.4, D-6). This scan is a secondary check that catches overconfident *phrasing* in the prose and
suggests calmer wording.
"""

from __future__ import annotations

import re

from ..config import get_settings

# marker -> suggested calmer rephrasing (equity-research bank, adapted from Lab 6.1's medical bank)
OVERCONFIDENCE_MARKERS = {
    "proven": "supported by evidence",
    "guaranteed": "likely",
    "guarantee": "expect",
    "always": "in most cases",
    "never": "rarely",
    "certainly": "probably",
    "definitely": "likely",
    "undoubtedly": "probably",
    "no risk": "low risk",
    "risk-free": "lower-risk",
    "riskless": "lower-risk",
    "100%": "high probability",
    "surefire": "promising",
    "can't lose": "has downside risk too",
    "will skyrocket": "may rise",
    "will crash": "may decline",
}

_MARKER_RE = re.compile("|".join(re.escape(m) for m in OVERCONFIDENCE_MARKERS), re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def scan_overconfidence(text: str) -> dict:
    """Return {score, verdict, flagged, suggestions}. score = flagged sentences / total sentences."""
    s = get_settings()
    sentences = [x for x in _SENT_SPLIT.split(text or "") if x.strip()]
    if not sentences:
        return {"score": 0.0, "verdict": "PASS", "flagged": [], "suggestions": {}}

    flagged, suggestions = [], {}
    for sent in sentences:
        found = _MARKER_RE.findall(sent)
        if found:
            flagged.append(sent.strip())
            for m in found:
                key = m.lower()
                if key in OVERCONFIDENCE_MARKERS:
                    suggestions[key] = OVERCONFIDENCE_MARKERS[key]

    score = len(flagged) / len(sentences)
    verdict = "PASS" if score < s.overconfidence_warn else ("WARN" if score < s.overconfidence_fail else "FAIL")
    return {"score": round(score, 4), "verdict": verdict, "flagged": flagged, "suggestions": suggestions}


def _brief_prose(brief: dict) -> str:
    """Collect the free-text prose of a brief for scanning (claims + rationale)."""
    parts = [brief.get("executive_summary", ""), brief.get("confidence_rationale", "")]
    for group in ("bull_case", "bear_case", "key_risks"):
        for ev in brief.get(group, []) or []:
            parts.append(ev.get("claim", ""))
    for branch in brief.get("theses", []) or []:
        parts.extend([
            branch.get("thesis", ""), branch.get("mechanism", ""),
            branch.get("time_horizon", ""),
        ])
        for key in (
            "catalysts", "watch_items", "invalidation_conditions", "evidence_rationales",
        ):
            parts.extend(branch.get(key, []) or [])
    return " ".join(p for p in parts if p)


def _soften(text: str) -> str:
    """Swap each overconfident marker for its calmer phrasing (deterministic, meaning-preserving).

    Uses the SAME marker bank as the scan, so every phrase the scan would flag is rewritten in place
    ("guaranteed to rise" -> "likely to rise", "no risk" -> "low risk"). Case-insensitive match; the
    replacement is looked up by the lower-cased matched text.
    """
    return _MARKER_RE.sub(lambda m: OVERCONFIDENCE_MARKERS[m.group(0).lower()], text or "")


def _soften_brief(brief: dict) -> dict:
    """Return a copy of the brief with overconfident phrasing softened in rationale + every claim.

    Copies are shallow-per-item so the originals on the blackboard are never mutated. Only the `claim`
    text is touched — citations, values and kinds are untouched, so provenance is preserved (a softened
    claim still points at the exact same snippet, and the swap is small enough that source-support
    verification, which runs next, is unaffected).
    """
    out = dict(brief)
    out["executive_summary"] = _soften(brief.get("executive_summary", ""))
    out["confidence_rationale"] = _soften(brief.get("confidence_rationale", ""))
    for group in ("bull_case", "bear_case", "key_risks"):
        out[group] = [({**ev, "claim": _soften(ev["claim"])} if ev.get("claim") else ev)
                      for ev in (brief.get(group) or [])]
    out["theses"] = []
    for original in brief.get("theses", []) or []:
        branch = dict(original)
        for key in ("thesis", "mechanism", "time_horizon"):
            branch[key] = _soften(branch.get(key, ""))
        for key in (
            "catalysts", "watch_items", "invalidation_conditions", "evidence_rationales",
        ):
            branch[key] = [_soften(item) for item in (branch.get(key, []) or [])]
        out["theses"].append(branch)
    return out


def confidence_guard_node(state: dict) -> dict:
    """Scan the draft's prose for overconfident phrasing, then ACT on the verdict (§9.2, CP 6.1).

    The verdict is no longer inert (F-07 Problem A — it used to be computed and then flow to the next
    node unchanged). Now:

      * PASS  -> pass through untouched.
      * WARN  -> deterministically SOFTEN the phrasing (marker -> calmer rephrase) and proceed; the
                 softened brief replaces `draft_brief` so the source guard and monitor see the calm text.
      * FAIL  -> soften the shown text too, but ALSO raise `overconfident` (fail-closed). Prose that is
                 pervasively overconfident (≥ half the sentences) is escalated to a human by the monitor
                 (§10.3) instead of being shipped on a cosmetic word-swap alone — a calmer surface does
                 not fix an over-claiming thesis.
    """
    brief = state.get("draft_brief", {}) or {}
    result = scan_overconfidence(_brief_prose(brief))

    # nothing flagged -> behave exactly as before (no brief rewrite, no fail-closed flag).
    if not result["flagged"]:
        return {"overconfidence": result,
                "log": [f"confidence_guard: {result['verdict']} (score={result['score']})"]}

    softened = _soften_brief(brief)
    rescan = scan_overconfidence(_brief_prose(softened))
    fail_closed = result["verdict"] == "FAIL"
    out = {"draft_brief": softened, "overconfidence": result}
    if fail_closed:
        out["overconfident"] = True   # monitor hard-trigger -> HITL (§10.3)
    out["log"] = [(f"confidence_guard: {result['verdict']} -> softened to {rescan['verdict']} "
                  f"(score {result['score']}->{rescan['score']}) fail_closed={fail_closed}")]
    return out
