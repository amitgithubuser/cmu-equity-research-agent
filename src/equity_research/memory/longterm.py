"""Long-term memory — persists across runs (architecture §6.1, §6.2 move 3, CP 2.1).

Two things survive between runs:
  * **filings** → the Chroma vector index (that's `rag/`), and
  * **a running record of each company's prior-quarter numbers and management tone** → *this module*.

That second store is what makes the **quarter-over-quarter comparison** possible — "margins slipped
vs. last quarter and management sounds more cautious." Without a place to remember last quarter, the
agent can only ever describe the present; with it, the brief can describe a *trend* and a *change in
tone*.

Design (mirrors `rag/store.py`'s two-backend shape so the rest of the code never branches):
  * `InMemoryLongTermStore` — a dict-backed store for tests / ephemeral runs.
  * `JSONLongTermStore`     — the same interface, persisted to one JSON file per process dir, so a
    record written on Monday's run is there on Tuesday's.
  * `get_longterm_store()`  — returns JSON-backed when `settings.memory_dir` is set, else in-memory.

The store speaks in `PeriodRecord`s (one company + one period): the headline numbers computed that
period and a management-tone score. `compare_to_prior()` is the QoQ read the Researcher uses.

Move-3 rule (§6.2): findings are written **as they are found**, so they survive the working-memory
compaction/pruning that discards the narrative. This module is the durable side of that rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ordered list of period labels good enough to sort chronologically as plain strings when they are
# ISO-ish ("2025-01-31") or fiscal ("FY2025", "Q2-2026"). We sort defensively (see `_period_sort_key`).


@dataclass
class PeriodRecord:
    """One company's snapshot for one reporting period — the unit of long-term memory."""

    company: str
    period: str                              # "FY2025", "Q2-2026", or an ISO date
    metrics: dict = field(default_factory=dict)   # {ratio_name: value} computed that period (via calc)
    tone: float | None = None                # management-tone score in [-1, 1] (transcript/news sentiment)
    note: str = ""                           # optional short human-readable context

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PeriodRecord:
        return cls(company=d["company"], period=d["period"], metrics=dict(d.get("metrics", {})),
                   tone=d.get("tone"), note=d.get("note", ""))


def _period_sort_key(period: str) -> tuple:
    """Sort periods chronologically. ISO dates sort as strings; fiscal labels sort by (year, sub).

    "FY2024" < "FY2025"; "Q1-2025" < "Q2-2025" < "Q1-2026"; ISO "2024-01-31" < "2025-01-31".
    Falls back to the raw string so unknown formats still order deterministically.
    """
    p = str(period)
    digits = "".join(ch for ch in p if ch.isdigit())
    # pull a 4-digit year if present; use the remaining leading digit(s) as an intra-year sub-order
    year = ""
    if "-" in p and p.split("-")[-1].isdigit() and len(p.split("-")[-1]) == 4:
        year = p.split("-")[-1]                       # "Q2-2026" -> "2026"
        sub = "".join(ch for ch in p.split("-")[0] if ch.isdigit()) or "0"
        return (1, int(year), int(sub), p)
    if digits[:4].isdigit() and len(digits) >= 4:
        return (1, int(digits[:4]), int(digits[4:6] or 0), p)  # "2025-01-31" / "FY2025"
    return (0, 0, 0, p)


class InMemoryLongTermStore:
    """Dict-backed store. Key = (company, period). Newer write to the same key overwrites (idempotent)."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], PeriodRecord] = {}

    def put(self, record: PeriodRecord) -> None:
        self._records[(record.company.upper(), record.period)] = record

    def get(self, company: str, period: str) -> PeriodRecord | None:
        return self._records.get((company.upper(), period))

    def history(self, company: str) -> list[PeriodRecord]:
        """All records for a company, oldest → newest."""
        recs = [r for (c, _), r in self._records.items() if c == company.upper()]
        return sorted(recs, key=lambda r: _period_sort_key(r.period))

    def prior(self, company: str, period: str) -> PeriodRecord | None:
        """The most recent record for `company` STRICTLY older than `period` (the QoQ baseline)."""
        target = _period_sort_key(period)
        older = [r for r in self.history(company) if _period_sort_key(r.period) < target]
        return older[-1] if older else None


class JSONLongTermStore(InMemoryLongTermStore):
    """Same interface, persisted to `<memory_dir>/longterm.json`. Loads on init, saves on every put."""

    def __init__(self, memory_dir: str) -> None:
        super().__init__()
        self._path = Path(memory_dir) / "longterm.json"
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return                                    # corrupt/unreadable → start empty, don't crash a run
        for d in raw.get("records", []):
            rec = PeriodRecord.from_dict(d)
            self._records[(rec.company.upper(), rec.period)] = rec

    def put(self, record: PeriodRecord) -> None:
        super().put(record)
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [r.to_dict() for r in self._records.values()]}
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def get_longterm_store(settings=None):
    """Return the durable store when `memory_dir` is configured, else an in-memory one.

    Mirrors `rag.store.get_store`: callers get a working store either way and never branch on backend.
    """
    from ..config import get_settings

    settings = settings or get_settings()
    memory_dir = getattr(settings, "memory_dir", "") or ""
    if memory_dir:
        try:
            return JSONLongTermStore(memory_dir)
        except OSError:
            pass                                      # unwritable dir → fall back to in-memory
    return InMemoryLongTermStore()


# --------------------------------------------------------------------------- QoQ comparison (the read)
def compare_to_prior(store, company: str, period: str, current_metrics: dict,
                     *, current_tone: float | None = None) -> dict:
    """Compare this period's numbers + tone against the most recent prior record (§6.1 QoQ).

    Returns a structured diff the Researcher can turn into evidence and the monitor can read:

        {"has_prior": bool, "prior_period": str|None,
         "metric_deltas": {name: {"current", "prior", "delta", "direction"}},
         "tone_delta": float|None, "tone_direction": str,
         "summary": "gross_margin down 2.1pp vs FY2024; tone more cautious"}

    A metric present now but absent in the prior record is skipped (no fabricated baseline). Deltas are
    plain differences; direction is up/down/flat so downstream code needn't re-derive it.
    """
    prior = store.prior(company, period)
    if prior is None:
        return {"has_prior": False, "prior_period": None, "metric_deltas": {},
                "tone_delta": None, "tone_direction": "unknown", "summary": "no prior period on record"}

    deltas: dict[str, dict] = {}
    for name, cur in (current_metrics or {}).items():
        base = prior.metrics.get(name)
        if base is None or not isinstance(cur, (int, float)):
            continue
        delta = round(float(cur) - float(base), 6)
        deltas[name] = {"current": float(cur), "prior": float(base), "delta": delta,
                        "direction": _direction(delta)}

    tone_delta = None
    tone_direction = "unknown"
    if current_tone is not None and prior.tone is not None:
        tone_delta = round(float(current_tone) - float(prior.tone), 6)
        tone_direction = _tone_direction(tone_delta)

    return {"has_prior": True, "prior_period": prior.period, "metric_deltas": deltas,
            "tone_delta": tone_delta, "tone_direction": tone_direction,
            "summary": _summarize(prior.period, deltas, tone_delta, tone_direction)}


def _direction(delta: float, eps: float = 1e-9) -> str:
    return "flat" if abs(delta) <= eps else ("up" if delta > 0 else "down")


def _tone_direction(delta: float, eps: float = 1e-6) -> str:
    if abs(delta) <= eps:
        return "unchanged"
    return "more optimistic" if delta > 0 else "more cautious"


def _summarize(prior_period: str, deltas: dict, tone_delta: float | None, tone_direction: str) -> str:
    bits = []
    for name, d in deltas.items():
        # margins/growth read naturally in percentage points; show pp for the 0-1 ratios
        if abs(d["delta"]) < 1:
            bits.append(f"{name} {d['direction']} {abs(d['delta']) * 100:.1f}pp vs {prior_period}")
        else:
            bits.append(f"{name} {d['direction']} {abs(d['delta']):.2f} vs {prior_period}")
    if tone_delta is not None and tone_direction not in ("unchanged", "unknown"):
        bits.append(f"tone {tone_direction}")
    return "; ".join(bits) if bits else f"no comparable metrics vs {prior_period}"
