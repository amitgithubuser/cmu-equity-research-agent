"""Lab 6.2 `TrajectoryLogger` — records every action the agent takes (architecture §10.1).

Each step is `{action, args, observation, scratchpad, status}`. Two views, exactly as in the lab:
  * `full_view()`   — everything, including the private `scratchpad` (for an explicit local audit).
  * `monitor_view()`— **scratchpad stripped**, so the suspicion monitor scores the agent on what it
    actually *did and said*, not on private reasoning. (Lab 6.2's evasion case hides intent in the
    scratchpad; excluding it by default is the point — the monitor must catch bad *actions/outputs*.)

LangSmith tracing (`LANGSMITH_TRACING=true`) captures nodes, tools, tokens, and latency on a live run.
Only safe run metadata is attached explicitly; this logger is the offline, in-state, deterministic record
the monitor reads (and tests assert on).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    action: str
    args: dict = field(default_factory=dict)
    observation: str = ""
    scratchpad: str = ""          # private reasoning — recorded but excluded from the monitor view
    status: str = "ok"            # "ok" | "error" | "refused"

    def public(self) -> dict:
        """The monitor-facing projection: everything EXCEPT the private scratchpad."""
        return {"action": self.action, "args": self.args,
                "observation": self.observation, "status": self.status}

    def full(self) -> dict:
        return {"action": self.action, "args": self.args, "observation": self.observation,
                "scratchpad": self.scratchpad, "status": self.status}


@dataclass
class TrajectoryLogger:
    """Append-only record of the agent's actions for one run."""

    trajectory_id: str = "run"
    steps: list[TrajectoryStep] = field(default_factory=list)

    def log(self, action: str, *, args: dict | None = None, observation: str = "",
            scratchpad: str = "", status: str = "ok") -> TrajectoryStep:
        step = TrajectoryStep(action=action, args=args or {}, observation=observation,
                              scratchpad=scratchpad, status=status)
        self.steps.append(step)
        return step

    def monitor_view(self) -> list[dict]:
        """What the SabotageMonitor sees — scratchpad excluded by default (Lab 6.2)."""
        return [s.public() for s in self.steps]

    def full_view(self) -> list[dict]:
        """The complete audit trail, scratchpad included (LangSmith / human review)."""
        return [s.full() for s in self.steps]
