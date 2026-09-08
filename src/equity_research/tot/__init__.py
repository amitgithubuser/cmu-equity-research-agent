"""Tree-of-Thought thesis engine (architecture §8, CP 4.1). ToT runs ONLY at the thesis stage."""

from .engine import confidence_from_gap, prune, score_branches
from .llm_roles import llm_batch_scorer, llm_generator, llm_scorer
from .rubric import critic_panel, evidence_gate, is_borderline, rubric_score

__all__ = [
    "confidence_from_gap",
    "critic_panel",
    "evidence_gate",
    "is_borderline",
    "llm_batch_scorer",
    "llm_generator",
    "llm_scorer",
    "prune",
    "rubric_score",
    "score_branches",
]
