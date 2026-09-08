"""Centralized settings — every model id, threshold, and cap in one place.

Architecture §14; implementation-plan §1.2. Keeping the knobs here means the ablation and
calibration work (Phase 10) tunes them in one file, and tests can override them cheaply.

Each open-question (OQ) knob is tagged with its kind:
  🔧 [default]   — engineering choice; change freely with a note.
  📊 [calibrate] — provisional; the FINAL value comes from data in Phase 10 (do not hand-tune to pass a test).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable knobs. Loaded from env / `.env`; overridable per-test via `get_settings.cache_clear()`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- models (architecture §2, §14) ---
    reasoning_model: str = "claude-opus-4-8"   # Researcher / Analyst / Critic / Editor
    routing_model: str = "claude-haiku-4-5"     # input_guard classification + cheap routing
    temperature: float = 0.1                    # low for reasoning nodes; determinism where it matters
    max_tokens: int = 4096
    llm_timeout_seconds: int = 120               # bound a single hosted-model request
    llm_max_retries: int = 2                     # retry transient transport/provider failures only

    # --- memory (§6, OQ-1) 🔧 [default] ---
    history_quarters: int = 8   # quarters of numbers + tone stored (QoQ/YoY trend)
    history_years: int = 3      # annual 10-Ks stored for structural trend
    reason_quarters: int = 4    # quarters the agent reasons over by default
    # working-memory compaction (CP 2.1 §2 — the "narrative compresses, findings never discarded" rule)
    context_char_budget: int = 24000   # 🔧 soft ceiling of scratchpad chars before compaction kicks in
    keep_recent_turns: int = 3          # 🔧 verbatim tail kept when older turns are summarized
    # long-term memory (§6.1): a durable cross-run record of prior-quarter numbers + management tone.
    # Empty string => in-memory only (tests / ephemeral runs); a path => JSON-backed, survives restarts.
    memory_dir: str = ".memory"

    # --- RAG (D-9, §6) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # open-source, local, no API key
    # embeddings backend: "auto" = real HF model if installed else deterministic hashing fallback;
    # "hashing" forces the offline deterministic embedder (the test suite sets this so it never
    # downloads a model and stays reproducible); "hf" forces the real model.
    embeddings_backend: str = "auto"
    chroma_dir: str = ".chroma"                        # persistent local store
    chunk_size: int = 1200      # OQ-2 🔧 ~one 10-K Item-level idea per chunk
    chunk_overlap: int = 150    # OQ-2 🔧 so an idea isn't cut at a boundary
    retrieve_k: int = 5         # passages kept after MMR
    retrieve_fetch_k: int = 20  # candidates before MMR
    mmr_lambda: float = 0.5     # relevance vs. diversity (Lab 3.2)
    refuse_floor: float = 0.35  # OQ-3 📊 [calibrate P10] min similarity to accept a passage
    # Transcript search is already restricted by company + document type. Hashing embeddings score
    # long conversational chunks lower than dense filing prose, so use a separate conservative floor
    # after that strict filter rather than falsely declaring an indexed transcript unavailable.
    transcript_refuse_floor: float = 0.12

    # --- tools (§7, OQ-4) 🔧 [default] ---
    vendor_divergence_rel: float = 0.05  # |calc-vendor|/|vendor| above this -> divergence_flag
    # Two-tier bar so a benign vendor gap stays advisory but a severe one forces a human (F-08):
    # the pipeline sets `divergence_flag` at `vendor_divergence_rel` (mild); the MONITOR independently
    # recomputes and only calls it a `source_conflict` (a hard HITL trigger) past this larger gap.
    source_conflict_rel: float = 0.25   # OQ-4 📊 [calibrate P10] monitor's severe calc-vs-vendor gap

    # --- ToT (§8, OQ-5/6) ---
    tot_root_branches: int = 4       # angles at the root
    tot_keep_per_side: int = 2       # OQ-5 📊 [calibrate P10] top-k kept per side per level
    # Root generation plus exactly one Analyst refinement after Critic feedback.
    tot_max_depth: int = 2
    critic_pass_score: float = 0.6   # OQ-5 📊 [calibrate P10] rubric threshold to survive
    critic_panel_margin: float = 0.1 # OQ-5 🔧 within +/- this of pass -> escalate to 3-vote panel
    analyst_critic_iters: int = 2    # OQ-6 🔧 Analyst<->Critic iterations per level
    editor_reretrieve_max: int = 1   # one bounded source-grounding repair before review
    evidence_balance_retries: int = 1
    report_repair_retries: int = 1

    # --- guardrails (§9, Lab 6.1) ---
    overconfidence_warn: float = 0.3
    overconfidence_fail: float = 0.5
    source_support_floor: float = 0.5   # 📊 per-claim support similarity to count as "supported"
    groundedness_target: float = 0.95   # CI gate (Phase 11)

    # --- HITL / monitor (§10, Lab 6.2) ---
    escalation_threshold: float = 50.0  # OQ-7 📊 [calibrate P10 via ROC/AUC] suspicion cutoff placeholder
    target_fpr: float = 0.10            # max false-positive rate on benign runs (the real decision)
    human_revision_max: int = 1         # reviewer may send one bounded pass back through planning

    # --- secrets / env ---
    anthropic_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "equity-research-analyst"

    # --- data source config ---
    # EDGAR requires a descriptive User-Agent (name + email). Set via env for real runs.
    edgar_user_agent: str = "CMU-Agentic-AI-Capstone research@example.com"
    # Public web transcript discovery/download. Requests are bounded and re-run for every analysis.
    source_timeout_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. In tests: `get_settings.cache_clear()` then re-call to pick up env overrides."""
    return Settings()
