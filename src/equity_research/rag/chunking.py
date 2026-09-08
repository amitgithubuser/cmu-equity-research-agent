"""Section-aware chunking (architecture §6.4, CP 3.1 §4).

The design choice CP 3.1 stresses: split along the document's OWN structure first — the numbered
Items of a 10-K, speaker turns in a transcript — **then** size-limit within a section, with a small
overlap so an idea isn't cut at a boundary. Splitting on structure before size is what keeps a chunk
to roughly one self-contained idea (and lets us tag each chunk with the section it came from).

Pure Python — no LLM, no embeddings — so it's fully unit-testable.
"""

from __future__ import annotations

import re

# 10-K / 10-Q "Item 1A." style headings. Captures the item label so it can go in chunk metadata.
_ITEM_RE = re.compile(r"(?im)^\s*(item\s+\d+[a-z]?\.?[^\n]{0,80})")
# Transcript speaker turns: "Jensen Huang -- CEO" or "Operator:" at line start.
_SPEAKER_RE = re.compile(r"(?m)^\s*([A-Z][A-Za-z.\- ]{2,40}(?:--[^\n]{0,40})?:|Operator)\s")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a filing/transcript into (section_label, section_text) using its own structure.

    Tries 10-K Items first, then transcript speaker turns; falls back to one "full document" section.
    """
    matches = list(_ITEM_RE.finditer(text))
    if len(matches) >= 2:
        return _slice_on_matches(text, matches, label_group=1)

    matches = list(_SPEAKER_RE.finditer(text))
    if len(matches) >= 2:
        return _slice_on_matches(text, matches, label_group=1)

    return [("full document", text.strip())]


def _slice_on_matches(text: str, matches, label_group: int) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = " ".join(m.group(label_group).split()).rstrip(":").strip()
        body = text[start:end].strip()
        if body:
            sections.append((label, body))
    return sections


def _window(body: str, size: int, overlap: int) -> list[str]:
    """Size-limit a section into overlapping windows, preferring paragraph/sentence boundaries."""
    if len(body) <= size:
        return [body]
    chunks: list[str] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(body):
        end = min(i + size, len(body))
        # try not to cut mid-sentence: back up to the last boundary in the tail of the window
        if end < len(body):
            tail = body.rfind(". ", i + step, end)
            if tail == -1:
                tail = body.rfind("\n", i + step, end)
            if tail != -1:
                end = tail + 1
        chunks.append(body[i:end].strip())
        if end >= len(body):
            break
        i = end - overlap
    return [c for c in chunks if c]


def chunk_document(text: str, meta: dict, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[dict]:
    """Section-aware chunk a document into `[{"text", "metadata"}]`.

    Each chunk's metadata carries the base `meta` (company/period/source/url) PLUS the `section` label
    it came from and its `chunk` index — used to BOTH cite and filter (§6.4).
    """
    out: list[dict] = []
    for section_label, section_text in split_into_sections(text):
        for j, piece in enumerate(_window(section_text, chunk_size, chunk_overlap)):
            out.append({
                "text": piece,
                "metadata": {**meta, "section": section_label, "chunk": len(out), "section_part": j},
            })
    return out
