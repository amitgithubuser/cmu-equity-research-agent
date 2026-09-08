"""Deterministic cleanup for source and model text before retrieval or display."""

from __future__ import annotations

import html
import re

_FINANCE_BOUNDARY = re.compile(
    r"(?i)(?<=[a-z0-9])(?=(?:trillion|billion|million|toward|opportunity|revenue|margin|"
    r"growth|guidance|demand|supply|risk|year|quarter)\b)"
)


def clean_public_text(value: str | None) -> str:
    """Decode entities, remove stray Markdown markers, and repair joined finance words.

    SEC inline-XBRL HTML and PDF extraction occasionally leave fragments such as
    ``*&#x35;00*billionBlackwellandRubinopportunity``. This function only fixes decoding and spacing;
    it never asks a model to rewrite source meaning.
    """
    text = str(value or "")
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ").replace("\u200b", " ").replace("Â±", "+/-")
    text = text.replace("*", " ")
    protected = {"QoQ": "__QOQ__", "YoY": "__YOY__"}
    for label, placeholder in protected.items():
        text = text.replace(label, placeholder)
    text = re.sub(r"(?<=[a-z])and(?=[A-Z])", " and ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    for label, placeholder in protected.items():
        text = text.replace(placeholder, label)
    text = _FINANCE_BOUNDARY.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
