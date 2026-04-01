"""
Detect strict output / instruction constraints in a prompt.

This is intentionally lightweight and explainable: it returns which constraints
were detected and a normalized score in [0, 1] suitable for complexity scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from app.utils.math_utils import clamp01
from app.utils.text import canonicalize_prompt_for_matching


@dataclass(frozen=True)
class ConstraintSignal:
    score: float
    matched: List[str]


_FORMAT_PHRASES: List[str] = [
    "return only",
    "output only",
    "only output",
    "respond only",
    "exactly",
    "must",
    "strictly",
    "no extra",
    "do not",
    "don't",
    "without",
    "include",
    "exclude",
    "use the following format",
    "in the following format",
    "format:",
    "schema",
    "json",
    "yaml",
    "csv",
    "markdown",
    "table",
    "bullet points",
    "numbered list",
]

_REGEX_CONSTRAINTS: List[Tuple[str, re.Pattern[str]]] = [
    ("exactly_n_items", re.compile(r"\bexactly\s+\d+\b", re.IGNORECASE)),
    ("json_schema", re.compile(r"\bjson\b.*\bschema\b|\bschema\b.*\bjson\b", re.IGNORECASE)),
    ("no_explanations", re.compile(r"\b(no|without)\b.{0,30}\b(explanation|explanations|reasoning)\b", re.IGNORECASE)),
    ("word_limit", re.compile(r"\b(under|<=|less than)\s+\d+\s*(words|tokens)\b", re.IGNORECASE)),
    ("character_limit", re.compile(r"\b(under|<=|less than)\s+\d+\s*(characters|chars)\b", re.IGNORECASE)),
    ("single_line", re.compile(r"\bsingle\s+line\b|\bon one line\b", re.IGNORECASE)),
    ("valid_json_only", re.compile(r"\bvalid\s+json\b|\bjson\s+only\b", re.IGNORECASE)),
]


def detect_constraints(prompt: str) -> ConstraintSignal:
    """
    Return constraint score and matched evidence strings.

    Scoring is based on number of distinct constraint cues (phrases / regex hits)
    capped and normalized.
    """
    if not prompt.strip():
        return ConstraintSignal(score=0.0, matched=[])

    canon = canonicalize_prompt_for_matching(prompt)
    matched: List[str] = []

    low = canon.lower()
    for p in _FORMAT_PHRASES:
        if p in low:
            matched.append(p)

    for name, rx in _REGEX_CONSTRAINTS:
        if rx.search(prompt):
            matched.append(name)

    # Normalize: 0–2 hits => small, 3–6 hits => moderate, >=7 => high
    distinct = sorted(set(matched))
    score = clamp01(len(distinct) / 7.0)
    return ConstraintSignal(score=score, matched=distinct)

