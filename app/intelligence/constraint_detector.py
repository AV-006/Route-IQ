"""
Detect strict output / instruction constraints in a prompt.

Upgraded to focus on *instruction rigidity* instead of just surface keywords:
- explicit prohibitions (do not / without / no X)
- strict requirements (must / exactly / only)
- formatting constraints (JSON / table / bullet points)
- ordered steps (first/then/after that/finally)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.utils.math_utils import clamp01
from app.utils.text import canonicalize_prompt_for_matching


@dataclass(frozen=True)
class ConstraintSignal:
    score: float
    evidence: List[str]
    detail: Dict[str, object]


_REQUIREMENT_PHRASES: List[str] = [
    "must",
    "strictly",
    "exactly",
    "only",
    "use only",
    "return only",
    "output only",
    "respond only",
    "no extra",
    "do not skip",
    "do not",
    "don't",
    "without",
    "no recursion",
    "no assumptions",
    "justify every",
    "concise",
    "detailed",
    "deeply",
    "not superficially",
]

_FORMAT_PHRASES: List[str] = [
    "use the following format",
    "in the following format",
    "format:",
    "schema",
    "json",
    "yaml",
    "csv",
    "markdown",
    "table",
    "table format",
    "bullet points",
    "in bullet points",
    "numbered list",
]

_ORDERED_STEP_PHRASES: List[str] = [
    "first",
    "then",
    "after that",
    "next",
    "finally",
    "and then",
    "also",
]

_REGEX_CONSTRAINTS: List[Tuple[str, re.Pattern[str]]] = [
    ("exactly_n_items", re.compile(r"\bexactly\s+\d+\b", re.IGNORECASE)),
    ("json_schema", re.compile(r"\bjson\b.*\bschema\b|\bschema\b.*\bjson\b", re.IGNORECASE)),
    ("no_explanations", re.compile(r"\b(no|without)\b.{0,30}\b(explanation|explanations|reasoning)\b", re.IGNORECASE)),
    ("word_limit", re.compile(r"\b(under|<=|less than)\s+\d+\s*(words|tokens)\b", re.IGNORECASE)),
    ("character_limit", re.compile(r"\b(under|<=|less than)\s+\d+\s*(characters|chars)\b", re.IGNORECASE)),
    ("single_line", re.compile(r"\bsingle\s+line\b|\bon one line\b", re.IGNORECASE)),
    ("valid_json_only", re.compile(r"\bvalid\s+json\b|\bjson\s+only\b", re.IGNORECASE)),
    ("n_bullets", re.compile(r"\b\d+\s+bullet\s+points?\b", re.IGNORECASE)),
    ("n_sentences", re.compile(r"\b\d+\s+sentences?\b", re.IGNORECASE)),
]


def detect_constraints_rigidity(prompt: str) -> ConstraintSignal:
    """
    Return constraint rigidity score and evidence.

    This is not pure keyword counting. We combine:
    - requirement/prohibition cues (rigidity)
    - format constraints (output strictness)
    - ordered steps (structure rigidity)
    """
    if not prompt.strip():
        return ConstraintSignal(score=0.0, evidence=[], detail={})

    canon = canonicalize_prompt_for_matching(prompt)

    low = canon.lower()
    req_hits = [p for p in _REQUIREMENT_PHRASES if p in low]
    fmt_hits = [p for p in _FORMAT_PHRASES if p in low]
    step_hits = [p for p in _ORDERED_STEP_PHRASES if re.search(rf"\b{re.escape(p)}\b", low)]

    rx_hits: List[str] = []
    for name, rx in _REGEX_CONSTRAINTS:
        if rx.search(prompt):
            rx_hits.append(name)

    # Explicit prohibition density: "do not"/"without"/"no X"
    prohibitions = 0
    prohibitions += low.count("do not")
    prohibitions += low.count("don't")
    prohibitions += len(re.findall(r"\bwithout\b", low))
    prohibitions += len(re.findall(r"\bno\s+\w+", low))

    # Components saturate independently, then blend.
    req_score = clamp01(len(set(req_hits)) / 4.0)
    # Formatting constraints are a strong form of rigidity; saturate faster.
    fmt_score = clamp01(len(set(fmt_hits)) / 2.0)
    step_score = clamp01(len(set(step_hits)) / 3.0)
    rx_score = clamp01(len(set(rx_hits)) / 2.0)
    prohibition_score = clamp01(prohibitions / 3.0)

    score = clamp01(
        0.30 * req_score
        + 0.20 * prohibition_score
        + 0.28 * fmt_score
        + 0.12 * step_score
        + 0.10 * rx_score
    )

    # Hard-count constraints (e.g., "6 bullet points", "under 100 words") increase rigidity a lot.
    hard_count_markers = {
        "exactly_n_items",
        "word_limit",
        "character_limit",
        "n_bullets",
        "n_sentences",
    }
    has_hard_count = any(h in hard_count_markers for h in set(rx_hits))
    if has_hard_count:
        score = clamp01(score + 0.18)
    if has_hard_count and fmt_score >= 0.8:
        score = clamp01(score + 0.08)

    # If the prompt mixes prohibitions with strong requirements, it's typically rigid.
    if req_score >= 0.5 and prohibition_score >= 0.25:
        score = clamp01(score + 0.25)

    evidence: List[str] = []
    evidence += sorted(set(req_hits))
    evidence += sorted(set(fmt_hits))
    evidence += sorted(set(step_hits))
    evidence += sorted(set(rx_hits))
    if prohibitions:
        evidence.append(f"prohibitions≈{prohibitions}")

    detail = {
        "requirement_hits": sorted(set(req_hits)),
        "format_hits": sorted(set(fmt_hits)),
        "ordered_step_hits": sorted(set(step_hits)),
        "regex_hits": sorted(set(rx_hits)),
        "prohibition_count": prohibitions,
        "req_score": round(req_score, 4),
        "format_score": round(fmt_score, 4),
        "step_score": round(step_score, 4),
        "regex_score": round(rx_score, 4),
        "prohibition_score": round(prohibition_score, 4),
    }

    return ConstraintSignal(score=round(score, 4), evidence=evidence, detail=detail)


# Backward-compatible alias (if any older code still imports detect_constraints)
def detect_constraints(prompt: str) -> ConstraintSignal:  # pragma: no cover
    return detect_constraints_rigidity(prompt)

