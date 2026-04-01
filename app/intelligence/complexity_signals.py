"""
Complexity signals for prompt-routing middleware.

Signals are designed to be:
- explainable (evidence + details)
- robust (not overly dependent on formatting)
- composable (weighted base score + escalation layer)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from app.core.constants import DOMAIN_NAMES
from app.intelligence.constraint_detector import detect_constraints_rigidity
from app.intelligence.task_parser import detect_multi_task
from app.models.schemas import TextFeatures
from app.utils.math_utils import clamp01, distribution_entropy, max_entropy_uniform
from app.utils.text import canonicalize_prompt_for_matching, word_boundary_contains


@dataclass(frozen=True)
class ComplexitySignalResult:
    name: str
    score: float  # [0,1]
    weight: float  # >=0
    evidence: List[str]
    detail: Dict[str, object]

    @property
    def contribution(self) -> float:
        return float(self.score) * float(self.weight)


_STEP_BY_STEP_PHRASES = [
    "step by step",
    "show your work",
    "show each step",
    "walk through",
    "dry run",
    "derive",
    "proof",
    "prove",
]

_REASONING_PHRASES = [
    "explain why",
    "justify",
    "compare",
    "tradeoff",
    "trade-offs",
    "trade off",
    "edge case",
    "edge cases",
    "assumption",
    "failure mode",
    "what changes if",
    "under what conditions",
    "under what assumptions",
    "why does this work",
    "why does this apply",
]

_AMBIGUITY_PHRASES = [
    "resolve ambiguity",
    "ambiguous",
    "clarify",
    "ask clarifying questions",
    "detect contradictions",
    "contradictions",
    "if unclear",
    "if it's unclear",
    "if uncertain",
    "make reasonable assumptions",
    "no assumptions",
]

_OPTIMIZATION_TRADEOFF_PHRASES = [
    "optimize",
    "optimization",
    "performance",
    "latency",
    "throughput",
    "tradeoff",
    "trade-offs",
    "benchmark",
    "bottleneck",
    "big o",
    "asymptotic",
]


# --- upgraded technical_structure ---

_TECHNICAL_PHRASES = [
    # algorithmic / interview-y phrasing
    "time complexity",
    "space complexity",
    "recurrence relation",
    "master theorem",
    "dry run",
    "state definition",
    "edge case",
    "test cases",
    "common mistakes",
    "asymptotic",
    "big o",
    "runtime",
    "memory usage",
    "constraints",
    "graph traversal",
    "dynamic programming",
    "memoization",
    "tabulation",
    "adjacency list",
    "binary tree",
    "linked list",
    "cycle detection",
    "detect a cycle",
    "write a function",
    "python script",
    "c++ implementation",
    "implementation",
    # engineering phrasing
    "api design",
    "system design",
    "architecture",
    "schema",
    "sql query",
    "json output",
    "debugging",
    "compile",
]

_TECHNICAL_MARKERS = [
    "c++",
    "cpp",
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "json",
    "api",
    "http",
    "grpc",
    "oauth",
    "jwt",
    "o(",
    "t(n",
    "dp",
    "recursion",
    "knapsack",
    "function",
    "array",
    "arrays",
    "stack",
    "queue",
    "hash map",
    "graph",
    "tree",
]

_TECH_STRUCT_RX: List[Tuple[str, re.Pattern[str]]] = [
    ("big_o", re.compile(r"\bO\([^)]+\)", re.IGNORECASE)),
    ("recurrence_Tn", re.compile(r"\bT\s*\(\s*n\s*\)\s*=", re.IGNORECASE)),
    ("language_tag_cpp", re.compile(r"\b(c\+\+|cpp)\b", re.IGNORECASE)),
    ("language_tag_python", re.compile(r"\bpython\b", re.IGNORECASE)),
    ("json_object", re.compile(r"\{\s*\"[^\"]+\"\s*:", re.DOTALL)),
    ("sql_select", re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL)),
]


# --- output_structure ---

_OUTPUT_FORMAT_PHRASES = [
    "in bullet points",
    "bullet points",
    "into json",
    "as json",
    "in json",
    "json",
    "in a table",
    "table format",
    "as a table",
    "markdown table",
    "numbered list",
]

_DELIVERABLE_VERBS = [
    "explain",
    "derive",
    "prove",
    "implement",
    "write code",
    "write a",
    "compare",
    "discuss",
    "summarize",
    "extract",
    "rewrite",
    "provide",
    "show",
]

_SEQUENCERS = [
    "first",
    "then",
    "after that",
    "next",
    "finally",
    "also",
    "and then",
]


def _distinct_matched_phrases(low: str, phrases: Sequence[str]) -> List[str]:
    return sorted({p for p in phrases if p in low})


def length_signal(features: TextFeatures, weight: float) -> ComplexitySignalResult:
    token_count = int(features.token_count)
    # Long prompts are harder, but short prompts still aren't "zero difficulty".
    score = clamp01(token_count / 120.0)
    return ComplexitySignalResult(
        name="length",
        score=round(score, 4),
        weight=weight,
        evidence=[f"token_count={token_count}"],
        detail={"token_count": token_count, "normalizer": 120.0},
    )


def multi_task_signal(prompt: str, weight: float) -> ComplexitySignalResult:
    ts = detect_multi_task(prompt)
    return ComplexitySignalResult(
        name="multi_task",
        score=round(ts.score, 4),
        weight=weight,
        evidence=list(ts.evidence),
        detail={"estimated_task_count": ts.estimated_task_count},
    )


def multi_domain_signal(domain_scores: Mapping[str, float], weight: float) -> ComplexitySignalResult:
    probs = {d: float(domain_scores.get(d, 0.0)) for d in DOMAIN_NAMES}
    ent = distribution_entropy(probs)
    ent_max = max_entropy_uniform(len(DOMAIN_NAMES))
    ent_norm = 0.0 if ent_max <= 0.0 else clamp01(ent / ent_max)

    active = sorted([v for v in probs.values() if v >= 0.17], reverse=True)
    active_count = len(active)
    active_score = clamp01(active_count / 3.0)

    top_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1 = top_sorted[0][1] if top_sorted else 0.0
    top2 = top_sorted[1][1] if len(top_sorted) > 1 else 0.0
    close_top2 = clamp01(1.0 - ((top1 - top2) / 0.30))

    score = clamp01(0.50 * ent_norm + 0.35 * active_score + 0.15 * close_top2)
    return ComplexitySignalResult(
        name="multi_domain",
        score=round(score, 4),
        weight=weight,
        evidence=[
            f"entropy_norm={round(ent_norm,4)}",
            f"active_domains>=0.17={active_count}",
            f"top1={round(top1,4)} top2={round(top2,4)}",
        ],
        detail={
            "entropy": round(ent, 6),
            "entropy_norm": round(ent_norm, 4),
            "active_domain_threshold": 0.17,
            "active_domain_count": active_count,
            "top1": round(top1, 4),
            "top2": round(top2, 4),
        },
    )


def constraints_signal(prompt: str, weight: float) -> ComplexitySignalResult:
    cs = detect_constraints_rigidity(prompt)
    return ComplexitySignalResult(
        name="constraints",
        score=round(cs.score, 4),
        weight=weight,
        evidence=list(cs.evidence),
        detail=dict(cs.detail),
    )


def step_by_step_signal(prompt: str, weight: float) -> ComplexitySignalResult:
    low = canonicalize_prompt_for_matching(prompt).lower()
    matched = _distinct_matched_phrases(low, _STEP_BY_STEP_PHRASES)
    score = clamp01(len(matched) / 2.0)
    return ComplexitySignalResult(
        name="step_by_step",
        score=round(score, 4),
        weight=weight,
        evidence=matched,
        detail={"matched_count": len(matched)},
    )


def reasoning_signal(prompt: str, domain_scores: Mapping[str, float], weight: float) -> ComplexitySignalResult:
    low = canonicalize_prompt_for_matching(prompt).lower()
    matched = _distinct_matched_phrases(low, _REASONING_PHRASES)
    phrase_score = clamp01(len(matched) / 3.0)

    dom = float(domain_scores.get("reasoning", 0.0))
    dom_score = clamp01(dom / 0.33)

    score = clamp01(max(dom_score, 0.70 * phrase_score))
    evidence = [f"reasoning_domain={round(dom,4)}"] + matched
    return ComplexitySignalResult(
        name="reasoning",
        score=round(score, 4),
        weight=weight,
        evidence=evidence,
        detail={
            "reasoning_domain_score": round(dom, 4),
            "phrase_score": round(phrase_score, 4),
            "dom_score_scaled": round(dom_score, 4),
        },
    )


def ambiguity_signal(prompt: str, weight: float) -> ComplexitySignalResult:
    low = canonicalize_prompt_for_matching(prompt).lower()
    matched = _distinct_matched_phrases(low, _AMBIGUITY_PHRASES)
    score = clamp01(len(matched) / 2.0)
    return ComplexitySignalResult(
        name="ambiguity",
        score=round(score, 4),
        weight=weight,
        evidence=matched,
        detail={"matched_count": len(matched)},
    )


def optimization_tradeoff_signal(prompt: str, weight: float) -> ComplexitySignalResult:
    low = canonicalize_prompt_for_matching(prompt).lower()
    matched = _distinct_matched_phrases(low, _OPTIMIZATION_TRADEOFF_PHRASES)
    has_tradeoff = any("trade" in m for m in matched)
    score = clamp01((len(matched) / 4.0) + (0.2 if has_tradeoff else 0.0))
    return ComplexitySignalResult(
        name="optimization_tradeoff",
        score=round(score, 4),
        weight=weight,
        evidence=matched,
        detail={"matched_count": len(matched), "has_tradeoff": has_tradeoff},
    )


def technical_structure_signal(prompt: str, features: TextFeatures, weight: float) -> ComplexitySignalResult:
    canon = canonicalize_prompt_for_matching(prompt)
    low = canon.lower()

    phrase_hits = [p for p in _TECHNICAL_PHRASES if p in low]
    marker_hits = [m for m in _TECHNICAL_MARKERS if word_boundary_contains(low, m)]

    rx_hits: List[str] = []
    for name, rx in _TECH_STRUCT_RX:
        if rx.search(prompt):
            rx_hits.append(name)

    # Primary: language intent (phrases + markers + regex). Support: low-level features.
    # Saturate faster: a few strong technical phrases/markers is enough.
    phrase_score = clamp01(len(set(phrase_hits)) / 4.0)
    marker_score = clamp01(len(set(marker_hits)) / 3.0)
    rx_score = clamp01(len(set(rx_hits)) / 2.0)

    codeish = float(features.code_symbol_ratio)
    special = float(features.special_char_ratio)
    digits = float(features.digit_ratio)
    newlines = float(min(1.0, features.newline_count / 8.0))
    feature_score = clamp01(0.45 * codeish + 0.20 * special + 0.25 * digits + 0.10 * newlines)

    intent_score = clamp01(0.48 * phrase_score + 0.32 * marker_score + 0.20 * rx_score)
    if len(set(phrase_hits)) >= 2 and len(set(rx_hits)) >= 1:
        intent_score = clamp01(intent_score + 0.12)
    score = clamp01(0.75 * intent_score + 0.25 * feature_score)

    evidence = sorted(set(phrase_hits))[:10] + sorted(set(rx_hits)) + sorted(set(marker_hits))[:10]
    evidence += [
        f"code_symbol_ratio={features.code_symbol_ratio}",
        f"digit_ratio={features.digit_ratio}",
        f"newline_count={features.newline_count}",
    ]

    return ComplexitySignalResult(
        name="technical_structure",
        score=round(score, 4),
        weight=weight,
        evidence=evidence,
        detail={
            "phrase_hits_count": len(set(phrase_hits)),
            "marker_hits_count": len(set(marker_hits)),
            "regex_hits": sorted(set(rx_hits)),
            "phrase_score": round(phrase_score, 4),
            "marker_score": round(marker_score, 4),
            "regex_score": round(rx_score, 4),
            "feature_score": round(feature_score, 4),
            "intent_score": round(intent_score, 4),
        },
    )


def output_structure_signal(prompt: str, weight: float) -> ComplexitySignalResult:
    canon = canonicalize_prompt_for_matching(prompt)
    low = canon.lower()

    sequencers = _distinct_matched_phrases(low, _SEQUENCERS)
    formats = _distinct_matched_phrases(low, _OUTPUT_FORMAT_PHRASES)

    # Count distinct deliverable verbs / deliverable types.
    deliverable_hits: List[str] = []
    for v in _DELIVERABLE_VERBS:
        if v in low:
            deliverable_hits.append(v)

    distinct_deliverables = sorted(set(deliverable_hits))

    # Heuristic: output structure is about *diverse deliverables* + *ordered structure* + *format constraints*.
    deliverable_score = clamp01(max(0.0, (len(distinct_deliverables) - 1) / 3.0))  # 1 verb=>0
    sequencer_score = clamp01(len(sequencers) / 3.0)
    format_score = clamp01(len(formats) / 2.0)

    # Formats and distinct deliverables often matter more than explicit "first/then".
    score = clamp01(0.55 * deliverable_score + 0.15 * sequencer_score + 0.30 * format_score)

    evidence = []
    if distinct_deliverables:
        evidence.append(f"deliverables={len(distinct_deliverables)}:{','.join(distinct_deliverables[:8])}")
    evidence += [f"sequencers={len(sequencers)}"] if sequencers else []
    evidence += formats

    return ComplexitySignalResult(
        name="output_structure",
        score=round(score, 4),
        weight=weight,
        evidence=evidence,
        detail={
            "distinct_deliverables": distinct_deliverables,
            "sequencers": sequencers,
            "formats": formats,
            "deliverable_score": round(deliverable_score, 4),
            "sequencer_score": round(sequencer_score, 4),
            "format_score": round(format_score, 4),
        },
    )

