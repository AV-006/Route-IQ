"""
Prompt Complexity Engine.

Adds an interpretable, demo-friendly estimate of how difficult a prompt will be
for an LLM to answer well, based on multiple explainable signals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from app.core.constants import DOMAIN_NAMES
from app.intelligence.constraint_detector import detect_constraints
from app.intelligence.task_parser import detect_multi_task
from app.models.schemas import TextFeatures
from app.utils.math_utils import clamp01, distribution_entropy, max_entropy_uniform


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


@dataclass(frozen=True)
class ComplexityResult:
    complexity_score: float
    complexity_band: str
    signals: Dict[str, ComplexitySignalResult]


_STEP_BY_STEP_PHRASES = [
    "step by step",
    "show your work",
    "show each step",
    "walk through",
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
    "edge case",
    "assumption",
    "whether",
    "what changes if",
    "under what conditions",
    "under what assumptions",
    "why does this work",
    "why does this apply",
]

_TECHNICAL_MARKERS_RX: List[Tuple[str, re.Pattern[str]]] = [
    ("code_fence", re.compile(r"```", re.MULTILINE)),
    ("inline_code", re.compile(r"`[^`]{1,40}`")),
    ("latex_math", re.compile(r"(\$[^$]+\$|\\frac\{|\\int\b|∫)")),
    ("complexity_notation", re.compile(r"\bO\([^)]+\)")),
]

_AMBIGUITY_PHRASES = [
    "if unclear",
    "if it's unclear",
    "if uncertain",
    "resolve ambiguity",
    "ambiguous",
    "clarify",
    "ask clarifying questions",
    "make reasonable assumptions",
]

_OPTIMIZATION_TRADEOFF_PHRASES = [
    "optimize",
    "performance",
    "latency",
    "throughput",
    "tradeoff",
    "trade-offs",
    "benchmark",
    "bottleneck",
    "improve",
    "tune",
]


class PromptComplexityEngine:
    """
    Computes a single overall prompt-level complexity score.

    This deliberately does NOT compute per-domain complexity. It uses the already
    computed domain distribution, prompt text features, and additional lexical
    cues to produce an interpretable score and breakdown.
    """

    def __init__(self) -> None:
        # Tunable weights; sum is not required to be 1 (we renormalize).
        self._weights: Dict[str, float] = {
            "length": 0.18,
            "multi_task": 0.18,
            "multi_domain": 0.14,
            "constraints": 0.14,
            "step_by_step": 0.08,
            "reasoning": 0.10,
            "technical_structure": 0.10,
            "ambiguity": 0.04,
            "optimization_tradeoff": 0.04,
        }

    def analyze(
        self,
        prompt: str,
        domain_scores: Mapping[str, float],
        text_features: TextFeatures,
    ) -> ComplexityResult:
        p = prompt.strip()

        signals: Dict[str, ComplexitySignalResult] = {}

        signals["length"] = self._length_signal(text_features)
        signals["multi_task"] = self._multi_task_signal(p)
        signals["multi_domain"] = self._multi_domain_signal(domain_scores)
        signals["constraints"] = self._constraints_signal(p)
        signals["step_by_step"] = self._step_by_step_signal(p)
        signals["reasoning"] = self._reasoning_signal(p, domain_scores)
        signals["technical_structure"] = self._technical_structure_signal(p, text_features)
        signals["ambiguity"] = self._ambiguity_signal(p)
        signals["optimization_tradeoff"] = self._optimization_tradeoff_signal(p)

        # Weighted average with renormalization (robust to weight edits).
        total_w = sum(max(0.0, s.weight) for s in signals.values())
        if total_w <= 0.0:
            score = 0.0
        else:
            score = sum(s.contribution for s in signals.values()) / total_w

        score = round(clamp01(score), 4)
        band = self._band(score)

        return ComplexityResult(
            complexity_score=score,
            complexity_band=band,
            signals=signals,
        )

    def _band(self, complexity_score: float) -> str:
        if complexity_score < 0.33:
            return "low"
        if complexity_score < 0.66:
            return "medium"
        return "high"

    def _length_signal(self, features: TextFeatures) -> ComplexitySignalResult:
        token_count = int(features.token_count)
        score = clamp01(token_count / 120.0)
        evidence: List[str] = [f"token_count={token_count}"]
        return ComplexitySignalResult(
            name="length",
            score=round(score, 4),
            weight=self._weights["length"],
            evidence=evidence,
            detail={"token_count": token_count, "normalizer": 120.0},
        )

    def _multi_task_signal(self, prompt: str) -> ComplexitySignalResult:
        ts = detect_multi_task(prompt)
        return ComplexitySignalResult(
            name="multi_task",
            score=round(ts.score, 4),
            weight=self._weights["multi_task"],
            evidence=list(ts.evidence),
            detail={"estimated_task_count": ts.estimated_task_count},
        )

    def _multi_domain_signal(self, domain_scores: Mapping[str, float]) -> ComplexitySignalResult:
        # Use entropy + "how many domains are meaningfully active".
        # - Single peaked => low complexity (single intent)
        # - Broad / mixed => higher complexity
        probs = {d: float(domain_scores.get(d, 0.0)) for d in DOMAIN_NAMES}
        ent = distribution_entropy(probs)
        ent_max = max_entropy_uniform(len(DOMAIN_NAMES))
        ent_norm = 0.0 if ent_max <= 0.0 else clamp01(ent / ent_max)

        active = sorted([v for v in probs.values() if v >= 0.18], reverse=True)
        active_count = len(active)
        active_score = clamp01(active_count / 3.0)  # 0,1,2,3+ active domains

        top_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top1 = top_sorted[0][1] if top_sorted else 0.0
        top2 = top_sorted[1][1] if len(top_sorted) > 1 else 0.0
        close_top2 = clamp01(1.0 - ((top1 - top2) / 0.35))  # close => higher complexity

        score = clamp01(0.55 * ent_norm + 0.30 * active_score + 0.15 * close_top2)
        evidence = [
            f"entropy_norm={round(ent_norm,4)}",
            f"active_domains>=0.18={active_count}",
            f"top1={round(top1,4)} top2={round(top2,4)}",
        ]
        return ComplexitySignalResult(
            name="multi_domain",
            score=round(score, 4),
            weight=self._weights["multi_domain"],
            evidence=evidence,
            detail={
                "entropy": round(ent, 6),
                "entropy_norm": round(ent_norm, 4),
                "active_domain_threshold": 0.18,
                "active_domain_count": active_count,
                "top1": round(top1, 4),
                "top2": round(top2, 4),
            },
        )

    def _constraints_signal(self, prompt: str) -> ComplexitySignalResult:
        cs = detect_constraints(prompt)
        return ComplexitySignalResult(
            name="constraints",
            score=round(cs.score, 4),
            weight=self._weights["constraints"],
            evidence=list(cs.matched),
            detail={"matched_count": len(cs.matched)},
        )

    def _step_by_step_signal(self, prompt: str) -> ComplexitySignalResult:
        low = prompt.lower()
        matched = [p for p in _STEP_BY_STEP_PHRASES if p in low]
        score = clamp01(len(set(matched)) / 2.0)
        return ComplexitySignalResult(
            name="step_by_step",
            score=round(score, 4),
            weight=self._weights["step_by_step"],
            evidence=sorted(set(matched)),
            detail={"matched_count": len(set(matched))},
        )

    def _reasoning_signal(self, prompt: str, domain_scores: Mapping[str, float]) -> ComplexitySignalResult:
        low = prompt.lower()
        matched = [p for p in _REASONING_PHRASES if p in low]
        phrase_score = clamp01(len(set(matched)) / 3.0)

        dom = float(domain_scores.get("reasoning", 0.0))
        # Domain score is already normalized across all domains; treat 0.25+ as strong.
        dom_score = clamp01(dom / 0.35)

        score = clamp01(max(dom_score, 0.65 * phrase_score))
        evidence = [f"reasoning_domain={round(dom,4)}"] + sorted(set(matched))
        return ComplexitySignalResult(
            name="reasoning",
            score=round(score, 4),
            weight=self._weights["reasoning"],
            evidence=evidence,
            detail={
                "reasoning_domain_score": round(dom, 4),
                "phrase_score": round(phrase_score, 4),
                "dom_score_scaled": round(dom_score, 4),
            },
        )

    def _technical_structure_signal(self, prompt: str, features: TextFeatures) -> ComplexitySignalResult:
        matched: List[str] = []
        for name, rx in _TECHNICAL_MARKERS_RX:
            if rx.search(prompt):
                matched.append(name)

        rx_score = clamp01(len(set(matched)) / 3.0)

        # Blend in existing low-level features.
        codeish = float(features.code_symbol_ratio)
        special = float(features.special_char_ratio)
        digits = float(features.digit_ratio)
        newlines = float(min(1.0, features.newline_count / 6.0))

        feature_score = clamp01(0.45 * codeish + 0.25 * special + 0.20 * digits + 0.10 * newlines)
        score = clamp01(0.55 * feature_score + 0.45 * rx_score)

        evidence = sorted(set(matched)) + [
            f"code_symbol_ratio={features.code_symbol_ratio}",
            f"special_char_ratio={features.special_char_ratio}",
            f"digit_ratio={features.digit_ratio}",
            f"newline_count={features.newline_count}",
        ]

        return ComplexitySignalResult(
            name="technical_structure",
            score=round(score, 4),
            weight=self._weights["technical_structure"],
            evidence=evidence,
            detail={
                "regex_markers": sorted(set(matched)),
                "regex_score": round(rx_score, 4),
                "feature_score": round(feature_score, 4),
            },
        )

    def _ambiguity_signal(self, prompt: str) -> ComplexitySignalResult:
        low = prompt.lower()
        matched = [p for p in _AMBIGUITY_PHRASES if p in low]
        score = clamp01(len(set(matched)) / 2.0)
        return ComplexitySignalResult(
            name="ambiguity",
            score=round(score, 4),
            weight=self._weights["ambiguity"],
            evidence=sorted(set(matched)),
            detail={"matched_count": len(set(matched))},
        )

    def _optimization_tradeoff_signal(self, prompt: str) -> ComplexitySignalResult:
        low = prompt.lower()
        matched = [p for p in _OPTIMIZATION_TRADEOFF_PHRASES if p in low]
        # Mild boost if "tradeoff"/"trade-offs" appears; common marker for harder responses.
        has_tradeoff = any("trade" in m for m in matched)
        score = clamp01((len(set(matched)) / 4.0) + (0.2 if has_tradeoff else 0.0))
        return ComplexitySignalResult(
            name="optimization_tradeoff",
            score=round(score, 4),
            weight=self._weights["optimization_tradeoff"],
            evidence=sorted(set(matched)),
            detail={"matched_count": len(set(matched)), "has_tradeoff": has_tradeoff},
        )

