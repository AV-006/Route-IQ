"""
Prompt Complexity Engine.

Adds an interpretable, demo-friendly estimate of how difficult a prompt will be
for an LLM to answer well, based on multiple explainable signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from app.intelligence.complexity_constants import BAND_THRESHOLDS, SIGNAL_WEIGHTS
from app.intelligence.complexity_escalation import apply_escalation
from app.intelligence.complexity_signals import (
    ComplexitySignalResult,
    ambiguity_signal,
    constraints_signal,
    length_signal,
    multi_domain_signal,
    multi_task_signal,
    optimization_tradeoff_signal,
    output_structure_signal,
    reasoning_signal,
    step_by_step_signal,
    technical_structure_signal,
)
from app.models.schemas import TextFeatures
from app.utils.math_utils import clamp01


@dataclass(frozen=True)
class ComplexityResult:
    complexity_score: float
    complexity_band: str
    signals: Dict[str, ComplexitySignalResult]
    escalation: Dict[str, object]


class PromptComplexityEngine:
    """
    Computes a single overall prompt-level complexity score.

    This deliberately does NOT compute per-domain complexity. It uses the already
    computed domain distribution, prompt text features, and additional lexical
    cues to produce an interpretable score and breakdown.
    """

    def __init__(self) -> None:
        # Weights live in complexity_constants.py; this class is intentionally light.
        self._weights = SIGNAL_WEIGHTS

    def analyze(
        self,
        prompt: str,
        domain_scores: Mapping[str, float],
        text_features: TextFeatures,
    ) -> ComplexityResult:
        p = prompt.strip()

        signals: Dict[str, ComplexitySignalResult] = {}

        signals["length"] = length_signal(text_features, self._weights.length)
        signals["multi_task"] = multi_task_signal(p, self._weights.multi_task)
        signals["output_structure"] = output_structure_signal(p, self._weights.output_structure)
        signals["multi_domain"] = multi_domain_signal(domain_scores, self._weights.multi_domain)
        signals["constraints"] = constraints_signal(p, self._weights.constraints)
        signals["step_by_step"] = step_by_step_signal(p, self._weights.step_by_step)
        signals["reasoning"] = reasoning_signal(p, domain_scores, self._weights.reasoning)
        signals["technical_structure"] = technical_structure_signal(p, text_features, self._weights.technical_structure)
        signals["ambiguity"] = ambiguity_signal(p, self._weights.ambiguity)
        signals["optimization_tradeoff"] = optimization_tradeoff_signal(p, self._weights.optimization_tradeoff)

        # Weighted average with renormalization (robust to weight edits).
        total_w = sum(max(0.0, s.weight) for s in signals.values())
        if total_w <= 0.0:
            base_score = 0.0
        else:
            base_score = sum(s.contribution for s in signals.values()) / total_w

        base_score = round(clamp01(base_score), 4)

        # Escalation layer (non-linear boost for hard combinations)
        signal_scores = {k: float(v.score) for k, v in signals.items()}
        esc = apply_escalation(base_score=base_score, domain_scores=domain_scores, signal_scores=signal_scores)

        score = round(clamp01(esc.final_score), 4)
        band = self._band(score)

        return ComplexityResult(
            complexity_score=score,
            complexity_band=band,
            signals=signals,
            escalation={
                "base_score": esc.base_score,
                "boosts_applied": [
                    {"rule": b.rule, "boost": round(b.boost, 4), "reason": b.reason}
                    for b in esc.boosts_applied
                ],
                "total_boost": esc.total_boost,
                "final_score": esc.final_score,
            },
        )

    def _band(self, complexity_score: float) -> str:
        if complexity_score < BAND_THRESHOLDS.very_low_max:
            return "very_low"
        if complexity_score < BAND_THRESHOLDS.low_max:
            return "low"
        if complexity_score < BAND_THRESHOLDS.medium_max:
            return "medium"
        if complexity_score < BAND_THRESHOLDS.high_max:
            return "high"
        return "very_high"

