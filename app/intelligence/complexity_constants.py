"""
Constants and tunables for the prompt complexity engine.

Kept separate so we can adjust thresholds/weights without touching scoring logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComplexityBandThresholds:
    """
    Band mapping thresholds (inclusive lower bound).

    Bands:
      [0.00, very_low_max) -> very_low
      [very_low_max, low_max) -> low
      [low_max, medium_max) -> medium
      [medium_max, high_max) -> high
      [high_max, 1.00] -> very_high
    """

    # Tuned so "single-task technical asks" land in low,
    # while composite structured prompts land in high/very_high.
    very_low_max: float = 0.08
    low_max: float = 0.35
    medium_max: float = 0.60
    high_max: float = 0.82


@dataclass(frozen=True)
class ComplexitySignalWeights:
    """
    Signal weights for the base complexity score.

    Escalation is applied *after* this base score.
    """

    length: float = 0.12
    multi_task: float = 0.14
    output_structure: float = 0.14
    multi_domain: float = 0.10
    constraints: float = 0.16
    step_by_step: float = 0.08
    reasoning: float = 0.10
    technical_structure: float = 0.12
    ambiguity: float = 0.02
    optimization_tradeoff: float = 0.02


@dataclass(frozen=True)
class EscalationConfig:
    """
    Caps and typical boost magnitudes for escalation layer.

    We cap total boost to avoid everything saturating at ~0.95.
    """

    total_boost_cap: float = 0.30
    final_score_cap: float = 0.98

    composite_technical_task_boost: float = 0.08
    math_algo_impl_boost: float = 0.14
    strong_constraints_multistep_boost: float = 0.09
    reasoning_planning_tradeoff_boost: float = 0.08
    technical_teaching_boost: float = 0.06


BAND_THRESHOLDS = ComplexityBandThresholds()
SIGNAL_WEIGHTS = ComplexitySignalWeights()
ESCALATION = EscalationConfig()

