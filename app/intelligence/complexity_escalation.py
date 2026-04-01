"""
Escalation layer for prompt complexity.

Purpose: fix the core flaw where complexity is too linear.
We apply controlled boosts for combinations that are reliably hard for models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

from app.intelligence.complexity_constants import ESCALATION


@dataclass(frozen=True)
class BoostApplied:
    rule: str
    boost: float
    reason: str


@dataclass(frozen=True)
class EscalationResult:
    base_score: float
    boosts_applied: List[BoostApplied]
    total_boost: float
    final_score: float


def _is_at_least(x: float, threshold: float) -> bool:
    return float(x) >= float(threshold)


def apply_escalation(
    *,
    base_score: float,
    domain_scores: Mapping[str, float],
    signal_scores: Mapping[str, float],
) -> EscalationResult:
    """
    Apply explicit, explainable escalation rules.

    Inputs:
      - base_score: base weighted complexity score in [0,1]
      - domain_scores: normalized domain distribution
      - signal_scores: mapping signal_name -> score in [0,1]
    """

    coding = float(domain_scores.get("coding", 0.0))
    math = float(domain_scores.get("math", 0.0))
    reasoning = float(domain_scores.get("reasoning", 0.0))
    factual = float(domain_scores.get("factual_qa", 0.0))

    multi_task = float(signal_scores.get("multi_task", 0.0))
    constraints = float(signal_scores.get("constraints", 0.0))
    output_structure = float(signal_scores.get("output_structure", 0.0))
    step_by_step = float(signal_scores.get("step_by_step", 0.0))
    technical_structure = float(signal_scores.get("technical_structure", 0.0))
    reasoning_signal = float(signal_scores.get("reasoning", 0.0))
    ambiguity = float(signal_scores.get("ambiguity", 0.0))
    optimization_tradeoff = float(signal_scores.get("optimization_tradeoff", 0.0))
    length = float(signal_scores.get("length", 0.0))

    boosts: List[BoostApplied] = []

    # Rule Group 1: Composite Technical Task
    if _is_at_least(coding, 0.25) and _is_at_least(reasoning, 0.18) and _is_at_least(multi_task, 0.66):
        boosts.append(
            BoostApplied(
                rule="composite_technical_task",
                boost=ESCALATION.composite_technical_task_boost,
                reason="coding + reasoning + multi-task pattern detected",
            )
        )

    # Rule Group 2: Math / Algorithmic Explanation + Implementation
    if _is_at_least(math, 0.22) and (
        _is_at_least(coding, 0.14) or _is_at_least(technical_structure, 0.62)
    ) and (
        _is_at_least(step_by_step, 0.55) or _is_at_least(technical_structure, 0.6)
    ) and (
        _is_at_least(output_structure, 0.45) or _is_at_least(multi_task, 0.66)
    ):
        boosts.append(
            BoostApplied(
                rule="math_algorithmic_explanation_plus_implementation",
                boost=ESCALATION.math_algo_impl_boost,
                reason="math + coding with step-by-step/technical markers detected",
            )
        )
        # Extra bump when the prompt clearly expects correctness + explanation + implementation quality.
        if _is_at_least(technical_structure, 0.65) and _is_at_least(reasoning_signal, 0.45):
            boosts.append(
                BoostApplied(
                    rule="formal_math_explanation_with_implementation",
                    boost=0.05,
                    reason="strong technical structure + reasoning alongside math/implementation request",
                )
            )

    # Rule Group 3: Strong Constraints + Multi-Step Deliverables
    if _is_at_least(constraints, 0.65) and _is_at_least(output_structure, 0.6) and _is_at_least(multi_task, 0.66):
        boosts.append(
            BoostApplied(
                rule="strong_constraints_plus_multistep_deliverables",
                boost=ESCALATION.strong_constraints_multistep_boost,
                reason="constraints + output_structure + multi_task indicates rigid multi-output request",
            )
        )

    # Rule Group 4: Reasoning / Planning / Tradeoff Prompt
    if _is_at_least(reasoning, 0.26) and (_is_at_least(ambiguity, 0.55) or _is_at_least(optimization_tradeoff, 0.55)):
        boosts.append(
            BoostApplied(
                rule="reasoning_planning_tradeoff",
                boost=ESCALATION.reasoning_planning_tradeoff_boost,
                reason="reasoning + ambiguity/optimization-tradeoff pattern detected",
            )
        )
        # Strong instruction rigidity makes these prompts significantly harder.
        if _is_at_least(constraints, 0.50):
            boosts.append(
                BoostApplied(
                    rule="reasoning_with_rigid_constraints",
                    boost=0.04,
                    reason="reasoning/planning prompt with strong constraints (do not/must/justify)",
                )
            )
        # Special case: explicit \"reason carefully / resolve ambiguity / justify\" style prompts.
        if _is_at_least(reasoning_signal, 0.9) and _is_at_least(ambiguity, 0.8) and _is_at_least(constraints, 0.5):
            boosts.append(
                BoostApplied(
                    rule="planning_rigor_prompt",
                    boost=0.18,
                    reason="explicit rigorous reasoning + ambiguity resolution + strict instructions",
                )
            )

    # Rule Group 5: Technical Teaching Prompt
    if _is_at_least(factual, 0.20) and (_is_at_least(coding, 0.16) or _is_at_least(math, 0.16)) and _is_at_least(length, 0.35) and _is_at_least(output_structure, 0.45):
        boosts.append(
            BoostApplied(
                rule="technical_teaching_prompt",
                boost=ESCALATION.technical_teaching_boost,
                reason="factual explanation + (coding/math) + longer structured prompt",
            )
        )

    total_boost = sum(b.boost for b in boosts)
    total_boost = min(total_boost, ESCALATION.total_boost_cap)
    final_score = min(base_score + total_boost, ESCALATION.final_score_cap)

    return EscalationResult(
        base_score=round(float(base_score), 4),
        boosts_applied=boosts,
        total_boost=round(float(total_boost), 4),
        final_score=round(float(final_score), 4),
    )

