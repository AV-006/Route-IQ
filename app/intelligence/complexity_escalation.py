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
class DeboostApplied:
    rule: str
    deboost: float
    reason: str


@dataclass(frozen=True)
class EscalationResult:
    base_score: float
    boosts_applied: List[BoostApplied]
    deboosts_applied: List[DeboostApplied]
    total_boost: float
    total_deboost: float
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
    summarization = float(domain_scores.get("summarization", 0.0))
    extraction = float(domain_scores.get("extraction", 0.0))
    transformation = float(domain_scores.get("transformation", 0.0))

    multi_task = float(signal_scores.get("multi_task", 0.0))
    constraints = float(signal_scores.get("constraints", 0.0))
    output_structure = float(signal_scores.get("output_structure", 0.0))
    step_by_step = float(signal_scores.get("step_by_step", 0.0))
    technical_structure = float(signal_scores.get("technical_structure", 0.0))
    reasoning_signal = float(signal_scores.get("reasoning", 0.0))
    cognitive_load = float(signal_scores.get("cognitive_load", 0.0))
    ambiguity = float(signal_scores.get("ambiguity", 0.0))
    optimization_tradeoff = float(signal_scores.get("optimization_tradeoff", 0.0))
    length = float(signal_scores.get("length", 0.0))

    boosts: List[BoostApplied] = []
    deboosts: List[DeboostApplied] = []

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
    if _is_at_least(reasoning, 0.26) and (
        _is_at_least(ambiguity, 0.55) or _is_at_least(optimization_tradeoff, 0.55) or _is_at_least(cognitive_load, 0.6)
    ):
        boosts.append(
            BoostApplied(
                rule="reasoning_planning_tradeoff",
                boost=ESCALATION.reasoning_planning_tradeoff_boost,
                reason="reasoning + ambiguity / tradeoff / cognitive-load pattern detected",
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
        if (
            _is_at_least(reasoning_signal, 0.8)
            and _is_at_least(ambiguity, 0.6)
            and _is_at_least(cognitive_load, 0.7)
        ):
            boosts.append(
                BoostApplied(
                    rule="planning_rigor_prompt",
                    boost=ESCALATION.deep_reasoning_combo_boost,
                    reason="explicit rigorous reasoning + ambiguity/assumptions + justification/alternatives",
                )
            )
        # Non-technical but clearly careful reasoning instructions.
        if _is_at_least(cognitive_load, 0.8) and not _is_at_least(technical_structure, 0.4):
            boosts.append(
                BoostApplied(
                    rule="explicit_careful_reasoning_language",
                    boost=ESCALATION.explicit_careful_reasoning_boost,
                    reason="language like 'do not jump to the answer' / 'reason carefully' detected",
                )
            )

    # Rule Group 5: Technical Teaching Prompt
    if (
        _is_at_least(factual, 0.20)
        and (_is_at_least(coding, 0.16) or _is_at_least(math, 0.16))
        and _is_at_least(length, 0.35)
        and _is_at_least(output_structure, 0.45)
    ):
        boosts.append(
            BoostApplied(
                rule="technical_teaching_prompt",
                boost=ESCALATION.technical_teaching_boost,
                reason="factual explanation + (coding/math) + longer structured prompt",
            )
        )

    # De-escalation: structure-heavy but cognitively light prompts.
    transform_only_domains = max(summarization, extraction, transformation)
    low_cognitive = (
        cognitive_load <= 0.25
        and reasoning_signal <= 0.25
        and ambiguity <= 0.25
        and optimization_tradeoff <= 0.25
    )

    # Strong structure (many deliverables / JSON / bullets) without much thinking difficulty.
    if _is_at_least(output_structure, 0.6) and _is_at_least(multi_task, 0.5) and low_cognitive and _is_at_least(
        transform_only_domains, 0.4
    ):
        deboosts.append(
            DeboostApplied(
                rule="structured_transform_only",
                deboost=ESCALATION.structured_transform_only_deboost,
                reason="summarize/extract/rewrite style multi-output prompt with low cognitive signals",
            )
        )
    # Mild correction when output structure dominates but there is at least some reasoning.
    elif _is_at_least(output_structure, 0.55) and cognitive_load < 0.5 and transform_only_domains > 0.3:
        deboosts.append(
            DeboostApplied(
                rule="structure_overweighted_vs_cognitive",
                deboost=ESCALATION.mild_structure_overweight_deboost,
                reason="output structure high while cognitive_load is moderate/low",
            )
        )

    total_boost = sum(b.boost for b in boosts)
    total_deboost = sum(d.deboost for d in deboosts)
    # Clamp within configured caps.
    total_boost = min(total_boost, ESCALATION.total_boost_cap)
    total_deboost = min(total_deboost, 0.20)

    final_score = base_score + total_boost - total_deboost
    final_score = min(max(final_score, 0.0), ESCALATION.final_score_cap)

    return EscalationResult(
        base_score=round(float(base_score), 4),
        boosts_applied=boosts,
        deboosts_applied=deboosts,
        total_boost=round(float(total_boost), 4),
        total_deboost=round(float(total_deboost), 4),
        final_score=round(float(final_score), 4),
    )

