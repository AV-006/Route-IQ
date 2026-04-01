from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.schemas import AnalyzeResponse
from app.registry.model_registry import ModelDefinition, ModelRegistry
from app.schemas.routing import ModelCandidateScore, SelectedModelResult


@dataclass(frozen=True)
class ScoreWeights:
    domain_match_score: float
    complexity_match_score: float
    capability_match_score: float
    latency_score: float
    cost_score: float
    quality_score: float


WEIGHTS_BY_BAND: dict[str, ScoreWeights] = {
    "low": ScoreWeights(0.22, 0.18, 0.18, 0.16, 0.16, 0.10),
    "medium": ScoreWeights(0.24, 0.20, 0.20, 0.12, 0.10, 0.14),
    "high": ScoreWeights(0.24, 0.22, 0.20, 0.08, 0.06, 0.20),
}


def select_best_model(analysis_result: AnalyzeResponse, registry: ModelRegistry) -> SelectedModelResult:
    """Rank eligible models and return a selected primary model with explainability."""
    selector = ModelSelectorService(registry)
    return selector.select(analysis_result)


class ModelSelectorService:
    _LATENCY_RANK = {"fast": 0, "medium": 1, "slow": 2}
    _COST_RANK = {"free": 0, "cheap": 1, "moderate": 2, "premium": 3}

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def select(self, analysis_result: AnalyzeResponse) -> SelectedModelResult:
        required_capabilities = self._infer_required_capabilities(analysis_result)
        eligible = self._filter_eligible_models(analysis_result, required_capabilities)
        if not eligible:
            eligible = self._registry.get_active_models()
        if not eligible:
            raise RuntimeError("No active models available for selection")

        scored = [
            self._score_model(model, analysis_result, required_capabilities)
            for model in eligible
        ]
        scored.sort(key=lambda x: x.final_score, reverse=True)
        top = scored[0]
        confidence = self._compute_selection_confidence(scored)

        return SelectedModelResult(
            selected_model=top.model_id,
            provider=top.provider,
            display_name=top.display_name,
            selection_confidence=confidence,
            selection_reasoning=top.reasons[:4],
            ranked_candidates=scored,
        )

    def _score_model(
        self,
        model: ModelDefinition,
        analysis: AnalyzeResponse,
        required_capabilities: list[str],
    ) -> ModelCandidateScore:
        reasons: list[str] = []
        penalties: list[str] = []

        complexity_band = self._normalize_complexity_band(analysis.complexity_band)
        weights = WEIGHTS_BY_BAND[complexity_band]

        domain = self._score_domain_match(model, analysis, reasons)
        complexity = self._score_complexity_match(model, analysis, reasons, penalties)
        capability = self._score_capability_match(model, required_capabilities, reasons, penalties)
        latency = self._score_latency(model, complexity_band, reasons, penalties)
        cost = self._score_cost(model, complexity_band, reasons, penalties)
        quality = self._score_quality(model, analysis, reasons)

        final_score = (
            weights.domain_match_score * domain
            + weights.complexity_match_score * complexity
            + weights.capability_match_score * capability
            + weights.latency_score * latency
            + weights.cost_score * cost
            + weights.quality_score * quality
        )

        return ModelCandidateScore(
            model_id=model.model_id,
            provider=model.provider,
            display_name=model.display_name,
            final_score=round(max(0.0, min(1.0, final_score)), 6),
            domain_match_score=round(domain, 6),
            complexity_match_score=round(complexity, 6),
            capability_match_score=round(capability, 6),
            latency_score=round(latency, 6),
            cost_score=round(cost, 6),
            quality_score=round(quality, 6),
            penalties=penalties,
            reasons=reasons,
        )

    def _filter_eligible_models(
        self, analysis: AnalyzeResponse, required_capabilities: list[str]
    ) -> list[ModelDefinition]:
        top_domains = set(analysis.top_domains)
        normalized_band = self._normalize_complexity_band(analysis.complexity_band)
        token_count = analysis.text_features.token_count

        out: list[ModelDefinition] = []
        for model in self._registry.get_active_models():
            if model.context_window > 0 and token_count > model.context_window:
                continue
            if normalized_band not in model.preferred_complexity:
                continue
            if not any(model.domain_scores.get(domain) >= 0.5 for domain in top_domains):
                continue
            # For hybrid cases, require at least half capabilities as soft eligibility gate.
            if required_capabilities:
                covered = sum(self._has_capability(model, cap) for cap in required_capabilities)
                if covered < max(1, len(required_capabilities) // 2):
                    continue
            out.append(model)
        return out

    def _score_domain_match(
        self,
        model: ModelDefinition,
        analysis: AnalyzeResponse,
        reasons: list[str],
    ) -> float:
        weighted_sum = 0.0
        weights_total = 0.0
        for domain, weight in analysis.domain_scores.items():
            if weight <= 0:
                continue
            weighted_sum += weight * model.domain_scores.get(domain)
            weights_total += weight
        score = weighted_sum / weights_total if weights_total > 0 else 0.0
        if score >= 0.75:
            reasons.append(f"Strong match for {analysis.top_domains[0]} domain")
        return score

    def _score_complexity_match(
        self,
        model: ModelDefinition,
        analysis: AnalyzeResponse,
        reasons: list[str],
        penalties: list[str],
    ) -> float:
        band = self._normalize_complexity_band(analysis.complexity_band)
        score = 0.35
        if band in model.preferred_complexity:
            score = 0.8
            reasons.append(f"Well suited for {band} complexity prompts")
        if band == "high":
            if model.supports_reasoning_heavy_tasks:
                score += 0.2
                reasons.append("Supports heavy reasoning for difficult prompts")
            else:
                score -= 0.25
                penalties.append("Weaker support for high-complexity reasoning")
        return max(0.0, min(1.0, score))

    def _score_capability_match(
        self,
        model: ModelDefinition,
        required_capabilities: list[str],
        reasons: list[str],
        penalties: list[str],
    ) -> float:
        if not required_capabilities:
            return 0.75
        matched = 0
        for cap in required_capabilities:
            if self._has_capability(model, cap):
                matched += 1
                reasons.append(f"Supports {cap} capability")
            else:
                penalties.append(f"Weaker support for {cap}")
        return matched / len(required_capabilities)

    def _score_latency(
        self,
        model: ModelDefinition,
        complexity_band: str,
        reasons: list[str],
        penalties: list[str],
    ) -> float:
        rank = self._LATENCY_RANK[model.latency_tier]
        base = 1.0 - (rank / 2.0)
        if complexity_band == "high":
            # High complexity cares less about latency.
            score = 0.7 + 0.3 * base
        else:
            score = base
        if score > 0.8:
            reasons.append("Low latency profile")
        elif score < 0.4:
            penalties.append("Higher latency")
        return max(0.0, min(1.0, score))

    def _score_cost(
        self,
        model: ModelDefinition,
        complexity_band: str,
        reasons: list[str],
        penalties: list[str],
    ) -> float:
        rank = self._COST_RANK[model.cost_tier]
        base = 1.0 - (rank / 3.0)
        if complexity_band == "high":
            score = 0.7 + 0.3 * base
        else:
            score = base
        if score > 0.8:
            reasons.append("Cost-efficient option")
        elif score < 0.4:
            penalties.append("More expensive than lightweight alternatives")
        return max(0.0, min(1.0, score))

    def _score_quality(
        self,
        model: ModelDefinition,
        analysis: AnalyzeResponse,
        reasons: list[str],
    ) -> float:
        domain_values = list(model.domain_scores.as_dict().values())
        base_quality = sum(domain_values) / len(domain_values)
        if analysis.complexity_band in ("high", "very_high") and model.supports_reasoning_heavy_tasks:
            base_quality = min(1.0, base_quality + 0.12)
        if analysis.confidence < 0.45:
            spread = max(domain_values) - min(domain_values)
            base_quality += (1.0 - spread) * 0.05
        if base_quality >= 0.8:
            reasons.append("High overall quality for challenging tasks")
        return max(0.0, min(1.0, base_quality))

    def _infer_required_capabilities(self, analysis: AnalyzeResponse) -> list[str]:
        required: set[str] = set()
        d = analysis.domain_scores

        if d.get("coding", 0.0) >= 0.30 or analysis.text_features.code_symbol_ratio > 0.03:
            required.add("coding")
        if d.get("extraction", 0.0) >= 0.25:
            required.add("structured_output")
        if d.get("reasoning", 0.0) >= 0.22:
            required.add("reasoning")
        if d.get("math", 0.0) >= 0.20:
            required.add("math")
        if d.get("summarization", 0.0) >= 0.25:
            required.add("summarization")
        if d.get("transformation", 0.0) >= 0.25:
            required.add("rewriting")
        if len([k for k, v in d.items() if v >= 0.15]) >= 3 and analysis.complexity_band in ("high", "very_high"):
            required.add("hybrid_reasoning")
        return sorted(required)

    def _has_capability(self, model: ModelDefinition, capability: str) -> bool:
        if capability == "structured_output":
            return model.supports_structured_output
        if capability == "coding":
            return model.supports_code_generation
        if capability in {"reasoning", "hybrid_reasoning", "math"}:
            return model.supports_reasoning_heavy_tasks or model.domain_scores.reasoning >= 0.7
        if capability in {"summarization", "rewriting"}:
            return model.domain_scores.summarization >= 0.65 or model.domain_scores.transformation >= 0.65
        return True

    def _normalize_complexity_band(self, band: str) -> str:
        if band in ("very_low", "low"):
            return "low"
        if band in ("high", "very_high"):
            return "high"
        return "medium"

    def _compute_selection_confidence(self, ranked: list[ModelCandidateScore]) -> float:
        top = ranked[0].final_score
        second = ranked[1].final_score if len(ranked) > 1 else 0.0
        gap = max(0.0, top - second)
        confidence = (0.65 * top) + (0.35 * min(1.0, gap * 3.0))
        return round(max(0.0, min(1.0, confidence)), 4)

