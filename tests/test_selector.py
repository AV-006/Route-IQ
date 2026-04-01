from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.models.schemas import (
    AnalyzeResponse,
    ComplexityEscalation,
    ComplexitySignal,
    DomainBreakdownEntry,
    TextFeatures,
)
from app.registry.model_registry import DOMAIN_KEYS, ModelRegistry
from app.services.model_selector import select_best_model


def _registry_default() -> ModelRegistry:
    registry = ModelRegistry(Path(__file__).resolve().parents[1] / "app" / "registry" / "models.json")
    registry.load_models()
    return registry


def _build_analysis(
    prompt: str,
    domain_scores: dict[str, float],
    top_domains: list[str],
    complexity_band: str,
    complexity_score: float,
    confidence: float,
    token_count: int = 40,
    code_symbol_ratio: float = 0.0,
) -> AnalyzeResponse:
    full_scores = {k: float(domain_scores.get(k, 0.0)) for k in DOMAIN_KEYS}
    per_domain = {
        k: DomainBreakdownEntry(
            semantic_score=0.5,
            keyword_score=0.2,
            pattern_score=0.1,
            intent_score=0.1,
            phrase_score=0.1,
            raw_score=0.5,
            raw_cosine_similarity=0.3,
            matched_keywords=[],
            matched_phrases=[],
        )
        for k in DOMAIN_KEYS
    }
    signals = {
        "length": ComplexitySignal(
            name="length",
            score=0.3,
            weight=0.2,
            contribution=0.06,
            evidence=[],
            detail={},
        )
    }
    return AnalyzeResponse(
        prompt=prompt,
        domain_scores=full_scores,
        top_domains=top_domains,
        confidence=confidence,
        per_domain_breakdown=per_domain,
        text_features=TextFeatures(
            token_count=token_count,
            avg_word_length=4.1,
            special_char_ratio=0.01,
            code_symbol_ratio=code_symbol_ratio,
            digit_ratio=0.05,
            uppercase_ratio=0.01,
            newline_count=0,
        ),
        complexity_score=complexity_score,
        complexity_band=complexity_band,  # type: ignore[arg-type]
        complexity_signals=signals,
        complexity_escalation=ComplexityEscalation(
            base_score=complexity_score,
            boosts_applied=[],
            total_boost=0.0,
            deboosts_applied=[],
            total_deboost=0.0,
            final_score=complexity_score,
        ),
    )


def test_selector_returns_model() -> None:
    analysis = _build_analysis(
        prompt="write code for BFS in python",
        domain_scores={"coding": 0.6, "reasoning": 0.2, "factual_qa": 0.2},
        top_domains=["coding", "reasoning", "factual_qa"],
        complexity_band="high",
        complexity_score=0.78,
        confidence=0.7,
        code_symbol_ratio=0.05,
    )
    result = select_best_model(analysis, _registry_default())
    assert result.selected_model
    assert result.display_name


def test_selector_ranks_candidates() -> None:
    analysis = _build_analysis(
        prompt="explain this architecture",
        domain_scores={"reasoning": 0.5, "factual_qa": 0.3, "summarization": 0.2},
        top_domains=["reasoning", "factual_qa", "summarization"],
        complexity_band="medium",
        complexity_score=0.55,
        confidence=0.64,
    )
    result = select_best_model(analysis, _registry_default())
    assert len(result.ranked_candidates) > 0
    scores = [x.final_score for x in result.ranked_candidates]
    assert scores == sorted(scores, reverse=True)


def test_high_math_prompt_prefers_math_capable_model() -> None:
    analysis = _build_analysis(
        prompt="prove this recurrence theorem by induction and justify every step",
        domain_scores={"math": 0.55, "reasoning": 0.30, "factual_qa": 0.15},
        top_domains=["math", "reasoning", "factual_qa"],
        complexity_band="high",
        complexity_score=0.88,
        confidence=0.76,
    )
    result = select_best_model(analysis, _registry_default())
    assert result.selected_model in {"anthropic.claude-3-sonnet", "mistral.mistral-large"}


def test_coding_prompt_prefers_coding_model() -> None:
    analysis = _build_analysis(
        prompt="implement optimized graph search with tests",
        domain_scores={"coding": 0.65, "reasoning": 0.2, "factual_qa": 0.15},
        top_domains=["coding", "reasoning", "factual_qa"],
        complexity_band="high",
        complexity_score=0.80,
        confidence=0.75,
        code_symbol_ratio=0.07,
    )
    result = select_best_model(analysis, _registry_default())
    assert result.selected_model == "anthropic.claude-3-sonnet"


def test_simple_summary_prompt_prefers_cheaper_model() -> None:
    analysis = _build_analysis(
        prompt="summarize this short email in two bullets",
        domain_scores={"summarization": 0.6, "transformation": 0.25, "factual_qa": 0.15},
        top_domains=["summarization", "transformation", "factual_qa"],
        complexity_band="low",
        complexity_score=0.2,
        confidence=0.72,
        token_count=18,
    )
    result = select_best_model(analysis, _registry_default())
    selected = next(x for x in result.ranked_candidates if x.model_id == result.selected_model)
    assert selected.model_id in {"anthropic.claude-3-haiku", "amazon.nova-lite"}


def test_selection_confidence_exists() -> None:
    analysis = _build_analysis(
        prompt="extract entities from text",
        domain_scores={"extraction": 0.55, "summarization": 0.25, "factual_qa": 0.2},
        top_domains=["extraction", "summarization", "factual_qa"],
        complexity_band="medium",
        complexity_score=0.4,
        confidence=0.58,
    )
    result = select_best_model(analysis, _registry_default())
    assert 0.0 <= result.selection_confidence <= 1.0


def test_analysis_endpoint_includes_model_selection() -> None:
    analysis = _build_analysis(
        prompt="write a python function",
        domain_scores={"coding": 0.7, "reasoning": 0.2, "factual_qa": 0.1},
        top_domains=["coding", "reasoning", "factual_qa"],
        complexity_band="medium",
        complexity_score=0.6,
        confidence=0.7,
        code_symbol_ratio=0.06,
    )

    class FakeAnalyzer:
        def analyze(self, prompt: str) -> AnalyzeResponse:
            return analysis.model_copy(update={"prompt": prompt})

    app = FastAPI()
    app.include_router(router)
    app.state.analyzer = FakeAnalyzer()
    app.state.model_registry = _registry_default()

    client = TestClient(app)
    resp = client.post("/analyze", json={"prompt": "write function for quicksort"})
    assert resp.status_code == 200
    payload = resp.json()
    assert "model_selection" in payload
    assert payload["model_selection"]["selected_model"]
    assert isinstance(payload["model_selection"]["ranked_candidates"], list)


def test_ineligible_models_are_filtered(tmp_path: Path) -> None:
    models = [
        {
            "model_id": "inactive.model",
            "display_name": "Inactive Model",
            "provider": "Test",
            "active": False,
            "bedrock_available": True,
            "free_tier_accessible": True,
            "latency_tier": "fast",
            "cost_tier": "cheap",
            "context_window": 16000,
            "preferred_complexity": ["low", "medium"],
            "domain_scores": {k: 0.8 for k in DOMAIN_KEYS},
            "supports_long_context": True,
            "supports_structured_output": True,
            "supports_code_generation": True,
            "supports_reasoning_heavy_tasks": True,
            "fallback_rank": 1,
            "notes": ""
        },
        {
            "model_id": "active.model",
            "display_name": "Active Model",
            "provider": "Test",
            "active": True,
            "bedrock_available": True,
            "free_tier_accessible": True,
            "latency_tier": "medium",
            "cost_tier": "cheap",
            "context_window": 16000,
            "preferred_complexity": ["low", "medium", "high"],
            "domain_scores": {k: 0.6 for k in DOMAIN_KEYS},
            "supports_long_context": True,
            "supports_structured_output": True,
            "supports_code_generation": True,
            "supports_reasoning_heavy_tasks": True,
            "fallback_rank": 2,
            "notes": ""
        }
    ]
    path = tmp_path / "models.json"
    path.write_text(json.dumps(models), encoding="utf-8")

    registry = ModelRegistry(path)
    registry.load_models()
    analysis = _build_analysis(
        prompt="coding task",
        domain_scores={"coding": 0.8, "reasoning": 0.2},
        top_domains=["coding", "reasoning", "factual_qa"],
        complexity_band="medium",
        complexity_score=0.5,
        confidence=0.8,
    )
    result = select_best_model(analysis, registry)
    ids = [x.model_id for x in result.ranked_candidates]
    assert "inactive.model" not in ids
    assert "active.model" in ids

