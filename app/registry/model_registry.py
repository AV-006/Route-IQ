from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

LatencyTier = Literal["fast", "medium", "slow"]
CostTier = Literal["free", "cheap", "moderate", "premium"]
ComplexityBand = Literal["low", "medium", "high"]

DOMAIN_KEYS: tuple[str, ...] = (
    "coding",
    "math",
    "reasoning",
    "summarization",
    "extraction",
    "creative_writing",
    "factual_qa",
    "transformation",
)


class ModelRegistryError(Exception):
    pass


class ModelRegistryValidationError(ModelRegistryError):
    pass


@dataclass(frozen=True, slots=True)
class DomainCapabilityScores:
    coding: float
    math: float
    reasoning: float
    summarization: float
    extraction: float
    creative_writing: float
    factual_qa: float
    transformation: float

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in DOMAIN_KEYS}

    def get(self, domain: str) -> float:
        if domain not in DOMAIN_KEYS:
            raise ModelRegistryValidationError(f"Unknown domain '{domain}'")
        return getattr(self, domain)


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    model_id: str
    display_name: str
    provider: str
    active: bool
    bedrock_available: bool
    free_tier_accessible: bool
    latency_tier: LatencyTier
    cost_tier: CostTier
    context_window: int
    preferred_complexity: tuple[ComplexityBand, ...]
    domain_scores: DomainCapabilityScores
    supports_long_context: bool
    supports_structured_output: bool
    supports_code_generation: bool
    supports_reasoning_heavy_tasks: bool
    fallback_rank: int
    notes: str = ""


class ModelRegistry:
    def __init__(self, models_path: str | Path | None = None) -> None:
        self._models_path = Path(models_path) if models_path else Path(__file__).with_name("models.json")
        self._models_by_id: dict[str, ModelDefinition] = {}

    def load_models(self) -> None:
        raw = json.loads(self._models_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ModelRegistryValidationError("models.json must be an array")

        models: list[ModelDefinition] = [self._parse_model(x) for x in raw]
        seen: set[str] = set()
        for model in models:
            if model.model_id in seen:
                raise ModelRegistryValidationError(f"Duplicate model_id: {model.model_id}")
            seen.add(model.model_id)
        self._models_by_id = {m.model_id: m for m in models}

    def get_all_models(self) -> list[ModelDefinition]:
        return list(self._models_by_id.values())

    def get_active_models(self) -> list[ModelDefinition]:
        return [m for m in self._models_by_id.values() if m.active and m.bedrock_available]

    def _parse_model(self, raw: Mapping[str, Any]) -> ModelDefinition:
        domain_raw = raw.get("domain_scores")
        if not isinstance(domain_raw, dict):
            raise ModelRegistryValidationError("domain_scores must be an object")
        domain_scores = self._parse_domain_scores(domain_raw)

        preferred = raw.get("preferred_complexity", [])
        if not isinstance(preferred, list) or not preferred:
            raise ModelRegistryValidationError("preferred_complexity must be a non-empty list")
        for p in preferred:
            if p not in {"low", "medium", "high"}:
                raise ModelRegistryValidationError(f"Invalid preferred_complexity value '{p}'")

        cost_tier = str(raw.get("cost_tier"))
        latency_tier = str(raw.get("latency_tier"))
        if cost_tier not in {"free", "cheap", "moderate", "premium"}:
            raise ModelRegistryValidationError(f"Invalid cost_tier '{cost_tier}'")
        if latency_tier not in {"fast", "medium", "slow"}:
            raise ModelRegistryValidationError(f"Invalid latency_tier '{latency_tier}'")

        return ModelDefinition(
            model_id=str(raw.get("model_id")),
            display_name=str(raw.get("display_name")),
            provider=str(raw.get("provider")),
            active=bool(raw.get("active")),
            bedrock_available=bool(raw.get("bedrock_available")),
            free_tier_accessible=bool(raw.get("free_tier_accessible")),
            latency_tier=latency_tier,  # type: ignore[arg-type]
            cost_tier=cost_tier,  # type: ignore[arg-type]
            context_window=int(raw.get("context_window", 0)),
            preferred_complexity=tuple(preferred),  # type: ignore[arg-type]
            domain_scores=domain_scores,
            supports_long_context=bool(raw.get("supports_long_context")),
            supports_structured_output=bool(raw.get("supports_structured_output")),
            supports_code_generation=bool(raw.get("supports_code_generation")),
            supports_reasoning_heavy_tasks=bool(raw.get("supports_reasoning_heavy_tasks")),
            fallback_rank=int(raw.get("fallback_rank", 0)),
            notes=str(raw.get("notes", "")),
        )

    def _parse_domain_scores(self, raw: Mapping[str, Any]) -> DomainCapabilityScores:
        vals: dict[str, float] = {}
        for key in DOMAIN_KEYS:
            if key not in raw:
                raise ModelRegistryValidationError(f"Missing domain score key '{key}'")
            value = float(raw[key])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ModelRegistryValidationError(f"Invalid domain score for '{key}': {value}")
            vals[key] = value
        return DomainCapabilityScores(**vals)  # type: ignore[arg-type]

