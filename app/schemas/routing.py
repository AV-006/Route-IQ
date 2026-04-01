from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCandidateScore(BaseModel):
    model_id: str
    provider: str
    display_name: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    domain_match_score: float = Field(..., ge=0.0, le=1.0)
    complexity_match_score: float = Field(..., ge=0.0, le=1.0)
    capability_match_score: float = Field(..., ge=0.0, le=1.0)
    latency_score: float = Field(..., ge=0.0, le=1.0)
    cost_score: float = Field(..., ge=0.0, le=1.0)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    penalties: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SelectedModelResult(BaseModel):
    selected_model: str
    provider: str
    display_name: str
    selection_confidence: float = Field(..., ge=0.0, le=1.0)
    selection_reasoning: list[str] = Field(default_factory=list)
    ranked_candidates: list[ModelCandidateScore] = Field(default_factory=list)

