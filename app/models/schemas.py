"""Request and response models for the Prompt Router Intelligence Engine."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """POST /analyze body."""

    prompt: str = Field(..., min_length=1, description="User prompt to analyze")


class DomainBreakdownEntry(BaseModel):
    """Per-domain scoring components before final normalization."""

    semantic_score: float = Field(..., ge=0.0, le=1.0)
    keyword_score: float = Field(..., ge=0.0, le=1.0)
    pattern_score: float = Field(..., ge=0.0, le=1.0)
    intent_score: float = Field(..., ge=0.0, le=1.0)
    phrase_score: float = Field(..., ge=0.0, le=1.0)
    raw_score: float = Field(..., ge=0.0, description="Weighted combination before global normalize")
    raw_cosine_similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Raw cosine vs domain prototype (not min-max relativized)",
    )
    matched_keywords: List[str] = Field(default_factory=list)
    matched_phrases: List[str] = Field(default_factory=list)


class TextFeatures(BaseModel):
    """Prompt-level lexical features for debugging and future routing."""

    token_count: int = Field(..., ge=0)
    avg_word_length: float = Field(..., ge=0.0)
    special_char_ratio: float = Field(..., ge=0.0, le=1.0)
    code_symbol_ratio: float = Field(..., ge=0.0, le=1.0)
    digit_ratio: float = Field(..., ge=0.0, le=1.0)
    uppercase_ratio: float = Field(..., ge=0.0, le=1.0)
    newline_count: int = Field(..., ge=0)


class AnalyzeResponse(BaseModel):
    """Full analysis result returned by POST /analyze."""

    prompt: str
    domain_scores: Dict[str, float]
    top_domains: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    per_domain_breakdown: Dict[str, DomainBreakdownEntry]
    text_features: TextFeatures
    complexity_score: float = Field(..., ge=0.0, le=1.0)
    complexity_band: Literal["very_low", "low", "medium", "high", "very_high"]
    complexity_signals: Dict[str, "ComplexitySignal"]
    complexity_escalation: "ComplexityEscalation"


class ComplexitySignal(BaseModel):
    """Explainable complexity signal for demo/debugging."""

    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0)
    contribution: float = Field(..., ge=0.0)
    evidence: List[str] = Field(default_factory=list)
    detail: Dict[str, object] = Field(default_factory=dict)


class ComplexityBoostApplied(BaseModel):
    rule: str
    boost: float = Field(..., ge=0.0)
    reason: str


class ComplexityEscalation(BaseModel):
    base_score: float = Field(..., ge=0.0, le=1.0)
    boosts_applied: List[ComplexityBoostApplied] = Field(default_factory=list)
    total_boost: float = Field(..., ge=0.0)
    deboosts_applied: List[ComplexityBoostApplied] = Field(default_factory=list)
    total_deboost: float = Field(..., ge=0.0)
    final_score: float = Field(..., ge=0.0, le=1.0)


AnalyzeResponse.model_rebuild()


class DomainMetadata(BaseModel):
    """Summary info for one supported domain (GET /domains)."""

    name: str
    anchor_count: int
    keyword_count: int
    pattern_count: int
    intent_verb_count: int
    phrase_boost_count: int


class DomainsListResponse(BaseModel):
    """GET /domains response."""

    domains: List[DomainMetadata]
    embedding_model: str


class HealthResponse(BaseModel):
    """GET / root info."""

    service: str
    version: str
    status: str
    embedding_model: str
    domains_supported: List[str]
