"""Centralized settings for the domain weighting engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for hybrid domain scoring components."""

    semantic: float = 0.55
    keyword: float = 0.20
    pattern: float = 0.10
    intent: float = 0.15

    def __post_init__(self) -> None:
        total = self.semantic + self.keyword + self.pattern + self.intent
        if abs(total - 1.0) > 1e-6:
            raise ValueError("ScoringWeights must sum to 1.0")


@dataclass(frozen=True)
class AppSettings:
    """Application settings loaded once at startup."""

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    scoring_weights: ScoringWeights = ScoringWeights()
    # Keyword scoring: cap raw keyword hit rate to avoid overpowering semantic
    keyword_match_cap: int = 8
    # Minimum regex pattern matches before maxing pattern score contribution paths
    pattern_score_max_hits: int = 4


# Singleton-style settings instance (immutable)
settings = AppSettings()
