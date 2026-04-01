"""Centralized settings for the domain weighting engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for hybrid domain scoring components."""

    semantic: float = 0.45
    keyword: float = 0.18
    pattern: float = 0.10
    intent: float = 0.12
    phrase: float = 0.15

    def __post_init__(self) -> None:
        total = (
            self.semantic + self.keyword + self.pattern + self.intent + self.phrase
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("ScoringWeights must sum to 1.0")


@dataclass(frozen=True)
class AppSettings:
    """Application settings loaded once at startup."""

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    scoring_weights: ScoringWeights = ScoringWeights()
    # Relative semantic: map (cos - min) / (max - min) then raise to this power
    semantic_relative_sharpen_exponent: float = 1.5
    # Mild competition / sharpening applied to per-domain raw scores before softmax norm
    pre_normalize_raw_sharpen_exponent: float = 1.35
    # Capped lexical scoring divisors (matches / divisor clamped to 1.0)
    keyword_match_normalize_divisor: float = 4.0
    pattern_match_normalize_divisor: float = 3.0
    phrase_match_normalize_divisor: float = 3.0
    intent_match_normalize_divisor: float = 3.0


# Singleton-style settings instance (immutable)
settings = AppSettings()
