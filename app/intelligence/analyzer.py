"""
High-level orchestration: embed prompt, score domains, normalize, attach features.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.constants import DOMAIN_NAMES
from app.intelligence.confidence import compute_confidence, top_k_domains
from app.intelligence.complexity import PromptComplexityEngine
from app.intelligence.domain_scorer import HybridDomainScorer
from app.intelligence.embeddings import EmbeddingService
from app.intelligence.feature_extractor import extract_text_features
from app.models.schemas import (
    AnalyzeResponse,
    ComplexityBoostApplied,
    ComplexityEscalation,
    ComplexitySignal,
    DomainBreakdownEntry,
    TextFeatures,
)
from app.utils.math_utils import normalize_nonneg_sum_to_one, uniform_distribution


class PromptDomainAnalyzer:
    """
    End-to-end analyzer using a shared EmbeddingService and HybridDomainScorer.

    Stateless aside from service references; safe for concurrent requests.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        scorer: HybridDomainScorer | None = None,
    ) -> None:
        self._embed = embedding_service
        self._scorer = scorer or HybridDomainScorer()
        self._complexity = PromptComplexityEngine()

    def analyze(self, prompt: str) -> AnalyzeResponse:
        """
        Produce domain weights, breakdown, confidence, and text features.

        Embedding and prototypes must be initialized on the EmbeddingService.
        """
        stripped = prompt.strip()
        text_features: TextFeatures = extract_text_features(stripped)

        prompt_vec = self._embed.encode_prompt(stripped)
        raw_cos = self._embed.raw_cosine_similarities(prompt_vec)
        semantic = self._embed.relative_semantic_scores(prompt_vec)
        breakdown = self._scorer.compute_breakdown(stripped, semantic, raw_cos)

        raw_scores = {d: breakdown[d].raw_score for d in DOMAIN_NAMES}
        exp = settings.pre_normalize_raw_sharpen_exponent
        sharpened = {d: max(0.0, raw_scores[d] ** exp) for d in DOMAIN_NAMES}
        normalized = normalize_nonneg_sum_to_one(sharpened)
        if not normalized:
            normalized = uniform_distribution(list(DOMAIN_NAMES))

        normalized = {k: round(v, 4) for k, v in normalized.items()}

        confidence = compute_confidence(normalized, token_count=text_features.token_count)
        tops = top_k_domains(normalized, k=3)

        complexity = self._complexity.analyze(
            prompt=stripped,
            domain_scores=normalized,
            text_features=text_features,
        )

        per_domain_public = {
            d: DomainBreakdownEntry(
                semantic_score=round(breakdown[d].semantic_score, 4),
                keyword_score=round(breakdown[d].keyword_score, 4),
                pattern_score=round(breakdown[d].pattern_score, 4),
                intent_score=round(breakdown[d].intent_score, 4),
                phrase_score=round(breakdown[d].phrase_score, 4),
                raw_score=round(breakdown[d].raw_score, 4),
                raw_cosine_similarity=round(breakdown[d].raw_cosine_similarity, 4),
                matched_keywords=list(breakdown[d].matched_keywords),
                matched_phrases=list(breakdown[d].matched_phrases),
            )
            for d in DOMAIN_NAMES
        }

        complexity_signals_public = {
            name: ComplexitySignal(
                name=s.name,
                score=round(s.score, 4),
                weight=round(s.weight, 4),
                contribution=round(s.contribution, 6),
                evidence=list(s.evidence),
                detail=dict(s.detail),
            )
            for name, s in complexity.signals.items()
        }

        return AnalyzeResponse(
            prompt=prompt,
            domain_scores=normalized,
            top_domains=tops,
            confidence=round(confidence, 4),
            per_domain_breakdown=per_domain_public,
            text_features=text_features,
            complexity_score=complexity.complexity_score,
            complexity_band=complexity.complexity_band,
            complexity_signals=complexity_signals_public,
            complexity_escalation=ComplexityEscalation(
                base_score=float(complexity.escalation.get("base_score", complexity.complexity_score)),
                boosts_applied=[
                    ComplexityBoostApplied(
                        rule=str(b.get("rule")),
                        boost=float(b.get("boost", 0.0)),
                        reason=str(b.get("reason")),
                    )
                    for b in list(complexity.escalation.get("boosts_applied", []))
                ],
                total_boost=float(complexity.escalation.get("total_boost", 0.0)),
                deboosts_applied=[
                    ComplexityBoostApplied(
                        rule=str(b.get("rule")),
                        boost=float(b.get("deboost", 0.0)),
                        reason=str(b.get("reason")),
                    )
                    for b in list(complexity.escalation.get("deboosts_applied", []))
                ],
                total_deboost=float(complexity.escalation.get("total_deboost", 0.0)),
                final_score=float(complexity.escalation.get("final_score", complexity.complexity_score)),
            ),
        )
