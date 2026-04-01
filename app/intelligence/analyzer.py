"""
High-level orchestration: embed prompt, score domains, normalize, attach features.
"""

from __future__ import annotations

from app.core.constants import DOMAIN_NAMES
from app.intelligence.confidence import compute_confidence, top_k_domains
from app.intelligence.domain_scorer import HybridDomainScorer
from app.intelligence.embeddings import EmbeddingService
from app.intelligence.feature_extractor import extract_text_features
from app.models.schemas import AnalyzeResponse, DomainBreakdownEntry, TextFeatures
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

    def analyze(self, prompt: str) -> AnalyzeResponse:
        """
        Produce domain weights, breakdown, confidence, and text features.

        Embedding and prototypes must be initialized on the EmbeddingService.
        """
        stripped = prompt.strip()
        prompt_vec = self._embed.encode_prompt(stripped)
        semantic = self._embed.semantic_scores_unit_interval(prompt_vec)
        breakdown = self._scorer.compute_breakdown(stripped, semantic)

        raw_scores = {d: breakdown[d].raw_score for d in DOMAIN_NAMES}
        normalized = normalize_nonneg_sum_to_one(raw_scores)
        if not normalized:
            normalized = uniform_distribution(list(DOMAIN_NAMES))

        normalized = {k: round(v, 4) for k, v in normalized.items()}

        confidence = compute_confidence(normalized)
        tops = top_k_domains(normalized, k=3)

        per_domain_public = {
            d: DomainBreakdownEntry(
                semantic_score=round(breakdown[d].semantic_score, 4),
                keyword_score=round(breakdown[d].keyword_score, 4),
                pattern_score=round(breakdown[d].pattern_score, 4),
                intent_score=round(breakdown[d].intent_score, 4),
                raw_score=round(breakdown[d].raw_score, 4),
            )
            for d in DOMAIN_NAMES
        }

        text_features: TextFeatures = extract_text_features(stripped)

        return AnalyzeResponse(
            prompt=prompt,
            domain_scores=normalized,
            top_domains=tops,
            confidence=round(confidence, 4),
            per_domain_breakdown=per_domain_public,
            text_features=text_features,
        )
