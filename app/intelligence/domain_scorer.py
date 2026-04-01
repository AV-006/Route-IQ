"""
Hybrid domain scoring: relative semantic + keyword + regex pattern + intent + phrase boosts.

Semantic scores (already per-prompt relative) are supplied from EmbeddingService; this module
combines them with capped lexical signals for a sharper, explainable distribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.core.config import settings
from app.core.constants import DOMAIN_NAMES
from app.intelligence.domain_configs import DOMAIN_REGISTRY, DomainDefinition
from app.utils.math_utils import clamp01
from app.utils.text import canonicalize_prompt_for_matching, word_boundary_contains


@dataclass(frozen=True)
class DomainRawBreakdown:
    """Intermediate scores for one domain before global normalization."""

    semantic_score: float
    keyword_score: float
    pattern_score: float
    intent_score: float
    phrase_score: float
    raw_score: float
    raw_cosine_similarity: float
    matched_keywords: List[str]
    matched_phrases: List[str]


class HybridDomainScorer:
    """
    Computes keyword, pattern, intent, and phrase subscores per domain and blends with semantic.

    Uses configurable weights from app.core.config.settings.scoring_weights.
    """

    def __init__(self) -> None:
        self._compiled_patterns: Dict[str, List[re.Pattern[str]]] = {}
        for name in DOMAIN_NAMES:
            pats = DOMAIN_REGISTRY[name].patterns
            self._compiled_patterns[name] = [
                re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats
            ]

    def _keyword_matches(self, text_canon: str, definition: DomainDefinition) -> Tuple[float, List[str]]:
        matched: List[str] = []
        for kw in definition.keywords:
            if word_boundary_contains(text_canon, kw):
                matched.append(kw)
        cap = settings.keyword_match_normalize_divisor
        score = clamp01(len(matched) / cap)
        return score, matched

    def _pattern_score(self, text_raw: str, domain: str) -> float:
        """Count regex hits on original text; cap normalization."""
        compiled = self._compiled_patterns[domain]
        matches = sum(1 for rx in compiled if rx.search(text_raw))
        cap = settings.pattern_match_normalize_divisor
        return clamp01(matches / cap)

    def _intent_matches(self, text_canon: str, definition: DomainDefinition) -> Tuple[float, int]:
        hits = 0
        for verb in definition.intent_verbs:
            if word_boundary_contains(text_canon, verb):
                hits += 1
        cap = settings.intent_match_normalize_divisor
        return clamp01(hits / cap), hits

    def _phrase_matches(self, text_canon: str, definition: DomainDefinition) -> Tuple[float, List[str]]:
        matched: List[str] = []
        low = text_canon.lower()
        for phrase in definition.phrase_boosts:
            p = phrase.lower().strip()
            if p and p in low:
                matched.append(phrase)
        cap = settings.phrase_match_normalize_divisor
        score = clamp01(len(matched) / cap)
        return score, matched

    def compute_breakdown(
        self,
        prompt: str,
        semantic_by_domain: Dict[str, float],
        raw_cosine_by_domain: Dict[str, float],
    ) -> Dict[str, DomainRawBreakdown]:
        """
        Build per-domain breakdown and weighted raw_score (pre-global-normalize).

        semantic_by_domain must be relative sharpened scores in [0,1] per domain.
        raw_cosine_by_domain is native cosine similarity per domain (for explainability).
        """
        w = settings.scoring_weights
        text_raw = prompt.strip()
        text_canon = canonicalize_prompt_for_matching(text_raw)

        breakdown: Dict[str, DomainRawBreakdown] = {}

        for domain in DOMAIN_NAMES:
            defn = DOMAIN_REGISTRY[domain]
            sem = clamp01(semantic_by_domain[domain])
            kw, kw_matched = self._keyword_matches(text_canon, defn)
            pat = self._pattern_score(text_raw, domain)
            intent, _ = self._intent_matches(text_canon, defn)
            phr, phr_matched = self._phrase_matches(text_canon, defn)
            r_cos = float(raw_cosine_by_domain[domain])

            raw = (
                w.semantic * sem
                + w.keyword * kw
                + w.pattern * pat
                + w.intent * intent
                + w.phrase * phr
            )

            breakdown[domain] = DomainRawBreakdown(
                semantic_score=sem,
                keyword_score=kw,
                pattern_score=pat,
                intent_score=intent,
                phrase_score=phr,
                raw_score=clamp01(raw),
                raw_cosine_similarity=r_cos,
                matched_keywords=list(kw_matched),
                matched_phrases=list(phr_matched),
            )

        return breakdown
