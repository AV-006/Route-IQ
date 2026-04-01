"""
Hybrid domain scoring: semantic + keyword + regex pattern + intent verbs.

Semantic scores are supplied by EmbeddingService; this module combines components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from app.core.config import settings
from app.core.constants import DOMAIN_NAMES
from app.intelligence.domain_configs import DOMAIN_REGISTRY, DomainDefinition
from app.utils.math_utils import clamp01
from app.utils.text import word_boundary_contains


@dataclass(frozen=True)
class DomainRawBreakdown:
    """Intermediate scores for one domain before global normalization."""

    semantic_score: float
    keyword_score: float
    pattern_score: float
    intent_score: float
    raw_score: float


class HybridDomainScorer:
    """
    Computes keyword, pattern, and intent subscores per domain and blends with semantic.

    Uses configurable weights from app.core.config.settings.scoring_weights.
    """

    def __init__(self) -> None:
        self._compiled_patterns: Dict[str, List[re.Pattern[str]]] = {}
        for name in DOMAIN_NAMES:
            pats = DOMAIN_REGISTRY[name].patterns
            self._compiled_patterns[name] = [
                re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats
            ]

    def _keyword_score(self, text: str, definition: DomainDefinition) -> float:
        """Fraction of keyword hits up to a cap, mapped to [0,1]."""
        hits = 0
        for kw in definition.keywords:
            if word_boundary_contains(text, kw):
                hits += 1
        cap = settings.keyword_match_cap
        return clamp01(hits / float(max(1, cap)))

    def _pattern_score(self, text: str, domain: str) -> float:
        """Count regex hits; saturate after pattern_score_max_hits."""
        compiled = self._compiled_patterns[domain]
        matches = 0
        for rx in compiled:
            if rx.search(text):
                matches += 1
        max_hits = settings.pattern_score_max_hits
        return clamp01(matches / float(max(1, max_hits)))

    def _intent_score(self, text: str, definition: DomainDefinition) -> float:
        """Detect domain action verbs; saturate quickly to avoid dominating."""
        hits = 0
        for verb in definition.intent_verbs:
            if word_boundary_contains(text, verb):
                hits += 1
        return clamp01(hits / 3.0)

    def compute_breakdown(
        self,
        prompt: str,
        semantic_by_domain: Dict[str, float],
    ) -> Dict[str, DomainRawBreakdown]:
        """
        Build per-domain breakdown and weighted raw_score (pre-global-normalize).

        semantic_by_domain must be in [0,1] per domain.
        """
        w = settings.scoring_weights
        breakdown: Dict[str, DomainRawBreakdown] = {}

        for domain in DOMAIN_NAMES:
            defn = DOMAIN_REGISTRY[domain]
            sem = clamp01(semantic_by_domain[domain])
            kw = self._keyword_score(prompt, defn)
            pat = self._pattern_score(prompt, domain)
            intent = self._intent_score(prompt, defn)

            raw = (
                w.semantic * sem
                + w.keyword * kw
                + w.pattern * pat
                + w.intent * intent
            )

            breakdown[domain] = DomainRawBreakdown(
                semantic_score=sem,
                keyword_score=kw,
                pattern_score=pat,
                intent_score=intent,
                raw_score=clamp01(raw),
            )

        return breakdown
