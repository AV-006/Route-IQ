"""
Embedding model lifecycle: load once, precompute domain prototype vectors.

Uses sentence-transformers with BAAI/bge-small-en-v1.5 and L2-normalized prototypes.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.constants import DOMAIN_NAMES
from app.intelligence.domain_configs import DOMAIN_REGISTRY
from app.utils.math_utils import clamp01, cosine_similarity, normalize_l2

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Owns the SentenceTransformer model and per-domain prototype embeddings.

    Prototypes are the L2-normalized mean of anchor embeddings per domain.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._model: SentenceTransformer | None = None
        self._prototypes: Dict[str, NDArray[np.floating]] = {}
        self._initialized = False

    @property
    def model_name(self) -> str:
        return settings.embedding_model_name

    def initialize(self) -> None:
        """Load model and compute prototypes exactly once."""
        with self._lock:
            if self._initialized:
                return
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._prototypes = self._compute_prototypes()
            self._initialized = True
            logger.info("Embedding prototypes ready for %d domains", len(self._prototypes))

    def _compute_prototypes(self) -> Dict[str, NDArray[np.floating]]:
        assert self._model is not None
        prototypes: Dict[str, NDArray[np.floating]] = {}
        for domain in DOMAIN_NAMES:
            anchors: List[str] = DOMAIN_REGISTRY[domain].anchors
            vectors = self._model.encode(
                anchors,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            mean_vec = np.mean(vectors, axis=0).astype(np.float64)
            prototypes[domain] = normalize_l2(mean_vec)
        return prototypes

    def encode_prompt(self, prompt: str) -> NDArray[np.floating]:
        """Encode a single prompt; L2-normalize for cosine-as-dot-product."""
        if not self._initialized or self._model is None:
            raise RuntimeError("EmbeddingService not initialized; call initialize() at startup")
        vec = self._model.encode(
            [prompt],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )[0].astype(np.float64)
        return normalize_l2(vec)

    def raw_cosine_similarities(self, prompt_vec: NDArray[np.floating]) -> Dict[str, float]:
        """Pairwise cosine similarity in native [-1, 1] range (L2-normalized vectors)."""
        return {
            domain: cosine_similarity(prompt_vec, self._prototypes[domain])
            for domain in DOMAIN_NAMES
        }

    def relative_semantic_scores(self, prompt_vec: NDArray[np.floating]) -> Dict[str, float]:
        """
        Within-prompt relative semantic scores for hybrid routing.

        Raw cosines are min-max normalized per prompt, then sharpened:

            relative = (sim - min_sim) / (max_sim - min_sim + eps)
            score = clamp01(relative ** sharpen_exponent)

        This spreads mass across domains for a single prompt instead of flattening
        all instruction-like texts toward similar raw cosine values.
        """
        raw = self.raw_cosine_similarities(prompt_vec)
        sims = [raw[d] for d in DOMAIN_NAMES]
        min_s = min(sims)
        max_s = max(sims)
        eps = 1e-8
        exp = settings.semantic_relative_sharpen_exponent
        scores: Dict[str, float] = {}
        for domain in DOMAIN_NAMES:
            sim = raw[domain]
            relative = (sim - min_s) / (max_s - min_s + eps)
            scores[domain] = clamp01(relative**exp)
        return scores

    def semantic_scores_unit_interval(self, prompt_vec: NDArray[np.floating]) -> Dict[str, float]:
        """
        Back-compat alias: returns the same relative sharpened scores as
        :meth:`relative_semantic_scores` (not raw cosine mapped to [0,1]).
        """
        return self.relative_semantic_scores(prompt_vec)

    @property
    def prototypes(self) -> Dict[str, NDArray[np.floating]]:
        if not self._initialized:
            raise RuntimeError("EmbeddingService not initialized")
        return self._prototypes


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Global embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
