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
from app.utils.math_utils import cosine_similarity, cosine_to_unit_interval, normalize_l2

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

    def semantic_scores_unit_interval(self, prompt_vec: NDArray[np.floating]) -> Dict[str, float]:
        """
        Cosine similarity between prompt and each prototype, mapped to [0,1].
        """
        scores: Dict[str, float] = {}
        for domain in DOMAIN_NAMES:
            proto = self._prototypes[domain]
            cos = cosine_similarity(prompt_vec, proto)
            scores[domain] = cosine_to_unit_interval(cos)
        return scores

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
