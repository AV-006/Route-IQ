"""Numeric helpers: cosine similarity, entropy, distribution normalization."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine similarity between two 1-D vectors. Assumes non-zero norms."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_l2(vec: NDArray[np.floating]) -> NDArray[np.floating]:
    """L2-normalize a vector; return zero vector if norm is zero."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def distribution_entropy(probs: Mapping[str, float]) -> float:
    """
    Shannon entropy (natural log) for a discrete distribution.
    Ignores keys with zero probability.
    """
    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= float(p) * math.log(p)
    return h


def max_entropy_uniform(n_categories: int) -> float:
    """Entropy of uniform distribution over n categories."""
    if n_categories <= 1:
        return 0.0
    p = 1.0 / n_categories
    return -float(n_categories) * p * math.log(p)


def clamp01(x: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, x))


def normalize_nonneg_sum_to_one(scores: dict[str, float]) -> dict[str, float]:
    """
    Clamp negatives to zero; if sum is zero return empty dict for caller to handle.
    Otherwise return probabilities summing to 1.0.
    """
    clipped = {k: max(0.0, v) for k, v in scores.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in clipped.items()}


def uniform_distribution(keys: list[str]) -> dict[str, float]:
    """Equal weight over all keys."""
    if not keys:
        return {}
    p = 1.0 / len(keys)
    return {k: p for k in keys}


def cosine_to_unit_interval(cos_sim: float) -> float:
    """
    Map cosine similarity from [-1, 1] to [0, 1] for downstream hybrid scoring.
    """
    return clamp01((float(cos_sim) + 1.0) / 2.0)
