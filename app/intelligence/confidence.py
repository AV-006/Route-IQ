"""
Confidence estimate from the final domain distribution.

Higher confidence when the distribution is peaky (low entropy, large top1-top2 gap).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from app.core.constants import DOMAIN_NAMES
from app.utils.math_utils import distribution_entropy, max_entropy_uniform


def sorted_domain_scores(domain_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """Return (domain, score) pairs sorted descending by score."""
    return sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)


def compute_confidence(domain_scores: Dict[str, float]) -> float:
    """
    Combine normalized entropy margin and top-1 vs top-2 separation.

    - Entropy near log(n) (uniform) yields low contribution.
    - Entropy near 0 (single spike) yields high contribution.
    - Larger gap between best and second-best domain increases confidence.

    Returns a value in [0, 1].
    """
    n = len(DOMAIN_NAMES)
    if n == 0:
        return 0.0

    # Ensure we only use supported domains in order
    probs = {k: float(domain_scores.get(k, 0.0)) for k in DOMAIN_NAMES}
    total = sum(probs.values())
    if total <= 0.0:
        return 0.0
    probs = {k: v / total for k, v in probs.items()}

    h = distribution_entropy(probs)
    h_max = max_entropy_uniform(n)
    if h_max <= 0.0:
        entropy_component = 1.0
    else:
        # Low entropy => high component
        entropy_component = 1.0 - (h / h_max)
        entropy_component = max(0.0, min(1.0, entropy_component))

    ranked = sorted_domain_scores(probs)
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = top1 - top2
    # Gap of ~0.35+ should feel "confident" for 8-way distribution
    gap_component = max(0.0, min(1.0, gap / 0.35))

    # Weight entropy more than gap to penalize ambiguous flat distributions
    combined = 0.55 * entropy_component + 0.45 * gap_component

    # Mild boost if top share is very dominant
    dominance = top1**0.5  # sqrt softens extremes
    combined = 0.85 * combined + 0.15 * dominance

    return max(0.0, min(1.0, float(combined)))


def top_k_domains(domain_scores: Dict[str, float], k: int = 3) -> List[str]:
    """Return top-k domain names by descending score."""
    ranked = sorted_domain_scores(domain_scores)
    return [name for name, _ in ranked[:k]]
