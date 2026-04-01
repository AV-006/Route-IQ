"""
Confidence estimate from the final domain distribution.

Higher confidence when the distribution is peaky (low entropy, large top1-top2 gap)
and when the dominant domain clearly leads. Very short or weak prompts are damped.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from app.core.constants import DOMAIN_NAMES
from app.utils.math_utils import distribution_entropy, max_entropy_uniform


def sorted_domain_scores(domain_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """Return (domain, score) pairs sorted descending by score."""
    return sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)


def compute_confidence(
    domain_scores: Dict[str, float],
    token_count: int | None = None,
) -> float:
    """
    Combine entropy margin, top-1 vs top-2 separation, and leader share.

    - Entropy near log(n) (uniform) yields low contribution.
    - Entropy near 0 (single spike) yields high contribution.
    - Larger gap between best and second-best domain increases confidence.
    - Strong top-1 share and top-1+top2 mass increase confidence for clear routing.
    - Very short prompts (e.g. \"hi\") are intentionally damped.

    Returns a value in [0, 1].
    """
    n = len(DOMAIN_NAMES)
    if n == 0:
        return 0.0

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
        entropy_component = 1.0 - (h / h_max)
        entropy_component = max(0.0, min(1.0, entropy_component))

    ranked = sorted_domain_scores(probs)
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    top3 = ranked[2][1] if len(ranked) > 2 else 0.0
    gap = top1 - top2
    # ~0.22+ gap on a sharpened distribution should read as confident
    gap_component = max(0.0, min(1.0, gap / 0.22))

    # Leader share: rewards a single clear winner
    leader_component = max(0.0, min(1.0, (top1 - 1.0 / n) / (1.0 - 1.0 / n)))

    # Top-2 mass: mixed-but-meaningful prompts (e.g. coding + reasoning)
    top2_mass = top1 + top2
    ceiling = min(1.0, 2.5 / n * 2)  # ~0.625 for n=8
    concentration_component = max(0.0, min(1.0, (top2_mass - 2.0 / n) / (ceiling - 2.0 / n + 1e-8)))

    uniform = 1.0 / n
    margin_vs_uniform = max(0.0, top1 - uniform) / (1.0 - uniform + 1e-8)
    margin_component = max(0.0, min(1.0, margin_vs_uniform))

    combined = (
        0.28 * entropy_component
        + 0.30 * gap_component
        + 0.18 * leader_component
        + 0.12 * concentration_component
        + 0.12 * margin_component
    )

    # Penalize three-way ties: if third place is close to second, reduce slightly
    if top2 > 1e-9:
        runner_spread = (top2 - top3) / top2
        if runner_spread < 0.25:
            combined *= 0.92

    weak_signal = 1.0
    if token_count is not None:
        if token_count <= 1:
            weak_signal = 0.22
        elif token_count <= 2:
            weak_signal = 0.48
        elif token_count <= 3:
            weak_signal = 0.78

    if top1 < uniform + 0.04:
        weak_signal *= 0.75

    combined = max(0.0, min(1.0, combined * weak_signal))
    return float(combined)


def top_k_domains(domain_scores: Dict[str, float], k: int = 3) -> List[str]:
    """Return top-k domain names by descending score."""
    ranked = sorted_domain_scores(domain_scores)
    return [name for name, _ in ranked[:k]]
