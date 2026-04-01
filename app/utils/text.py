"""Text normalization helpers for keyword and intent matching."""

from __future__ import annotations

import re


def normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for robust substring matching."""
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def word_boundary_contains(haystack: str, needle: str) -> bool:
    """
    Return True if needle appears as a substring after normalization.
    Uses word boundaries when needle is alphanumeric to reduce false positives.
    """
    h = normalize_for_match(haystack)
    n = needle.lower().strip()
    if not n:
        return False
    if re.fullmatch(r"[a-z0-9]+", n):
        return re.search(rf"\b{re.escape(n)}\b", h) is not None
    return n in h
