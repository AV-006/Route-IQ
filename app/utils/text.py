"""Text normalization helpers for keyword and intent matching."""

from __future__ import annotations

import re

# Common spelling / alias normalization so keywords and phrases match reliably.
_LINKED_LIST_RE = re.compile(r"\blinkedlist\b", re.IGNORECASE)
_CPP_TOKEN_RE = re.compile(r"\bc\+\+\b", re.IGNORECASE)
_OOP_SPACING_RE = re.compile(r"\bobject\s*oriented\b", re.IGNORECASE)


def normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for robust substring matching."""
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def canonicalize_prompt_for_matching(text: str) -> str:
    """
    Normalize for lexical scoring: whitespace, lowercase, and light alias expansion.

    Improves hits for e.g. linkedlist vs linked list, C++ vs cpp, object-oriented forms.
    """
    h = normalize_for_match(text)
    h = _LINKED_LIST_RE.sub("linked list", h)
    h = _CPP_TOKEN_RE.sub(" cpp ", h)
    h = re.sub(r"\bcpp\b", " cpp ", h, flags=re.IGNORECASE)
    h = _OOP_SPACING_RE.sub("object oriented", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


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
