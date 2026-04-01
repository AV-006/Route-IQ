"""Prompt-level text features for debugging and downstream routing."""

from __future__ import annotations

import re
from typing import Dict

from app.models.schemas import TextFeatures


_CODE_SYMBOL_RE = re.compile(r"[{}()\[\];=<>\|`~]")


def extract_text_features(prompt: str) -> TextFeatures:
    """
    Compute lightweight lexical statistics from the raw prompt.

    token_count uses whitespace splitting (approximate word tokens).
    """
    if not prompt:
        return TextFeatures(
            token_count=0,
            avg_word_length=0.0,
            special_char_ratio=0.0,
            code_symbol_ratio=0.0,
            digit_ratio=0.0,
            uppercase_ratio=0.0,
            newline_count=0,
        )

    tokens = prompt.split()
    n_chars = len(prompt)
    n_tokens = len(tokens)

    if n_tokens == 0:
        avg_word_len = 0.0
    else:
        avg_word_len = sum(len(t) for t in tokens) / n_tokens

    special_chars = sum(1 for c in prompt if not c.isalnum() and not c.isspace())
    denom_vis = max(1, n_chars)
    special_char_ratio = min(1.0, special_chars / denom_vis)

    code_syms = len(_CODE_SYMBOL_RE.findall(prompt))
    code_symbol_ratio = min(1.0, code_syms / denom_vis)

    digits = sum(1 for c in prompt if c.isdigit())
    digit_ratio = min(1.0, digits / denom_vis)

    uppercase = sum(1 for c in prompt if c.isupper())
    uppercase_ratio = min(1.0, uppercase / denom_vis)

    newline_count = prompt.count("\n")

    return TextFeatures(
        token_count=n_tokens,
        avg_word_length=round(avg_word_len, 4),
        special_char_ratio=round(special_char_ratio, 4),
        code_symbol_ratio=round(code_symbol_ratio, 4),
        digit_ratio=round(digit_ratio, 4),
        uppercase_ratio=round(uppercase_ratio, 4),
        newline_count=newline_count,
    )


def text_features_to_public_dict(features: TextFeatures) -> Dict[str, object]:
    """Serialize features for JSON-friendly responses."""
    return features.model_dump()
