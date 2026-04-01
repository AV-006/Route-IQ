"""
Heuristic task parsing for prompt complexity.

We avoid heavy NLP: the goal is to detect *multi-task* prompts and provide
explainable evidence (e.g., bullet lists, numbered steps, "and then", etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.utils.math_utils import clamp01


@dataclass(frozen=True)
class TaskSignal:
    estimated_task_count: int
    score: float
    evidence: List[str]


_BULLET_RE = re.compile(r"(^|\n)\s*([-*•]|(\d+[\).\]]))\s+", re.MULTILINE)
_SEQUENCER_RE = re.compile(
    r"\b(and then|then|also|as well as|in addition|after that|next|finally)\b",
    re.IGNORECASE,
)
_MULTI_VERB_RE = re.compile(
    r"\b(explain|justify|compare|analyze|derive|prove|solve|summarize|extract|rewrite|refactor|debug|optimize|implement|design|build)\b",
    re.IGNORECASE,
)


def detect_multi_task(prompt: str) -> TaskSignal:
    if not prompt.strip():
        return TaskSignal(estimated_task_count=0, score=0.0, evidence=[])

    evidence: List[str] = []

    bullets = list(_BULLET_RE.finditer(prompt))
    if bullets:
        evidence.append(f"bullets_or_numbered_items={len(bullets)}")

    sequencers = list(_SEQUENCER_RE.finditer(prompt))
    if sequencers:
        evidence.append(f"sequencers={len(sequencers)}")

    verbs = list(_MULTI_VERB_RE.finditer(prompt))
    distinct_verbs = sorted({m.group(0).lower() for m in verbs})
    if distinct_verbs:
        evidence.append(f"distinct_task_verbs={len(distinct_verbs)}:{','.join(distinct_verbs[:8])}")

    # Estimate tasks:
    # - If bullets exist, treat each bullet/numbered line as a task.
    # - Else approximate from sequencers and distinct verb variety.
    #
    # Important: many real prompts are multi-task via comma-separated verb lists
    # ("summarize..., extract..., and rewrite...") without explicit sequencers.
    if bullets:
        tasks = min(12, len(bullets))
    else:
        tasks = 1

        # Sequencers strongly imply multiple steps.
        tasks += min(4, len(sequencers) // 2)  # multiple sequencers to count as extra tasks

        # Verb diversity: treat distinct task verbs as deliverables, not just "tone".
        # 2 verbs -> likely 2 tasks, 3 verbs -> 3 tasks, 5+ -> 4+ tasks.
        if len(distinct_verbs) >= 2:
            tasks = max(tasks, 2)
        if len(distinct_verbs) >= 3:
            tasks = max(tasks, 3)
        if len(distinct_verbs) >= 5:
            tasks = max(tasks, 4)

        tasks = min(6, tasks)

    tasks = max(1, tasks)

    # Score: 1 task => 0.0, 2 => ~0.33, 3 => ~0.66, >=4 => 1.0
    score = clamp01((tasks - 1) / 3.0)
    return TaskSignal(estimated_task_count=tasks, score=score, evidence=evidence)

