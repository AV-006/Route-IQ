"""
Sample prompts for manual QA, demos, and mixed-domain behavior checks.

Run from repository root (after dependencies and optional model cache):

    python -m app.tests.sample_prompts

This loads the embedding model once and prints scores for all samples.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Allow running as script: repo root on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass(frozen=True)
class Sample:
    label: str
    prompt: str
    category: str


SAMPLES: List[Sample] = [
    Sample("complexity_low_factual", "What is photosynthesis?", "complexity_low"),
    Sample(
        "complexity_medium_single_task_constraints",
        "Explain what polymorphism means in OOP and give one short example. Keep it under 120 words.",
        "complexity_medium",
    ),
    Sample(
        "complexity_high_multitask",
        "Compare BFS vs DFS, explain why one should be preferred for shortest paths, analyze edge cases, "
        "then implement BFS in Python and return only valid JSON with keys: explanation, code, complexity.",
        "complexity_high",
    ),
    Sample("coding_linked_list", "Write a Python function to detect a cycle in a linked list", "coding"),
    Sample("math_integral", "Solve this integral step by step: int x*exp(-x^2) dx", "math"),
    Sample("summarize_paper", "Summarize this research paper in 5 bullet points", "summarization"),
    Sample("extract_entities", "Extract all company names from this paragraph", "extraction"),
    Sample("rewrite_email", "Rewrite this email in a professional tone", "transformation"),
    Sample("compare_bfs_dfs", "Compare BFS and DFS with examples", "reasoning+coding"),
    Sample("reasoning_theorem_apply", "Explain why this theorem applies in this situation and what changes if the condition is weakened", "reasoning"),
    Sample("reasoning_hold_if_change", "Would this still hold if the assumption changed? Analyze edge cases and justify your answer.", "reasoning"),
    Sample("reasoning_prefer_bfs", "When should BFS be preferred over DFS, and why does that choice apply here?", "reasoning+coding"),
    Sample("noncoding_detect_generic", "Detect anomalies in quarterly revenue and explain likely causes", "reasoning"),
    Sample("horror_story", "Write a short horror story set in a hospital", "creative_writing"),
    Sample("polymorphism_fact", "Explain what polymorphism means in OOP", "factual_qa"),
    Sample("mixed_math_cpp", "Solve this recurrence relation T(n)=2T(n/2)+n and implement the iterative bottom-up solution in C++", "mixed"),
    Sample("mixed_summarize_extract", "Summarize this earnings call and extract guidance numbers as a small table", "mixed"),
    Sample("mixed_compare_code", "Compare microservices vs modular monolith with a short code-free checklist and one Python pseudocode sketch for feature flags", "mixed"),
    Sample("mixed_proof_impl", "Prove sqrt(5) is irrational and then write a Python snippet that approximates sqrt(5) using continued fractions", "mixed"),
    Sample("mixed_style_facts", "Rewrite this paragraph in simpler language and explain any jargon terms you simplify", "mixed"),
    Sample("ambiguous_help", "Help me with this thing", "ambiguous"),
    Sample("ambiguous_do_it", "Do it better", "ambiguous"),
    Sample("ambiguous_stuff", "stuff", "ambiguous"),
    Sample("weak_hi", "Hi", "weak_short"),
    Sample("weak_ok", "ok thanks", "weak_short"),
    Sample("weak_fix", "fix pls", "weak_short"),
    Sample("reasoning_tradeoff", "Justify picking eventual consistency over strong consistency for a high-write social feed", "reasoning"),
    Sample("factual_capital", "What is photosynthesis and what are its inputs and outputs?", "factual_qa"),
    Sample("transformation_yaml", "Convert this JSON into YAML with brief comments on each key", "transformation"),
    Sample("extraction_invoice", "Pull line items from this invoice text into JSON with sku, qty, price", "extraction"),
    Sample("summarize_tldr", "TL;DR this wall of text into three sentences for executives", "summarization"),
    Sample("coding_sql", "Optimize this slow PostgreSQL query that filters on a JSONB field", "coding"),
]


def run_demo() -> None:
    """Initialize embedder and print analysis for each sample."""
    from app.intelligence.analyzer import PromptDomainAnalyzer
    from app.intelligence.embeddings import get_embedding_service

    print("Loading model (first run may download weights)...", flush=True)
    svc = get_embedding_service()
    svc.initialize()
    analyzer = PromptDomainAnalyzer(svc)

    for s in SAMPLES:
        result = analyzer.analyze(s.prompt)
        print("\n" + "=" * 72)
        print(f"[{s.category}] {s.label}")
        print(s.prompt[:120] + ("..." if len(s.prompt) > 120 else ""))
        print(f"top3: {result.top_domains}  confidence: {result.confidence}")
        print(json.dumps(result.domain_scores, indent=2))


if __name__ == "__main__":
    run_demo()
