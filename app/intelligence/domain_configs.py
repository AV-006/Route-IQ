"""
Configuration-driven domain definitions: anchors, keywords, regex patterns, intent verbs.

Anchors are embedded and averaged into per-domain prototype vectors at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.core.constants import DOMAIN_NAMES


@dataclass(frozen=True)
class DomainDefinition:
    """Single domain configuration."""

    name: str
    anchors: List[str]
    keywords: List[str]
    patterns: List[str]
    intent_verbs: List[str]


def _domains() -> Dict[str, DomainDefinition]:
    return {
        "coding": DomainDefinition(
            name="coding",
            anchors=[
                "Refactor this Flask route handler to use async SQLAlchemy sessions "
                "and add proper error handling for duplicate keys.",
                "My React useEffect is firing twice in Strict Mode and leaking listeners; "
                "show a minimal fix using cleanup functions.",
                "Write a Bash script that tails a log file, parses JSON lines, and alerts "
                "when error_rate exceeds 5% over a 60-second window.",
                "Optimize this PostgreSQL query that filters on a JSONB field; explain "
                "which index type you would add and why.",
                "Implement a thread-safe bounded queue in Java without using java.util.concurrent.",
                "Debug this segfault in a C++ smart-pointer refactor; the crash happens "
                "when the temporary shared_ptr goes out of scope.",
                "Design a REST endpoint spec for idempotent order cancellation with "
                "ETags and conflict responses.",
                "Explain how Kosaraju’s algorithm finds strongly connected components and "
                "give pseudocode with time complexity.",
                "Convert this nested promise chain in Node.js to async/await with "
                "consistent timeout and cancellation.",
                "Sketch a simple CI job in GitHub Actions that caches pip, runs pytest "
                "with coverage gates, and uploads HTML reports.",
                "This pandas groupby is producing unexpected NaNs after merging on "
                "timestamps with timezone offsets—what’s the robust fix?",
                "Write a unit test suite for a payment webhook handler that deduplicates "
                "events using event_id and handles retries safely.",
                "Profile why this Python dict lookup loop is slow over a 10M-row CSV "
                "and propose a faster data structure layout.",
                "Create an OpenAPI snippet for PATCH /users/{id} with partial updates "
                "and validation error schema.",
            ],
            keywords=[
                "code",
                "implement",
                "function",
                "class",
                "api",
                "rest",
                "graphql",
                "sql",
                "query",
                "database",
                "postgres",
                "mysql",
                "mongodb",
                "typescript",
                "javascript",
                "python",
                "java",
                "cpp",
                "c++",
                "rust",
                "go",
                "bash",
                "script",
                "docker",
                "kubernetes",
                "github",
                "unittest",
                "pytest",
                "jest",
                "refactor",
                "debug",
                "stack trace",
                "compiler",
                "runtime",
                "oauth",
                "jwt",
                "websocket",
                "async",
                "promise",
                "callback",
                "pandas",
                "numpy",
                "tensorflow",
                "pytorch",
            ],
            patterns=[
                r"\bdef\s+\w+\s*\(",
                r"\b(class|interface|enum)\s+\w+",
                r"\b(import|from)\s+\w+",
                r"\{[\s\S]*\}",
                r"```",
                r"\bSELECT\b",
                r"\bINSERT\b",
                r"\bUPDATE\b",
                r"https?://",
                r"\.tsx?\b",
                r"\.py\b",
                r"\bgit\b",
                r"\bpytest\b",
                r"\bnpm\b",
            ],
            intent_verbs=[
                "implement",
                "write",
                "code",
                "debug",
                "fix",
                "refactor",
                "optimize",
                "profile",
                "deploy",
                "containerize",
                "test",
                "unit test",
            ],
        ),
        "math": DomainDefinition(
            name="math",
            anchors=[
                "Evaluate the definite integral from 0 to pi of x*sin(x) dx using "
                "integration by parts.",
                "Solve this system of linear equations over the reals and state whether "
                "the solution is unique.",
                "A fair die is rolled until the sum exceeds 20; find the expected number "
                "of rolls.",
                "Prove by induction that 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2 for all "
                "positive integers n.",
                "Derive the quadratic formula from ax^2 + bx + c = 0 by completing the square.",
                "Find all real solutions to sin(2x) = cos(x) on the interval [0, 2pi].",
                "How many onto functions exist from a 5-element set to a 3-element set?",
                "Minimize x^2 + y^2 subject to x + y = 1 using Lagrange multipliers.",
                "Compute the eigenvalues of the 2x2 matrix [[2,1],[1,2]] and diagonalize it.",
                "A train travels between two cities with a steady headwind; set up the "
                "equation if the return trip speed differs by v and total time is T.",
                "Show that sqrt(2) is irrational using a proof by contradiction.",
                "Approximate ln(1.1) using a second-order Taylor polynomial around x=0 "
                "and bound the error.",
                "In how many ways can you tile a 2×n board with dominoes? Give a recurrence.",
                "Find the area between y = x^2 and y = 2x - x^2 enclosed by their intersections.",
            ],
            keywords=[
                "integral",
                "derivative",
                "limit",
                "calculus",
                "algebra",
                "equation",
                "matrix",
                "eigenvalue",
                "probability",
                "expectation",
                "variance",
                "proof",
                "induction",
                "contradiction",
                "theorem",
                "lemma",
                "trigonometry",
                "sin",
                "cos",
                "tan",
                "recurrence",
                "combinatorics",
                "permutation",
                "binomial",
                "taylor",
                "series",
                "optimization",
                "lagrange",
                "graph theory",
                "gcd",
                "lcm",
                "mod",
                "modulo",
            ],
            patterns=[
                r"\b∫|integral\b",
                r"\b\d+\s*x\^?\d*\b",
                r"\$\$[\s\S]+?\$\$",
                r"\$[^$]+\$",
                r"\\frac\{",
                r"\\int",
                r"\beigenvalue",
                r"\bprove\b",
                r"by induction",
                r"\bP\(|E\[",
            ],
            intent_verbs=[
                "solve",
                "compute",
                "evaluate",
                "derive",
                "prove",
                "simplify",
                "integrate",
                "differentiate",
                "minimize",
                "maximize",
            ],
        ),
        "reasoning": DomainDefinition(
            name="reasoning",
            anchors=[
                "Compare microservices versus a modular monolith for a team of eight "
                "engineers shipping weekly; what breaks first at scale?",
                "Explain why eventual consistency can surface duplicate charges in a "
                "payment system and how you would detect and reconcile them.",
                "What tradeoffs do you make when choosing between breadth-first search "
                "and depth-first search for large sparse graphs?",
                "A mobile app feels laggy only on Android low-memory devices; outline "
                "a debugging strategy and likely root causes.",
                "Justify whether you would prioritize latency or throughput for a "
                "real-time leaderboard with bursty traffic.",
                "Analyze edge cases when using floating-point equality in financial "
                "rounding versus decimal arithmetic.",
                "Given two cache eviction policies, which reduces tail latency for skewed "
                "key access patterns and why?",
                "How would you decide between synchronous vs asynchronous replication "
                "for a regional outage scenario?",
                "Walk through the failure modes if a leader election service flakes "
                "during a network partition.",
                "Contrast optimistic vs pessimistic concurrency for checkout carts "
                "when inventory is contested.",
                "When does adding more indexes hurt an OLTP workload and how would you measure it?",
                "Explain why naive sharding by user_id hash can create hot shards and "
                "how to mitigate without perfect foresight.",
                "What are the ethical risks if a hiring classifier is trained on historical "
                "promotion data; how would you audit it?",
                "Compare zero-trust network segmentation vs VPN-only access for contractors.",
            ],
            keywords=[
                "compare",
                "contrast",
                "tradeoff",
                "trade-off",
                "justify",
                "why",
                "because",
                "edge case",
                "failure mode",
                "risk",
                "scenario",
                "strategy",
                "analysis",
                "evaluate options",
                "decision",
                "architecture",
                "scalability",
                "reliability",
                "latency",
                "throughput",
                "consistency",
                "availability",
                "partition",
                "reasoning",
                "assumption",
                "implication",
            ],
            patterns=[
                r"\bcompare\b.*\b(and|vs\.?|versus)\b",
                r"\btrade-?offs?\b",
                r"\bwhat (if|happens when)\b",
                r"\bpros and cons\b",
                r"\bjustify\b",
                r"\bexplain why\b",
                r"\bedge cases?\b",
                r"\bfailure modes?\b",
                r"\bwhich (is )?better\b",
            ],
            intent_verbs=[
                "compare",
                "contrast",
                "analyze",
                "justify",
                "evaluate",
                "assess",
                "reason",
                "debate",
            ],
        ),
        "summarization": DomainDefinition(
            name="summarization",
            anchors=[
                "Summarize this 2,000-word article into five crisp bullet points aimed "
                "at executives with no domain jargon.",
                "Give a tight TL;DR of the attached earnings call transcript highlighting "
                "guidance changes only.",
                "Condense this literature review into a one-page memo while preserving "
                "citation placeholders.",
                "Produce a structured recap of this conference talk: main claim, three "
                "supporting ideas, and one caveat.",
                "I have messy lecture notes on operating systems; compress them into an "
                "outline suitable for exam review.",
                "Summarize the methods and results sections of this paper without quoting "
                "long sentences verbatim.",
                "Turn this long Slack thread into a short decision log: decision, owners, "
                "open questions.",
                "Provide an executive summary of this policy document with risks called out.",
                "Boil this quarterly strategy deck down to three takeaways and one metric to watch.",
                "Give me a neutral summary of both sides of this debate with no loaded language.",
                "Reduce this technical RFC to a five-sentence overview a PM can skim.",
                "Summarize the plot of this novel for a reading group, spoiler-free until "
                "the halfway mark.",
                "Compress this multi-day workshop agenda into time-boxed themes only.",
                "Create an abstractive summary of the patient handoff note focusing on "
                "action items (fictional scenario).",
            ],
            keywords=[
                "summarize",
                "summary",
                "tl;dr",
                "condense",
                "compress",
                "abstract",
                "recap",
                "overview",
                "high-level",
                "executive summary",
                "in short",
                "bullet points",
                "key points",
                "shorter",
                "briefly",
                "synopsis",
                "digest",
                "outline",
                "minutes",
                "highlights",
            ],
            patterns=[
                r"\bsummar(y|ize)\b",
                r"\btl;?dr\b",
                r"\bin (a few|two|three) sentences\b",
                r"\bexecutive summary\b",
                r"\bkey takeaways\b",
                r"\bhigh-?level overview\b",
            ],
            intent_verbs=[
                "summarize",
                "condense",
                "compress",
                "recap",
                "abbreviate",
            ],
        ),
        "extraction": DomainDefinition(
            name="extraction",
            anchors=[
                "Extract all invoice line items as JSON with keys sku, qty, unit_price_cents, tax_code.",
                "Pull out every date mentioned in this email thread and normalize them to ISO-8601.",
                "From this messy paragraph, list each company name and its stock ticker if given.",
                "Identify all medications, dosages, and frequencies in this discharge summary (example text).",
                "Parse this résumé into structured fields: education[], experience[], skills[].",
                "What are the three main experimental findings stated explicitly in this abstract?",
                "Create a table of all URLs and their anchor text from this HTML snippet.",
                "Extract named entities (people, organizations, locations) from this news clip.",
                "Pull bullet-ready action items from this meeting transcript with an owner if stated.",
                "Given this CSV snippet, return only rows where status equals overdue and sum amount.",
                "List every cited paper title and year from this bibliography block.",
                "Extract constraints from this project brief: budget, deadline, must-haves, out-of-scope.",
                "From this policy, extract penalties in a machine-readable list of (clause, consequence).",
                "Return the cap table implied by this paragraph: founder shares, investor tranche, pool.",
            ],
            keywords=[
                "extract",
                "pull out",
                "parse",
                "entities",
                "fields",
                "json",
                "csv",
                "table",
                "schema",
                "key findings",
                "list all",
                "enumerate",
                "named entity",
                "regex",
                "capture",
                "structured",
                "annotation",
                "slot filling",
                "information extraction",
            ],
            patterns=[
                r"\bextract\b",
                r"\bpull out\b",
                r"\b(?:as|into) (json|csv|yaml)\b",
                r"\bkey findings\b",
                r"\bentity\b",
                r"\bnamed entities\b",
                r"\bline items\b",
            ],
            intent_verbs=[
                "extract",
                "parse",
                "pull",
                "identify",
                "enumerate",
                "list",
            ],
        ),
        "creative_writing": DomainDefinition(
            name="creative_writing",
            anchors=[
                "Write a 600-word noir short story about a museum guard who hears "
                "footsteps after closing, ending on an ambiguous reveal.",
                "Draft a sonnet about migratory birds using slant rhyme and volta in line 9.",
                "Compose dialogue between two siblings negotiating who keeps a haunted heirloom.",
                "World-build three factions on a tidally locked planet with conflicting "
                "resource ethics—no combat scenes yet.",
                "Turn this logline into a three-act beat sheet for a contained thriller.",
                "Write a cold-open scene for a sitcom pilot set in a chaotic community "
                "garden with at least one running gag.",
                "Pen lyrics for a folk song about winter train travel in iambic verse.",
                "Describe a surreal marketplace where memories are bartered; focus on "
                "sensory detail, not plot.",
                "Create character voice sheets for an optimistic engineer and a cynical "
                "botanist on an expedition.",
                "Write an epistolary mini-story as a series of increasingly unhinged "
                "postcards over a summer.",
                "Sketch a scene with subtext-only conflict at a dinner table; no explicit insults.",
                "Generate a myth explaining why the moon flickers in this fantasy culture.",
                "Outline a children’s fable about patience starring a tortoise who isn’t "
                "a clichés retread.",
                "Write a microfiction under 100 words about the last bookstore on Earth.",
            ],
            keywords=[
                "story",
                "poem",
                "fiction",
                "novel",
                "scene",
                "dialogue",
                "character",
                "plot",
                "screenplay",
                "script",
                "lyrics",
                "song",
                "world-building",
                "worldbuilding",
                "creative",
                "imaginative",
                "fable",
                "myth",
                "fantasy",
                "science fiction",
                "tone",
                "voice",
                "narrative",
                "prose",
                "stanza",
                "metaphor",
            ],
            patterns=[
                r"\b(write|pen|draft)\b.*\b(stor(y|ies)|novella|poem|screenplay)\b",
                r"\bworld-?build",
                r"\bcharacter arcs?\b",
                r"\bsonnet\b",
                r"\bflash fiction\b",
                r"\bmicrofiction\b",
            ],
            intent_verbs=[
                "write",
                "compose",
                "draft",
                "imagine",
                "invent",
                "narrate",
            ],
        ),
        "factual_qa": DomainDefinition(
            name="factual_qa",
            anchors=[
                "Explain how vaccines train the adaptive immune system using plain language.",
                "What is the difference between HTTP and HTTPS at a high level for a nontechnical audience?",
                "Define photosynthesis and list the main inputs and outputs.",
                "What does 'quorum' mean in distributed systems, and why does it matter?",
                "Outline the causes of the Great Depression in two short paragraphs for a student.",
                "How do transformers use self-attention conceptually—no equations, just intuition?",
                "What is the capital of Mongolia, and what languages are commonly spoken there?",
                "Describe the water cycle step by step for a middle school worksheet.",
                "What is polymorphism in object-oriented programming and give one example?",
                "Explain buoyancy to a curious ten-year-old without using equations.",
                "What is CRISPR and what problem in genetics was it created to address?",
                "Who was Ada Lovelace and why is she significant in computing history?",
                "What is inflation, and how do central banks typically respond?",
                "Teach me the rules of pickleball focusing on the kitchen line and scoring.",
            ],
            keywords=[
                "what is",
                "define",
                "definition",
                "explain",
                "describe",
                "how does",
                "why does",
                "who was",
                "meaning of",
                "overview",
                "introduction",
                "concept",
                "basics",
                "fundamentals",
                "teach",
                "for beginners",
                "ELI5",
                "in simple terms",
                "fact",
                "background",
            ],
            patterns=[
                r"\bwhat (is|are)\b",
                r"\bhow does\b .{0,40}\bwork\b",
                r"\bdefine\b",
                r"\bexplain\b",
                r"\btell me about\b",
                r"\bcan you explain\b",
            ],
            intent_verbs=[
                "explain",
                "define",
                "describe",
                "clarify",
                "teach",
                "outline",
            ],
        ),
        "transformation": DomainDefinition(
            name="transformation",
            anchors=[
                "Rewrite this customer complaint reply to sound calm, empathetic, and "
                "policy-safe without changing facts.",
                "Paraphrase the following paragraph to avoid plagiarism while preserving meaning.",
                "Convert this bulleted meeting notes block into a polished status email.",
                "Turn this passive-voice lab report excerpt into concise active voice.",
                "Simplify this legal-sounding clause for a general audience without adding new commitments.",
                "Translate this casual message into formal British English suitable for a board update.",
                "Reformat this unstructured recipe into numbered steps and a grocery list.",
                "Change the tone from sarcastic to sincere while keeping the same sequence of ideas.",
                "Expand this one-line idea into a polite LinkedIn outreach message under 120 words.",
                "Condense this 200-word pitch into a 50-word cold email hook plus CTA.",
                "Convert this CSV into a Markdown table with aligned columns.",
                "Rewrite this SQL query comment block into a short design note for PR reviewers.",
                "Turn this JSON object into equivalent YAML with comments explaining keys.",
                "Style-transfer this product blurb to sound like a luxury brand without invented specs.",
            ],
            keywords=[
                "rewrite",
                "rephrase",
                "paraphrase",
                "reword",
                "convert",
                "translate",
                "reformat",
                "transform",
                "change tone",
                "tone down",
                "simplify",
                "plain language",
                "formalize",
                "informal to formal",
                "style",
                "edit for",
                "polish",
                "sanitize",
                "standardize",
            ],
            patterns=[
                r"\brewrite\b",
                r"\bparaphrase\b",
                r"\brephrase\b",
                r"\bturn this into\b",
                r"\bconvert\b.*\b(to|into)\b",
                r"\bchange (the )?tone\b",
                r"\bin (a )?more (professional|formal|casual)\b",
            ],
            intent_verbs=[
                "rewrite",
                "rephrase",
                "convert",
                "simplify",
                "reformat",
                "translate",
            ],
        ),
    }


DOMAIN_REGISTRY: Dict[str, DomainDefinition] = _domains()


def validate_registry() -> None:
    """Ensure registry matches supported domain list exactly."""
    missing = set(DOMAIN_NAMES) - set(DOMAIN_REGISTRY.keys())
    extra = set(DOMAIN_REGISTRY.keys()) - set(DOMAIN_NAMES)
    if missing or extra:
        raise RuntimeError(f"Domain registry mismatch. Missing: {missing}, Extra: {extra}")
    for name in DOMAIN_NAMES:
        d = DOMAIN_REGISTRY[name]
        if not (12 <= len(d.anchors) <= 15):
            raise RuntimeError(f"Domain {name} should have 12–15 anchors, got {len(d.anchors)}")
        if len(d.keywords) < 8:
            raise RuntimeError(f"Domain {name} needs realistic keyword coverage")


validate_registry()
