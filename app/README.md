# Prompt Domain Weighting Engine

Production-oriented **stage-1** component for an LLM router: given a user prompt, it returns a **soft distribution** over eight semantic domains (weights sum to 1.0), top domains, a **confidence** score, per-domain score breakdowns, and lightweight **text features** for debugging and future routing.

It also returns an overall **prompt complexity** estimate (one score per prompt, not per-domain), with an explainable signal breakdown.

No model routing, no Bedrock/LLM calls, no persistence—**prompt analysis only**.

## Supported domains

| Domain | Role |
|--------|------|
| `coding` | Implementation, debugging, APIs, SQL, tooling |
| `math` | Algebra, calculus, probability, proofs, discrete math |
| `reasoning` | Compare/contrast, tradeoffs, justification, systems thinking |
| `summarization` | Condense long text, TL;DR, executive summaries |
| `extraction` | Structured pull-outs: entities, fields, JSON/CSV |
| `creative_writing` | Stories, poems, scripts, voice, world-building |
| `factual_qa` | Definitions, concept explanation, educational Q&A |
| `transformation` | Rewrite, paraphrase, tone/format conversion |

## Install

Requires **Python 3.11+**.

```bash
cd app
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

The first request (or the sample runner) will download **`BAAI/bge-small-en-v1.5`** from Hugging Face (~130 MB).

## Run the API

From the **repository root** (parent of `app/`):

```bash
source app/.venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /` — health and service metadata  
- `GET /domains` — domain config stats (anchor/keyword/pattern counts)  
- `POST /analyze` — main analysis endpoint  

### Sample request

```bash
curl -X POST http://localhost:8000/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\": \"Solve this graph problem using BFS and write C++ code\"}"
```

### Sample response (shape)

```json
{
  "prompt": "Solve this graph problem using BFS and write C++ code",
  "domain_scores": {
    "coding": 0.34,
    "math": 0.12,
    "reasoning": 0.22,
    "summarization": 0.05,
    "extraction": 0.04,
    "creative_writing": 0.03,
    "factual_qa": 0.08,
    "transformation": 0.12
  },
  "top_domains": ["coding", "reasoning", "transformation"],
  "confidence": 0.78,
  "complexity_score": 0.62,
  "complexity_band": "medium",
  "complexity_signals": {
    "length": {
      "name": "length",
      "score": 0.18,
      "weight": 0.18,
      "contribution": 0.0324,
      "evidence": ["token_count=22"],
      "detail": {"token_count": 22, "normalizer": 120.0}
    }
  },
  "per_domain_breakdown": {
    "coding": {
      "semantic_score": 0.82,
      "keyword_score": 0.62,
      "pattern_score": 0.25,
      "intent_score": 0.67,
      "raw_score": 0.71
    }
  },
  "text_features": {
    "token_count": 12,
    "avg_word_length": 4.2,
    "special_char_ratio": 0.01,
    "code_symbol_ratio": 0.0,
    "digit_ratio": 0.0,
    "uppercase_ratio": 0.08,
    "newline_count": 0
  }
}
```

*(Exact numbers depend on the prompt and model.)*

## Run sample prompts locally

From repository root:

```bash
python -m app.tests.sample_prompts
```

Loads the model once, then prints `domain_scores`, `top_domains`, and `confidence` for curated examples—including mixed-domain, ambiguous, and weak-signal prompts.

## Architecture

```
app/
├── main.py                 # FastAPI app + startup embedding load
├── api/routes.py           # HTTP endpoints
├── core/                   # Settings + domain name constants
├── intelligence/
│   ├── domain_configs.py   # Anchors, keywords, regex, intent verbs per domain
│   ├── embeddings.py       # SentenceTransformer + prototype vectors
│   ├── domain_scorer.py    # Hybrid scoring
│   ├── confidence.py       # Distribution sharpness + top-1 vs top-2 gap
│   ├── feature_extractor.py
│   └── analyzer.py         # Orchestration
├── models/schemas.py       # Pydantic request/response
└── utils/                  # Text matching + numpy helpers
```

**Startup:** the embedding model loads once. For each domain, **12–15 anchor prompts** are embedded; their mean vector is **L2-normalized** to form a **prototype**. Prototypes stay in memory.

**Per request:** the prompt is embedded once, cosine similarity to each prototype is mapped to **[0, 1]**. Keyword hits, regex patterns, and intent verbs produce auxiliary scores in **[0, 1]**. These are combined with fixed weights (default: 55% semantic, 20% keyword, 10% pattern, 15% intent). Raw scores are **clamped non-negative** and **normalized to sum to 1.0** (uniform fallback if everything is zero).

## Why anchor prompts matter

Prototype quality dominates separation between domains. Lazy or repetitive anchors (“write code” ×15) collapse distinct subtypes into one blob and hurt **mixed-domain** prompts. Anchors should cover **real user tasks** across subtypes (e.g., math: word problems vs proofs vs probability). If two domains drift together in production, **edit `domain_configs.py`**: add discriminative anchors for the confused pair, avoid near-duplicates, and restart the app so prototypes recompute.

## How hybrid scoring works

- **Semantic (primary):** embedding similarity to the domain prototype.  
- **Keyword / pattern / intent (boosters):** surface cues that disambiguate short prompts or jargon—capped so they **cannot override** a strong semantic signal for long, rich prompts.  
- **Multi-domain:** no softmax winner-take-all; all domains get a weight after normalization, so **parallel intent** (e.g., math + coding) can show up as two meaningful modes.

**Confidence** blends (1) how **peaky** the distribution is vs uniform (entropy) and (2) the **gap** between the best and second-best domain, with a small boost if the top domain is very dominant.

## Limitations

- English-centric anchors and model (`bge-small-en-v1.5`).  
- Domain boundaries are fuzzy by design; some prompts legitimately spread mass across many domains.  
- Very short or vague prompts yield **lower confidence** and flatter distributions.  
- No personalization, no user feedback loop, no online learning.

## Future improvements

- Calibration layer from logged labels; per-tenant anchor packs.  
- Second-stage classifier or small cross-encoder rerank on top-3 domains.  
- Multilingual anchors + multilingual embedding model.  
- Per-domain temperature / concentration priors for sharper or softer routing.

## Configuration tuning

- **Score blend:** `app/core/config.py` → `ScoringWeights` and keyword/pattern caps.  
- **Domains:** `app/intelligence/domain_configs.py` only (then restart).

## License

Hackathon / demo use—verify licenses for `sentence-transformers`, PyTorch, and the BGE checkpoint for your deployment.
