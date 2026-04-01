"""FastAPI route definitions for the Prompt Domain Weighting Engine."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.constants import DOMAIN_NAMES
from app.intelligence.analyzer import PromptDomainAnalyzer
from app.intelligence.domain_configs import DOMAIN_REGISTRY
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DomainMetadata,
    DomainsListResponse,
    HealthResponse,
)
from app.registry.model_registry import ModelRegistry
from app.schemas.invocation import ModelInvocationRequest, ModelInvocationResponse
from app.services.model_invoker import invoke_model
from app.services.model_selector import select_best_model

router = APIRouter()


def _get_analyzer(request: Request) -> PromptDomainAnalyzer:
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        raise RuntimeError("Analyzer not initialized; application lifespan may have failed")
    return analyzer


def _get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise RuntimeError("Model registry not initialized; application lifespan may have failed")
    return registry


@router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Service health and identity."""
    return HealthResponse(
        service="Prompt Domain Weighting Engine",
        version="0.1.0",
        status="ok",
        embedding_model=settings.embedding_model_name,
        domains_supported=list(DOMAIN_NAMES),
    )


@router.get("/domains", response_model=DomainsListResponse)
async def list_domains() -> DomainsListResponse:
    """Supported domains with configuration cardinality."""
    meta: list[DomainMetadata] = []
    for name in DOMAIN_NAMES:
        d = DOMAIN_REGISTRY[name]
        meta.append(
            DomainMetadata(
                name=name,
                anchor_count=len(d.anchors),
                keyword_count=len(d.keywords),
                pattern_count=len(d.patterns),
                intent_verb_count=len(d.intent_verbs),
                phrase_boost_count=len(d.phrase_boosts),
            )
        )
    return DomainsListResponse(
        domains=meta,
        embedding_model=settings.embedding_model_name,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_prompt(
    body: AnalyzeRequest,
    request: Request,
) -> AnalyzeResponse:
    """
    Analyze a user prompt and return soft domain weights with full breakdown.

    Uses hybrid semantic + keyword + pattern + intent scoring; results sum to 1.0.
    """
    analyzer = _get_analyzer(request)
    registry = _get_registry(request)
    analysis = analyzer.analyze(body.prompt)
    selection = select_best_model(analysis, registry)
    fallback_ids = [c.model_id for c in selection.ranked_candidates if c.model_id != selection.selected_model]
    invocation = invoke_model(
        body.prompt,
        selection.selected_model,
        fallback_model_ids=fallback_ids,
        registry=registry,
    )
    return analysis.model_copy(update={"model_selection": selection, "invocation_response": invocation})


@router.post("/models/invoke", response_model=ModelInvocationResponse)
async def invoke_model_endpoint(
    body: ModelInvocationRequest,
    request: Request,
) -> ModelInvocationResponse:
    registry = _get_registry(request)
    return invoke_model(
        body.prompt,
        body.model_id,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stop_sequences=body.stop_sequences,
        registry=registry,
    )


@router.post("/models/select")
async def select_model_debug(
    body: AnalyzeRequest,
    request: Request,
) -> dict:
    """Debug endpoint: run analyzer + selector and return compact routing trace."""
    analyzer = _get_analyzer(request)
    registry = _get_registry(request)
    analysis = analyzer.analyze(body.prompt)
    selection = select_best_model(analysis, registry)
    return {
        "prompt": body.prompt,
        "analysis_summary": {
            "top_domains": analysis.top_domains,
            "confidence": analysis.confidence,
            "complexity_score": analysis.complexity_score,
            "complexity_band": analysis.complexity_band,
        },
        "selected_model": {
            "model_id": selection.selected_model,
            "provider": selection.provider,
            "display_name": selection.display_name,
            "selection_confidence": selection.selection_confidence,
            "selection_reasoning": selection.selection_reasoning,
        },
        "ranked_candidates": [candidate.model_dump() for candidate in selection.ranked_candidates],
    }
