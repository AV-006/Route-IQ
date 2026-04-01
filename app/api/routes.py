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

router = APIRouter()


def _get_analyzer(request: Request) -> PromptDomainAnalyzer:
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        raise RuntimeError("Analyzer not initialized; application lifespan may have failed")
    return analyzer


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
    return analyzer.analyze(body.prompt)
