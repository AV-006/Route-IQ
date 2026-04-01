"""
FastAPI entrypoint for the Prompt Domain Weighting Engine.

Run from repository root:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.intelligence.analyzer import PromptDomainAnalyzer
from app.intelligence.embeddings import get_embedding_service
from app.registry.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model and domain prototypes once at startup."""
    embedder = get_embedding_service()
    logger.info("Initializing embedding service...")
    embedder.initialize()
    app.state.analyzer = PromptDomainAnalyzer(embedder)
    registry = ModelRegistry()
    registry.load_models()
    app.state.model_registry = registry
    logger.info("Analyzer ready.")
    yield


app = FastAPI(
    title="Prompt Domain Weighting Engine",
    description="Soft multi-domain relevance scores for LLM routing (stage 1).",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


__all__ = ["app"]
