from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelInvocationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    stop_sequences: Optional[list[str]] = None


class ModelInvocationResponse(BaseModel):
    model_id: str
    provider: str
    display_name: str
    response_text: str
    tokens_used: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0.0)
    error: Optional[str] = None
    fallback_used: bool
    fallback_model_id: Optional[str] = None

