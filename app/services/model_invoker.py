from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, ReadTimeoutError

from app.registry.model_registry import ModelRegistry
from app.schemas.invocation import ModelInvocationResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvokerConfig:
    max_retries: int = 2
    request_timeout_s: float = 30.0
    retry_backoff_base_s: float = 0.35
    retry_backoff_max_s: float = 3.0
    region_name: str = "us-east-1"

    @staticmethod
    def from_env() -> "InvokerConfig":
        def _get_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return default if raw is None else int(raw)

        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return default if raw is None else float(raw)

        return InvokerConfig(
            max_retries=_get_int("ROUTER_INVOKE_MAX_RETRIES", 2),
            request_timeout_s=_get_float("ROUTER_INVOKE_TIMEOUT_S", 30.0),
            retry_backoff_base_s=_get_float("ROUTER_INVOKE_BACKOFF_BASE_S", 0.35),
            retry_backoff_max_s=_get_float("ROUTER_INVOKE_BACKOFF_MAX_S", 3.0),
            region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
        )


def _sleep_backoff(attempt_idx: int, cfg: InvokerConfig) -> None:
    delay = min(cfg.retry_backoff_max_s, cfg.retry_backoff_base_s * (2**attempt_idx))
    time.sleep(delay)


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (EndpointConnectionError, ReadTimeoutError)):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        http = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"ThrottlingException", "TooManyRequestsException", "RequestTimeout", "ServiceUnavailableException"}:
            return True
        if isinstance(http, int) and 500 <= http <= 599:
            return True
    return False


def _bedrock_client(cfg: InvokerConfig):
    # Avoid global client to keep tests and env config simple.
    return boto3.client("bedrock-runtime", region_name=cfg.region_name)


def _build_payload_for_model(
    prompt: str,
    model_id: str,
    *,
    max_tokens: int,
    temperature: float,
    stop_sequences: Optional[list[str]],
) -> dict[str, Any]:
    mid = model_id.lower()

    if mid.startswith("anthropic."):
        payload: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        return payload

    # Generic fallback payload (works for several text-generation style models).
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop_sequences:
        payload["stop"] = stop_sequences
    return payload


def _extract_text_and_tokens(model_response: dict[str, Any]) -> tuple[str, int]:
    # Be tolerant: Bedrock responses differ across providers/models.
    text: Optional[str] = None
    tokens = 0

    if isinstance(model_response.get("outputText"), str):
        text = model_response["outputText"]
    elif isinstance(model_response.get("completion"), str):
        text = model_response["completion"]
    elif isinstance(model_response.get("generation"), str):
        text = model_response["generation"]
    elif isinstance(model_response.get("text"), str):
        text = model_response["text"]
    else:
        content = model_response.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                text = first["text"]
        results = model_response.get("results")
        if text is None and isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for k in ("outputText", "text", "completion"):
                    if isinstance(first.get(k), str):
                        text = first[k]
                        break

    usage = model_response.get("usage")
    if isinstance(usage, dict):
        for key in ("total_tokens", "tokens", "totalTokens"):
            if isinstance(usage.get(key), int):
                tokens = int(usage[key])
                break
        if tokens == 0:
            in_tok = usage.get("input_tokens") or usage.get("inputTokens")
            out_tok = usage.get("output_tokens") or usage.get("outputTokens")
            if isinstance(in_tok, int) and isinstance(out_tok, int):
                tokens = int(in_tok + out_tok)

    return (text or "").strip(), max(0, tokens)


def _invoke_once(
    prompt: str,
    model_id: str,
    *,
    max_tokens: int,
    temperature: float,
    stop_sequences: Optional[list[str]],
    cfg: InvokerConfig,
) -> tuple[str, int, float]:
    client = _bedrock_client(cfg)
    payload = _build_payload_for_model(
        prompt,
        model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
    )

    started = time.perf_counter()
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    raw_body = resp.get("body")
    if hasattr(raw_body, "read"):
        body_bytes = raw_body.read()
    else:
        body_bytes = raw_body if isinstance(raw_body, (bytes, bytearray)) else b"{}"
    decoded = json.loads(body_bytes.decode("utf-8"))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    text, tokens = _extract_text_and_tokens(decoded)
    return text, tokens, elapsed_ms


def invoke_model(
    prompt: str,
    model_id: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stop_sequences: Optional[list[str]] = None,
    fallback_model_ids: Optional[list[str]] = None,
    registry: Optional[ModelRegistry] = None,
) -> ModelInvocationResponse:
    """
    Invoke a Bedrock model with retries and ordered fallback.

    - Retries transient failures up to ROUTER_INVOKE_MAX_RETRIES.
    - If primary fails permanently, attempts fallbacks in order.
    """
    cfg = InvokerConfig.from_env()

    all_models = [model_id] + [m for m in (fallback_model_ids or []) if m and m != model_id]
    last_error: Optional[str] = None
    errors: list[str] = []

    for idx, candidate_id in enumerate(all_models):
        is_fallback = idx > 0
        attempts = 0
        while True:
            try:
                text, tokens, latency_ms = _invoke_once(
                    prompt,
                    candidate_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                    cfg=cfg,
                )
                if not text:
                    raise RuntimeError("Empty model response")

                provider = "bedrock"
                display_name = candidate_id
                if registry is not None:
                    try:
                        m = next(x for x in registry.get_all_models() if x.model_id == candidate_id)
                        provider = m.provider
                        display_name = m.display_name
                    except StopIteration:
                        pass

                logger.info(
                    "model_invoke ok model_id=%s latency_ms=%.2f tokens=%d fallback=%s",
                    candidate_id,
                    latency_ms,
                    tokens,
                    is_fallback,
                )

                return ModelInvocationResponse(
                    model_id=candidate_id,
                    provider=provider,
                    display_name=display_name,
                    response_text=text,
                    tokens_used=tokens,
                    latency_ms=latency_ms,
                    error=None,
                    fallback_used=is_fallback,
                    fallback_model_id=(candidate_id if is_fallback else None),
                )
            except (ClientError, EndpointConnectionError, ReadTimeoutError, BotoCoreError, RuntimeError) as exc:
                attempts += 1
                msg = f"{candidate_id}: {type(exc).__name__}: {str(exc)}"
                last_error = msg
                errors.append(msg)

                transient = _is_transient_error(exc)
                if transient and attempts <= cfg.max_retries:
                    logger.warning("model_invoke retry attempt=%d err=%s", attempts, msg)
                    _sleep_backoff(attempts - 1, cfg)
                    continue

                logger.warning(
                    "model_invoke fail model_id=%s transient=%s attempts=%d err=%s",
                    candidate_id,
                    transient,
                    attempts,
                    msg,
                )
                break

    provider = "bedrock"
    display_name = model_id
    if registry is not None:
        try:
            m = next(x for x in registry.get_all_models() if x.model_id == model_id)
            provider = m.provider
            display_name = m.display_name
        except StopIteration:
            pass

    combined_error = "; ".join(errors[-6:]) if errors else (last_error or "Unknown invocation error")
    return ModelInvocationResponse(
        model_id=model_id,
        provider=provider,
        display_name=display_name,
        response_text="",
        tokens_used=0,
        latency_ms=0.0,
        error=combined_error,
        fallback_used=bool(fallback_model_ids),
        fallback_model_id=(fallback_model_ids[0] if fallback_model_ids else None),
    )

