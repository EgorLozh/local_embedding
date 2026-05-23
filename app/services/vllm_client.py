from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.exceptions import (
    InvalidUpstreamResponseError,
    ModelUnavailableError,
    UpstreamTimeoutError,
)

logger = structlog.get_logger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


class _RetryableUpstreamError(Exception):
    """Transient upstream failure eligible for retry."""


def _parse_embeddings(payload: dict[str, Any], expected_count: int) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise InvalidUpstreamResponseError("Response missing non-empty 'data' array")

    indexed: list[tuple[int, list[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise InvalidUpstreamResponseError("Invalid item in 'data' array")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise InvalidUpstreamResponseError("Invalid or empty embedding vector")
        if not all(isinstance(v, (int, float)) for v in embedding):
            raise InvalidUpstreamResponseError("Embedding values must be numeric")
        index = item.get("index", len(indexed))
        if not isinstance(index, int):
            raise InvalidUpstreamResponseError("Invalid embedding index")
        indexed.append((index, [float(v) for v in embedding]))

    indexed.sort(key=lambda pair: pair[0])
    embeddings = [vec for _, vec in indexed]

    if len(embeddings) != expected_count:
        raise InvalidUpstreamResponseError(
            f"Expected {expected_count} embeddings, got {len(embeddings)}"
        )
    return embeddings


class VllmEmbeddingClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    async def close(self) -> None:
        await self._client.aclose()

    async def check_health(self) -> bool:
        try:
            response = await self._client.get(
                self._settings.health_url,
                timeout=httpx.Timeout(5.0, connect=2.0),
            )
            return response.status_code == 200
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError):
            return False

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._settings.embedding_model,
            "input": texts if len(texts) > 1 else texts[0],
            "encoding_format": "float",
        }
        return await self._request_embeddings(payload, expected_count=len(texts))

    async def _request_embeddings(
        self, payload: dict[str, Any], expected_count: int
    ) -> list[list[float]]:
        start = time.perf_counter()
        log = logger.bind(
            model=self._settings.embedding_model,
            batch_size=expected_count,
        )

        try:
            response_data = await self._post_with_retry(payload)
        except UpstreamTimeoutError:
            log.error("vllm_timeout", latency_ms=_elapsed_ms(start))
            raise
        except (_RetryableUpstreamError, ModelUnavailableError) as exc:
            log.error("vllm_unavailable", latency_ms=_elapsed_ms(start))
            if isinstance(exc, ModelUnavailableError):
                raise
            raise ModelUnavailableError(
                "Embedding model is unavailable after retries"
            ) from exc
        except InvalidUpstreamResponseError:
            log.error("vllm_invalid_response", latency_ms=_elapsed_ms(start))
            raise

        try:
            embeddings = _parse_embeddings(response_data, expected_count)
        except InvalidUpstreamResponseError:
            log.error("vllm_parse_failed", latency_ms=_elapsed_ms(start))
            raise

        log.info("vllm_success", latency_ms=_elapsed_ms(start))
        return embeddings

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings
        attempt = retry(
            stop=stop_after_attempt(settings.retry_max_attempts),
            wait=wait_exponential(
                min=settings.retry_min_wait_sec,
                max=settings.retry_max_wait_sec,
            ),
            retry=retry_if_exception_type(_RetryableUpstreamError),
            reraise=True,
        )

        @attempt
        async def _do_post() -> dict[str, Any]:
            return await self._post_once(payload)

        return await _do_post()

    async def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings
        timeout = httpx.Timeout(
            settings.http_timeout_sec,
            connect=settings.http_connect_timeout_sec,
        )

        try:
            response = await self._client.post(
                settings.embeddings_url,
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError() from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ModelUnavailableError(
                "Cannot connect to embedding model service"
            ) from exc

        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableUpstreamError(
                f"Upstream returned status {response.status_code}"
            )

        if response.status_code >= 500:
            raise ModelUnavailableError(
                f"Embedding model returned status {response.status_code}"
            )

        if response.status_code >= 400:
            detail = response.text[:500]
            raise InvalidUpstreamResponseError(
                f"Upstream rejected request: {response.status_code} — {detail}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise InvalidUpstreamResponseError("Upstream returned invalid JSON") from exc


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def create_http_client(settings: Settings) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_keepalive,
    )
    timeout = httpx.Timeout(
        settings.http_timeout_sec,
        connect=settings.http_connect_timeout_sec,
    )
    return httpx.AsyncClient(limits=limits, timeout=timeout)
