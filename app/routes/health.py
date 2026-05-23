from fastapi import APIRouter, Request

from app.schemas.responses import HealthResponse
from app.services.vllm_client import VllmEmbeddingClient

router = APIRouter(tags=["health"])


def _get_client(request: Request) -> VllmEmbeddingClient:
    return request.app.state.vllm_client


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    client = _get_client(request)
    vllm_ok = await client.check_health()
    if vllm_ok:
        return HealthResponse(status="ok", vllm="ok")
    return HealthResponse(status="degraded", vllm="unavailable")
