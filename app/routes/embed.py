from fastapi import APIRouter, Request

from app.schemas.requests import BatchEmbedRequest, EmbedRequest
from app.schemas.responses import BatchEmbedResponse, EmbedResponse
from app.services.vllm_client import VllmEmbeddingClient

router = APIRouter(tags=["embeddings"])


def _get_client(request: Request) -> VllmEmbeddingClient:
    return request.app.state.vllm_client


@router.post("/embed", response_model=EmbedResponse)
async def embed_text(body: EmbedRequest, request: Request) -> EmbedResponse:
    client = _get_client(request)
    embedding = await client.embed(body.text)
    return EmbedResponse(embedding=embedding)


@router.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(body: BatchEmbedRequest, request: Request) -> BatchEmbedResponse:
    client = _get_client(request)
    embeddings = await client.embed_batch(body.texts)
    return BatchEmbedResponse(embeddings=embeddings)
