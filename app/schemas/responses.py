from typing import Literal

from pydantic import BaseModel, Field


class EmbedResponse(BaseModel):
    embedding: list[float] = Field(..., description="Dense embedding vector")


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ..., description="Embedding vectors in input order"
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(..., description="API health status")
    vllm: Literal["ok", "unavailable"] = Field(
        ..., description="vLLM upstream health status"
    )
