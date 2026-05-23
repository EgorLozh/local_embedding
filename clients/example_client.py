"""Example async Python client for the local embedding service."""

import asyncio
import os

import httpx

API_BASE_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8080")


async def embed_text(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        f"{API_BASE_URL}/embed",
        json={"text": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]


async def embed_batch(
    client: httpx.AsyncClient, texts: list[str]
) -> list[list[float]]:
    response = await client.post(
        f"{API_BASE_URL}/embed/batch",
        json={"texts": texts},
    )
    response.raise_for_status()
    return response.json()["embeddings"]


async def check_health(client: httpx.AsyncClient) -> dict:
    response = await client.get(f"{API_BASE_URL}/health")
    response.raise_for_status()
    return response.json()


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        health = await check_health(client)
        print("Health:", health)

        vector = await embed_text(client, "пример текста для embedding")
        print(f"Single embedding dimension: {len(vector)}")

        vectors = await embed_batch(client, ["первый текст", "второй текст"])
        print(f"Batch embeddings: {len(vectors)} vectors")
        for i, vec in enumerate(vectors):
            print(f"  [{i}] dimension={len(vec)}")


if __name__ == "__main__":
    asyncio.run(main())
