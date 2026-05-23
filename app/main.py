from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.exceptions import AppError
from app.logging import setup_logging
from app.routes import api_router
from app.services.vllm_client import VllmEmbeddingClient, create_http_client

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)

    http_client = create_http_client(settings)
    vllm_client = VllmEmbeddingClient(settings, http_client)
    app.state.vllm_client = vllm_client

    logger.info(
        "application_started",
        vllm_url=settings.embeddings_url,
        model=settings.embedding_model,
    )

    yield

    await vllm_client.close()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local Embedding Service",
        description="Production-ready embedding API backed by vLLM (BAAI/bge-m3)",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(api_router)

    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = ".".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "validation error")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return JSONResponse(
            status_code=422,
            content={"detail": "; ".join(messages) if messages else "Validation error"},
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        _request: Request, exc: ValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    return app


app = create_app()
