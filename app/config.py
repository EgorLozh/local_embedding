from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vllm_base_url: str = Field(default="http://vllm:8000", alias="VLLM_BASE_URL")
    vllm_embeddings_path: str = Field(
        default="/v1/embeddings", alias="VLLM_EMBEDDINGS_PATH"
    )
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")

    http_timeout_sec: float = Field(default=30.0, alias="HTTP_TIMEOUT_SEC")
    http_connect_timeout_sec: float = Field(
        default=5.0, alias="HTTP_CONNECT_TIMEOUT_SEC"
    )
    http_max_connections: int = Field(default=100, alias="HTTP_MAX_CONNECTIONS")
    http_max_keepalive: int = Field(default=20, alias="HTTP_MAX_KEEPALIVE")

    retry_max_attempts: int = Field(default=3, alias="RETRY_MAX_ATTEMPTS")
    retry_min_wait_sec: float = Field(default=0.5, alias="RETRY_MIN_WAIT_SEC")
    retry_max_wait_sec: float = Field(default=4.0, alias="RETRY_MAX_WAIT_SEC")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    batch_max_texts: int = Field(default=64, alias="BATCH_MAX_TEXTS")

    @property
    def embeddings_url(self) -> str:
        base = self.vllm_base_url.rstrip("/")
        path = self.vllm_embeddings_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    @property
    def health_url(self) -> str:
        return f"{self.vllm_base_url.rstrip('/')}/health"


@lru_cache
def get_settings() -> Settings:
    return Settings()
