from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Text must not be empty or whitespace only")
        return stripped


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Texts to embed")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        settings = get_settings()
        if len(texts) > settings.batch_max_texts:
            raise ValueError(
                f"Batch size exceeds maximum of {settings.batch_max_texts}"
            )
        validated: list[str] = []
        for i, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Text at index {i} must not be empty or whitespace only")
            validated.append(stripped)
        return validated
