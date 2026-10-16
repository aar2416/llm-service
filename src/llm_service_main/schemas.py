from typing import Literal

from pydantic import BaseModel, Field

from .constants import DEFAULT_TEMPERATURE, MAX_TEMPERATURE


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    model: str | None = Field(None, description="Model name; defaults to LLM_DEFAULT_MODEL")
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(None, gt=0)
    stream: bool = Field(False, description="Stream tokens as they are generated")
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0, le=MAX_TEMPERATURE)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class GenerateResponse(BaseModel):
    model: str
    content: str
    usage: Usage


class QueueRequest(GenerateRequest):
    """Message consumed from RabbitMQ. request_id is echoed on the result."""

    request_id: str | None = None
