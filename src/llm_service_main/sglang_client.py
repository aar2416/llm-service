"""Thin client for SGLang's OpenAI-compatible server (python -m sglang.launch_server)."""
import json
from collections.abc import AsyncIterator

import httpx

from .settings import Settings
from .constants import SGLANG_CHAT_PATH
from .errors import UnknownModel, UpstreamError
from .schemas import Message


class SGLangClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._http = httpx.AsyncClient(timeout=settings.request_timeout_s)

    def resolve(self, model: str | None) -> tuple[str, str]:
        name = model or self._settings.default_model
        endpoints = self._settings.model_endpoints
        if name not in endpoints:
            raise UnknownModel(name, sorted(endpoints))
        return name, endpoints[name].rstrip("/")

    async def chat(
        self, base_url: str, model: str, messages: list[Message],
        max_tokens: int, temperature: float,
    ) -> dict:
        try:
            r = await self._http.post(
                f"{base_url}{SGLANG_CHAT_PATH}",
                json={
                    "model": model,
                    "messages": [m.model_dump() for m in messages],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise UpstreamError(f"SGLang request failed: {e}") from e

    async def chat_stream(
        self, base_url: str, model: str, messages: list[Message],
        max_tokens: int, temperature: float,
    ) -> AsyncIterator[dict]:
        """Yield {"delta": str} per chunk, then {"usage": {...}} if the server reports it."""
        body = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            async with self._http.stream("POST", f"{base_url}{SGLANG_CHAT_PATH}", json=body) as r:
                if r.status_code >= 400:
                    await r.aread()
                    r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            yield {"delta": delta}
                    if chunk.get("usage"):
                        yield {"usage": chunk["usage"]}
        except httpx.HTTPError as e:
            raise UpstreamError(f"SGLang stream failed: {e}") from e

    async def aclose(self) -> None:
        await self._http.aclose()
