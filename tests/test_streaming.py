import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_service_main import token_budget
from llm_service_main.api.routes import router
from llm_service_main.errors import LLMServiceError
from llm_service_main.messaging.rabbitmq import RabbitConsumer
from llm_service_main.schemas import QueueRequest
from llm_service_main.service import LLMService
from llm_service_main.settings import Settings
from llm_service_main.sglang_client import SGLangClient
from tests.test_rabbit_backpressure import FakeChannel, FakeMessage, FakeQueue


class StreamClient(SGLangClient):
    async def chat_stream(self, base_url, model, messages, max_tokens, temperature):
        for piece in ["Hel", "lo", "!"]:
            yield {"delta": piece}
        yield {"usage": {"prompt_tokens": 5, "completion_tokens": 3}}


def settings():
    return Settings(model_endpoints={"qwen": "http://x"}, default_model="qwen")


def make_app(svc):
    app = FastAPI()
    app.include_router(router)
    app.state.settings, app.state.service = svc.settings, svc

    @app.exception_handler(LLMServiceError)
    async def h(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code}})

    return app


def test_http_stream_sse_and_budget_released(monkeypatch):
    monkeypatch.setattr(token_budget, "_load_tokenizer", lambda m: None)
    s = settings()
    svc = LLMService(s, StreamClient(s))
    client = TestClient(make_app(svc))
    r = client.post("/generate", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(l[len("data: "):]) for l in r.text.split("\n\n") if l]
    assert "".join(e["delta"] for e in events if "delta" in e) == "Hello!"
    assert events[-1] == {"done": True, "model": "qwen", "usage": {"input_tokens": 5, "output_tokens": 3}}
    assert svc.budget.used == 0


def test_http_stream_validation_error_is_real_http_error(monkeypatch):
    monkeypatch.setattr(token_budget, "_load_tokenizer", lambda m: None)
    s = Settings(context_window=100, model_endpoints={"qwen": "http://x"}, default_model="qwen")
    client = TestClient(make_app(LLMService(s, StreamClient(s))))
    r = client.post("/generate", json={"stream": True, "messages": [{"role": "user", "content": "a" * 1000}]})
    assert r.status_code == 422


async def test_rabbit_stream_publishes_chunks_then_final(monkeypatch):
    monkeypatch.setattr(token_budget, "_load_tokenizer", lambda m: None)
    s = settings()
    svc = LLMService(s, StreamClient(s))
    body = QueueRequest(request_id="r1", stream=True,
                        messages=[{"role": "user", "content": "hi"}]).model_dump_json().encode()
    msg = FakeMessage(body)
    consumer = RabbitConsumer(s, svc)
    consumer._queue, consumer._channel = FakeQueue([msg]), FakeChannel()
    consumer._loop_task = asyncio.create_task(consumer._consume_loop())
    await asyncio.sleep(0.2)
    consumer._loop_task.cancel()

    out = [p for _, p in consumer._channel.default_exchange.published]
    assert [p["status"] for p in out] == ["chunk", "chunk", "chunk", "ok"]
    assert [p["delta"] for p in out[:3]] == ["Hel", "lo", "!"]
    assert out[-1]["result"]["content"] == "Hello!"
    assert msg.acked and svc.budget.used == 0


async def test_sglang_sse_parsing():
    import httpx

    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    s = settings()
    c = SGLangClient(s)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=sse)))
    events = [e async for e in c.chat_stream("http://x", "qwen", [], 10, 0.5)]
    assert events == [{"delta": "Hi"}, {"delta": " there"}, {"usage": {"prompt_tokens": 4, "completion_tokens": 2}}]
