import asyncio
import json

from llm_service_main import token_budget
from llm_service_main.messaging.rabbitmq import RabbitConsumer
from llm_service_main.schemas import QueueRequest
from llm_service_main.service import LLMService
from llm_service_main.settings import Settings
from llm_service_main.sglang_client import SGLangClient


class SlowClient(SGLangClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.running = 0
        self.peak = 0
        self.gate = asyncio.Event()

    async def chat(self, base_url, model, messages, max_tokens, temperature):
        self.running += 1
        self.peak = max(self.peak, self.running)
        await self.gate.wait()
        self.running -= 1
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}


class FakeMessage:
    def __init__(self, body, reply_to=None):
        self.body, self.reply_to, self.correlation_id, self.acked = body, reply_to, None, False

    async def ack(self):
        self.acked = True


class FakeQueue:
    def __init__(self, messages):
        self.messages = list(messages)

    async def get(self, fail=True):
        return self.messages.pop(0) if self.messages else None


class FakeExchange:
    def __init__(self):
        self.published = []

    async def publish(self, msg, routing_key):
        self.published.append((routing_key, json.loads(msg.body)))


class FakeChannel:
    def __init__(self):
        self.default_exchange = FakeExchange()


async def test_requests_wait_in_queue_when_budget_full(monkeypatch):
    monkeypatch.setattr(token_budget, "_load_tokenizer", lambda m: None)
    s = Settings(context_window=10_000, default_max_output_tokens=1_000,
                 model_endpoints={"qwen": "http://x"}, default_model="qwen")
    client = SlowClient(s)
    svc = LLMService(s, client)

    # each request: ~3000 tokens in (12000 chars / 4) + 1000 out = 4000 -> only 2 fit in 10k
    body = QueueRequest(request_id="r", messages=[{"role": "user", "content": "a" * 12_000}]).model_dump_json().encode()
    msgs = [FakeMessage(body) for _ in range(5)]
    consumer = RabbitConsumer(s, svc)
    consumer._queue, consumer._channel = FakeQueue(msgs), FakeChannel()
    consumer._loop_task = asyncio.create_task(consumer._consume_loop())

    await asyncio.sleep(0.2)
    assert client.peak == 2  # budget caps parallelism
    assert len(consumer._queue.messages) == 2  # 2 running + 1 held waiting; 2 still in the queue
    assert 8_000 <= svc.budget.used <= 8_100  # two requests of ~4000

    client.gate.set()  # finish everything; waiting requests are released one by one
    await asyncio.sleep(0.3)
    assert all(m.acked for m in msgs)
    assert svc.budget.used == 0
    assert len(consumer._channel.default_exchange.published) == 5
    consumer._loop_task.cancel()
