import pytest

from llm_service_main import token_budget
from llm_service_main.settings import Settings
from llm_service_main.errors import TokenLimitExceeded, UnknownModel
from llm_service_main.schemas import GenerateRequest, Message
from llm_service_main.service import LLMService
from llm_service_main.sglang_client import SGLangClient
from llm_service_main.errors import BudgetTimeout
from llm_service_main.token_budget import TokenBudget, check_budget


@pytest.fixture(autouse=True)
def no_tokenizer(monkeypatch):
    monkeypatch.setattr(token_budget, "_load_tokenizer", lambda m: None)


def test_input_cap_is_two_thirds():
    assert Settings(context_window=30_000).max_input_tokens == 20_000


def test_over_cap_raises():
    s = Settings(context_window=30_000)
    with pytest.raises(TokenLimitExceeded):
        check_budget(s, 20_001, None)
    assert check_budget(s, 20_000, None) == 1_000  # default output reservation
    assert check_budget(s, 100, 50) == 50
    # output is clamped so input + output <= context window
    assert check_budget(Settings(context_window=30_000, max_input_ratio=1.0), 29_500, 5_000) == 500


class FakeClient(SGLangClient):
    async def chat(self, base_url, model, messages, max_tokens, temperature):
        return {
            "choices": [{"message": {"content": f"hi from {model}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }


def make(settings):
    return LLMService(settings, FakeClient(settings))


def msgs(text):
    return [Message(role="user", content=text)]


async def test_generate_picks_requested_model():
    s = Settings(model_endpoints={"qwen": "http://x", "other": "http://y"}, default_model="qwen")
    r = await make(s).generate(GenerateRequest(model="other", messages=msgs("yo")))
    assert r.content == "hi from other"


async def test_unknown_model():
    s = Settings(model_endpoints={"qwen": "http://x"}, default_model="qwen")
    with pytest.raises(UnknownModel):
        await make(s).generate(GenerateRequest(model="nope", messages=msgs("yo")))


async def test_too_long_input():
    s = Settings(context_window=100, model_endpoints={"qwen": "http://x"}, default_model="qwen")
    with pytest.raises(TokenLimitExceeded):
        await make(s).generate(GenerateRequest(messages=msgs("a" * 1000)))


async def test_budget_blocks_until_released():
    import asyncio

    b = TokenBudget(30_000)
    await b.acquire(4_000)  # e.g. 3000 input + 1000 output
    await b.acquire(26_000)
    assert b.free == 0

    waiter = asyncio.create_task(b.acquire(4_000))
    await asyncio.sleep(0.05)
    assert not waiter.done()  # waits: budget exhausted
    await b.release(4_000)
    await asyncio.wait_for(waiter, 1)
    assert b.used == 30_000


async def test_budget_parallel_small_requests():
    b = TokenBudget(30_000)
    for _ in range(7):
        await b.acquire(4_000)  # seven 4k requests fit in parallel
    assert b.free == 2_000
    with pytest.raises(BudgetTimeout):
        await b.acquire(4_000, timeout=0.05)
    assert b.used == 28_000  # timed-out waiter reserved nothing


async def test_generate_releases_budget():
    s = Settings(model_endpoints={"qwen": "http://x"}, default_model="qwen")
    svc = make(s)
    await svc.generate(GenerateRequest(messages=msgs("yo")))
    assert svc.budget.used == 0
