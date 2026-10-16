"""Token counting, per-request limits, and the shared in-flight token budget."""
import asyncio
from functools import lru_cache

from .constants import CHARS_PER_TOKEN_FALLBACK, TOKENIZER_CACHE_SIZE
from .errors import BudgetTimeout, TokenLimitExceeded
from .schemas import Message
from .settings import Settings


@lru_cache(maxsize=TOKENIZER_CACHE_SIZE)
def _load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model)
    except Exception:  # transformers missing or no network/cache
        return None


def count_tokens(model: str, messages: list[Message]) -> int:
    text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    tok = _load_tokenizer(model)
    if tok is not None:
        return len(tok.encode(text))
    return max(1, len(text) // CHARS_PER_TOKEN_FALLBACK)


def check_budget(settings: Settings, input_tokens: int, requested_max: int | None) -> int:
    """Per-request check. Raise if input exceeds the cap; return output tokens to reserve.

    Reserved output = requested max_tokens (or the default), clamped so that
    input + output never exceeds the context window.
    """
    if input_tokens > settings.max_input_tokens:
        raise TokenLimitExceeded(input_tokens, settings.max_input_tokens)
    wanted = requested_max or settings.default_max_output_tokens
    return min(wanted, settings.context_window - input_tokens)


class TokenBudget:
    """Shared pool of tokens (input + reserved output) for requests in flight.

    A request reserves its cost before running and releases it when done; if the
    pool is exhausted, acquire() waits until enough tokens are released.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._used = 0
        self._cond = asyncio.Condition()

    @property
    def used(self) -> int:
        return self._used

    @property
    def free(self) -> int:
        return self.capacity - self._used

    async def acquire(self, tokens: int, timeout: float | None = None) -> None:
        if tokens > self.capacity:
            raise ValueError(f"{tokens} tokens exceeds total budget {self.capacity}")

        async def _wait():
            async with self._cond:
                await self._cond.wait_for(lambda: self._used + tokens <= self.capacity)
                self._used += tokens

        try:
            await asyncio.wait_for(_wait(), timeout)
        except asyncio.TimeoutError:
            raise BudgetTimeout(
                f"Timed out after {timeout}s waiting for {tokens} tokens of budget"
            ) from None

    async def release(self, tokens: int) -> None:
        async with self._cond:
            self._used -= tokens
            self._cond.notify_all()

    async def wait_for_free(self) -> None:
        """Return once any budget is free (lets the consumer leave messages in RabbitMQ)."""
        async with self._cond:
            await self._cond.wait_for(lambda: self._used < self.capacity)
