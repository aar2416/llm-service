from collections.abc import AsyncIterator
from dataclasses import dataclass

from .schemas import GenerateRequest, GenerateResponse, Usage
from .settings import Settings
from .sglang_client import SGLangClient
from .token_budget import TokenBudget, check_budget, count_tokens


@dataclass(frozen=True)
class Plan:
    """A validated request plus the tokens it reserves from the shared budget."""

    model: str
    base_url: str
    input_tokens: int
    max_tokens: int

    @property
    def cost(self) -> int:
        return self.input_tokens + self.max_tokens


class LLMService:
    """Shared by the HTTP API and the RabbitMQ consumer."""

    def __init__(self, settings: Settings, client: SGLangClient):
        self.settings = settings
        self.client = client
        self.budget = TokenBudget(settings.context_window)

    def plan(self, req: GenerateRequest) -> Plan:
        """Validate model + token limits (raises) and compute the budget cost."""
        model, base_url = self.client.resolve(req.model)
        input_tokens = count_tokens(model, req.messages)
        max_tokens = check_budget(self.settings, input_tokens, req.max_tokens)
        return Plan(model, base_url, input_tokens, max_tokens)

    async def execute(self, plan: Plan, req: GenerateRequest) -> GenerateResponse:
        """Call SGLang. The caller must already hold plan.cost tokens of budget."""
        data = await self.client.chat(
            plan.base_url, plan.model, req.messages, plan.max_tokens, req.temperature
        )
        usage = data.get("usage", {})
        return GenerateResponse(
            model=plan.model,
            content=data["choices"][0]["message"]["content"],
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", plan.input_tokens),
                output_tokens=usage.get("completion_tokens", 0),
            ),
        )

    async def stream_events(self, plan: Plan, req: GenerateRequest) -> AsyncIterator[dict]:
        """Yield {"delta": str} events, then a final {"done": True, "model", "usage"}.

        The caller must already hold plan.cost tokens of budget.
        """
        usage = {}
        async for ev in self.client.chat_stream(
            plan.base_url, plan.model, req.messages, plan.max_tokens, req.temperature
        ):
            if "usage" in ev:
                usage = ev["usage"]
            else:
                yield ev
        yield {
            "done": True,
            "model": plan.model,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", plan.input_tokens),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    async def stream(self, req: GenerateRequest) -> AsyncIterator[dict]:
        """HTTP streaming path: validate, wait for budget, stream, always release."""
        plan = self.plan(req)  # raises before anything is reserved
        await self.budget.acquire(plan.cost, self.settings.budget_wait_timeout_s)
        try:
            async for ev in self.stream_events(plan, req):
                yield ev
        finally:
            await self.budget.release(plan.cost)

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        """HTTP path: wait for budget (up to budget_wait_timeout_s), run, release."""
        plan = self.plan(req)
        await self.budget.acquire(plan.cost, self.settings.budget_wait_timeout_s)
        try:
            return await self.execute(plan, req)
        finally:
            await self.budget.release(plan.cost)
