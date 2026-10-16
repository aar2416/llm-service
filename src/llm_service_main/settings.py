"""All environment-driven configuration (prefix LLM_, see .env.example)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from . import constants as c


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=c.ENV_PREFIX, env_file=c.ENV_FILE, extra="ignore")

    # Token budget: context_window is both the per-request limit and the total
    # tokens (input + reserved output) that may be in flight across all requests.
    context_window: int = c.DEFAULT_CONTEXT_WINDOW
    max_input_ratio: float = c.MAX_INPUT_RATIO
    default_max_output_tokens: int = c.DEFAULT_MAX_OUTPUT_TOKENS
    budget_wait_timeout_s: float = c.DEFAULT_BUDGET_WAIT_TIMEOUT_S

    # Models: name -> SGLang server base URL (each SGLang server serves one model)
    model_endpoints: dict[str, str] = {c.DEFAULT_MODEL: c.DEFAULT_SGLANG_URL}
    default_model: str = c.DEFAULT_MODEL
    request_timeout_s: float = c.DEFAULT_REQUEST_TIMEOUT_S

    # RabbitMQ
    rabbitmq_enabled: bool = True
    rabbitmq_url: str = c.DEFAULT_RABBITMQ_URL
    rabbitmq_request_queue: str = c.DEFAULT_REQUEST_QUEUE
    rabbitmq_result_queue: str = c.DEFAULT_RESULT_QUEUE

    @property
    def max_input_tokens(self) -> int:
        return int(self.context_window * self.max_input_ratio + 1e-6)


@lru_cache
def get_settings() -> Settings:
    return Settings()
