class LLMServiceError(Exception):
    status_code = 500
    code = "internal_error"


class TokenLimitExceeded(LLMServiceError):
    status_code = 422
    code = "token_limit_exceeded"

    def __init__(self, input_tokens: int, max_input_tokens: int):
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens
        super().__init__(
            f"Input is {input_tokens} tokens; maximum allowed is {max_input_tokens}"
        )


class UnknownModel(LLMServiceError):
    status_code = 404
    code = "unknown_model"

    def __init__(self, model: str, available: list[str]):
        super().__init__(f"Unknown model '{model}'. Available: {available}")


class BudgetTimeout(LLMServiceError):
    status_code = 503
    code = "token_budget_timeout"


class UpstreamError(LLMServiceError):
    status_code = 502
    code = "upstream_error"
