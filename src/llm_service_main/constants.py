"""Fixed values and defaults. Anything overridable via environment lives in settings.py."""

ENV_PREFIX = "LLM_"
ENV_FILE = ".env"

# Token budget
DEFAULT_CONTEXT_WINDOW = 30_000
MAX_INPUT_RATIO = 2 / 3  # input may use at most 2/3 of the context window
CHARS_PER_TOKEN_FALLBACK = 4  # used when no tokenizer is available
TOKENIZER_CACHE_SIZE = 8
# Output tokens reserved for a request that doesn't set max_tokens
DEFAULT_MAX_OUTPUT_TOKENS = 1_000
# How long an HTTP request waits for token budget before giving up (RabbitMQ waits indefinitely)
DEFAULT_BUDGET_WAIT_TIMEOUT_S = 60.0
# How often the RabbitMQ consumer re-checks an empty queue
RABBITMQ_EMPTY_POLL_INTERVAL_S = 0.2

# Models
DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_SGLANG_URL = "http://localhost:30000"
SGLANG_CHAT_PATH = "/v1/chat/completions"
DEFAULT_REQUEST_TIMEOUT_S = 120.0

# Request validation
DEFAULT_TEMPERATURE = 0.7
MAX_TEMPERATURE = 2.0

# RabbitMQ
DEFAULT_RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
DEFAULT_REQUEST_QUEUE = "llm.requests"
DEFAULT_RESULT_QUEUE = "llm.results"

# Result statuses
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_CHUNK = "chunk"  # streaming partial result over RabbitMQ
