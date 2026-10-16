import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.routes import router
from .settings import get_settings
from .errors import LLMServiceError
from .messaging.rabbitmq import RabbitConsumer
from .service import LLMService
from .sglang_client import SGLangClient

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = SGLangClient(settings)
    service = LLMService(settings, client)
    app.state.settings, app.state.service = settings, service

    consumer = None
    if settings.rabbitmq_enabled:
        consumer = RabbitConsumer(settings, service)
        await consumer.start()
    yield
    if consumer:
        await consumer.stop()
    await client.aclose()


app = FastAPI(title="llm-service", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(LLMServiceError)
async def service_error_handler(_: Request, exc: LLMServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": str(exc)}},
    )
