"""Event-driven entrypoint: consume QueueRequest messages, publish results.

Backpressure: the consumer only pulls a message from RabbitMQ when token budget
is free, and holds at most one message while waiting for enough budget. Everything
else stays in the queue (in order) until earlier requests finish and release tokens.
"""
import asyncio
import json
import logging

import aio_pika
from pydantic import ValidationError

from ..constants import RABBITMQ_EMPTY_POLL_INTERVAL_S, STATUS_CHUNK, STATUS_ERROR, STATUS_OK
from ..errors import LLMServiceError
from ..schemas import QueueRequest
from ..service import LLMService, Plan
from ..settings import Settings

log = logging.getLogger(__name__)


class RabbitConsumer:
    def __init__(self, settings: Settings, service: LLMService):
        self._s = settings
        self._service = service
        self._conn = None
        self._channel = None
        self._queue = None
        self._loop_task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._conn = await aio_pika.connect_robust(self._s.rabbitmq_url)
        self._channel = await self._conn.channel()
        self._queue = await self._channel.declare_queue(self._s.rabbitmq_request_queue, durable=True)
        await self._channel.declare_queue(self._s.rabbitmq_result_queue, durable=True)
        self._loop_task = asyncio.create_task(self._consume_loop())
        log.info("Consuming from %s", self._s.rabbitmq_request_queue)

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        await asyncio.gather(*self._inflight, return_exceptions=True)
        if self._conn:
            await self._conn.close()  # unacked messages go back to the queue

    async def _consume_loop(self) -> None:
        budget = self._service.budget
        while True:
            await budget.wait_for_free()
            message = await self._queue.get(fail=False)  # None if queue is empty
            if message is None:
                await asyncio.sleep(RABBITMQ_EMPTY_POLL_INTERVAL_S)
                continue

            request_id = message.correlation_id
            try:
                req = QueueRequest.model_validate_json(message.body)
                request_id = req.request_id or request_id
                plan = self._service.plan(req)  # raises on bad model / token limit
            except ValidationError as e:
                await self._reply(message, self._error(request_id, "invalid_request", str(e)))
                continue
            except LLMServiceError as e:
                await self._reply(message, self._error(request_id, e.code, str(e)))
                continue

            await budget.acquire(plan.cost)  # waits here (head of line) if budget is short
            task = asyncio.create_task(self._run(message, req, plan, request_id))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _run(self, message, req: QueueRequest, plan: Plan, request_id) -> None:
        try:
            if req.stream:
                payload = await self._run_stream(message, req, plan, request_id)
            else:
                result = (await self._service.execute(plan, req)).model_dump()
                payload = {"request_id": request_id, "status": STATUS_OK, "result": result}
        except LLMServiceError as e:
            payload = self._error(request_id, e.code, str(e))
        except Exception as e:
            log.exception("Unhandled error")
            payload = self._error(request_id, "internal_error", str(e))
        finally:
            await self._service.budget.release(plan.cost)
        await self._reply(message, payload)

    async def _run_stream(self, message, req: QueueRequest, plan: Plan, request_id) -> dict:
        """Publish a chunk message per delta, return the final message (full content + usage)."""
        parts: list[str] = []
        final: dict = {}
        async for ev in self._service.stream_events(plan, req):
            if "delta" in ev:
                parts.append(ev["delta"])
                await self._publish(message, {"request_id": request_id, "status": STATUS_CHUNK,
                                              "seq": len(parts), "delta": ev["delta"]})
            else:
                final = ev
        return {"request_id": request_id, "status": STATUS_OK, "result": {
            "model": final["model"], "content": "".join(parts), "usage": final["usage"]}}

    async def _publish(self, message, payload: dict) -> None:
        routing_key = message.reply_to or self._s.rabbitmq_result_queue
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(payload).encode(),
                content_type="application/json",
                correlation_id=payload["request_id"],
            ),
            routing_key=routing_key,
        )

    async def _reply(self, message: aio_pika.abc.AbstractIncomingMessage, payload: dict) -> None:
        # Reply to reply_to if provided, else the shared result queue; always ack
        # (failures are reported on the result queue, not retried).
        await self._publish(message, payload)
        await message.ack()

    @staticmethod
    def _error(request_id, code: str, message: str) -> dict:
        return {"request_id": request_id, "status": STATUS_ERROR,
                "error": {"code": code, "message": message}}
