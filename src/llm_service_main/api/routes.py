import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas import GenerateRequest, GenerateResponse

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/models")
async def models(request: Request):
    s = request.app.state.settings
    return {"default": s.default_model, "models": sorted(s.model_endpoints)}


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    """With "stream": true the reply is Server-Sent Events instead of one JSON body."""
    service = request.app.state.service
    if not req.stream:
        return await service.generate(req)

    events = service.stream(req)
    # Pull the first event before sending headers so validation, budget-timeout and
    # upstream errors still come back as proper HTTP errors rather than a broken stream.
    first = await anext(events)

    def frame(ev: dict) -> str:
        return "data: " + json.dumps(ev) + "\n\n"

    async def sse():
        try:
            yield frame(first)
            async for ev in events:
                yield frame(ev)
        finally:
            await events.aclose()  # releases the token budget on client disconnect

    return StreamingResponse(sse(), media_type="text/event-stream")
