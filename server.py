"""Veridian Orchestrator — FastAPI app.

Lifespan: open the MCP gateway connection, build the checkpointer, compile the
graph. The graph's MCP tool calls are the control plane; live per-scenario
telemetry stays on the Next.js SSE routes.
"""

import asyncio
import sys
from contextlib import asynccontextmanager

# Windows default loop (Proactor) is incompatible with psycopg async (the
# Postgres checkpointer). Selector loop works for psycopg + httpx + uvicorn.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from fastapi import FastAPI

from app import perf
from app.checkpointer import make_checkpointer
from app.config import ORCH_HOST, ORCH_PORT
from app.graph import build
from app.mcp_client import gateway
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    perf.setup_logging()  # timing logs → the orchestrator terminal
    # The gateway connects lazily per call, so the orchestrator starts even if
    # the gateway (:8100) isn't up yet — it only errors when a tool is used.
    checkpointer, cm = await make_checkpointer()
    app.state.gateway = gateway
    app.state.graph = build(checkpointer)
    app.state._checkpointer_cm = cm
    try:
        yield
    finally:
        if cm is not None:
            await cm.__aexit__(None, None, None)


app = FastAPI(title="Veridian Orchestrator", version="0.1.0", lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    # loop="none" → uvicorn uses the loop we create here, so the Selector policy
    # set above actually sticks (uvicorn.run() would otherwise reinstall Proactor).
    config = uvicorn.Config(app, host=ORCH_HOST, port=ORCH_PORT, loop="none")
    asyncio.run(uvicorn.Server(config).serve())
