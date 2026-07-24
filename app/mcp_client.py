"""MCP host/client — the orchestrator's connection to the agent gateway.

Opens a short-lived FastMCP client per operation (Streamable HTTP). Per-call
connect keeps the orchestrator resilient: it starts even if the gateway is down,
and survives gateway restarts — a call only fails if the gateway is unreachable
at the moment it's actually used. Tool bodies return JSON strings; call_tool_json
parses them.
"""

import json
import time
from typing import Any, Awaitable, Callable

from fastmcp import Client

from app import perf
from app.config import MCP_GATEWAY_URL

OnProgress = Callable[[float, float | None, str | None], Awaitable[None]]


def _parse(result: Any) -> dict:
    """Coerce a CallToolResult into a dict (tools return JSON strings)."""
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"items": data}
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}
    return {}


class Gateway:
    def __init__(self, url: str):
        self._url = url

    async def call_tool_json(self, name: str, args: dict, on_progress: OnProgress | None = None) -> dict:
        t0 = time.perf_counter()
        async with Client(self._url, progress_handler=on_progress) as client:
            connect_ms = round((time.perf_counter() - t0) * 1000)
            result = await client.call_tool(name, args, progress_handler=on_progress)
        perf.log(f"mcp:{name}", (time.perf_counter() - t0) * 1000, connect=f"{connect_ms}ms")
        return _parse(result)

    async def list_prompts(self) -> list:
        async with Client(self._url) as client:
            return await client.list_prompts()

    async def get_prompt(self, name: str, args: dict) -> Any:
        async with Client(self._url) as client:
            return await client.get_prompt(name, args)

    async def read_resource(self, uri: str) -> Any:
        async with Client(self._url) as client:
            return await client.read_resource(uri)


gateway = Gateway(MCP_GATEWAY_URL)
