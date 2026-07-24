"""Per-conversation progress fan-out.

Long MCP tool calls (generate_test_cases, generate_xosc) report progress via
MCP's notifications/progress while the underlying agent call is still open.
A node's coroutine can't yield mid-execution back to the SSE response, so we
publish those ticks onto a queue the /chat SSE loop drains concurrently with
the graph's own state-update stream — two producers, one output stream.
"""

import asyncio

_queues: dict[str, asyncio.Queue] = {}


def get_queue(conversation_id: str) -> asyncio.Queue:
    q = _queues.get(conversation_id)
    if q is None:
        q = asyncio.Queue()
        _queues[conversation_id] = q
    return q


def publish(conversation_id: str, event: dict) -> None:
    get_queue(conversation_id).put_nowait(event)


def clear(conversation_id: str) -> None:
    _queues.pop(conversation_id, None)
