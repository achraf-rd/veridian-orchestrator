"""FastAPI surface — SSE chat stream, tool-approval resume, state.

/chat sends the user's message to the agent and streams what it does back:
plain replies, tool calls, tool results, and approval gates. /resume continues
past a tool-approval gate. This is the DECISION stream, distinct from the app's
per-agent telemetry SSE (Option A split planes).

Memory is automatic: the graph checkpoints its `messages` per conversation_id,
so each /chat restores the prior turns — the agent remembers the conversation.

Long tool calls (generate_test_cases, generate_xosc, occasionally
refine_requirements) can run for minutes with no state-update event in
between — from the graph's perspective a single node is just "in flight".
To avoid the stream going silent for that whole span, `_run` merges TWO
sources into one output: the graph's own state-update stream, and the
per-conversation MCP progress-notification queue (`progress_bus`) that
nodes publish into while their tool call is still running. A heartbeat
comment is sent if neither has produced anything for a while, so no
intermediary considers the connection dead.
"""

import asyncio
import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from app import perf, progress_bus

router = APIRouter()

HEARTBEAT_SECONDS = 15.0

# Tasks currently driving a graph run, keyed by conversation_id — lets /interrupt
# find and cancel an in-flight /chat or /resume without the caller needing a
# separate job id (the conversation IS the job, 1 run at a time per conversation).
_RUNNING: dict[str, asyncio.Task] = {}


class ChatIn(BaseModel):
    conversation_id: str
    message: str
    # LLM thinking effort for this turn, chosen in the composer (low | medium | high).
    reasoning: str = "low"


class ResumeIn(BaseModel):
    conversation_id: str
    decision: dict | None = None


class InterruptIn(BaseModel):
    conversation_id: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _run(graph, graph_input, config, conversation_id: str) -> AsyncIterator[str]:
    """Merge graph state-update events with live MCP progress ticks, as SSE,
    with idle heartbeats, until the next interrupt / completion / error / stop."""
    queue = progress_bus.get_queue(conversation_id)
    terminal: dict = {}

    async def drive_graph() -> None:
        try:
            async for chunk in graph.astream(graph_input, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    interrupts = chunk["__interrupt__"]
                    value = interrupts[0].value if interrupts else {}
                    payload = value if isinstance(value, dict) else {"value": value}
                    terminal["event"] = {"type": "gate", **payload}
                    await queue.put(terminal["event"])
                    return
                for update in chunk.values():
                    for event in (update or {}).get("log", []):
                        await queue.put(event)
            terminal["event"] = {"type": "done"}
            await queue.put(terminal["event"])
        except asyncio.CancelledError:
            raise  # a /interrupt cancelled us — the queue already got "interrupted"
        except Exception as exc:  # surface failures instead of a dead stream
            terminal["event"] = {"type": "error", "detail": str(exc)}
            await queue.put(terminal["event"])

    turn_t0 = time.perf_counter()
    task = asyncio.create_task(drive_graph())
    _RUNNING[conversation_id] = task
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield _sse(event)
            if event is terminal.get("event") or event.get("type") == "interrupted":
                perf.log("turn", (time.perf_counter() - turn_t0) * 1000,
                         conv=conversation_id[:8], kind=event.get("type"))
                break
    finally:
        if not task.done():
            task.cancel()
        if _RUNNING.get(conversation_id) is task:
            _RUNNING.pop(conversation_id, None)
        progress_bus.clear(conversation_id)


async def _discard_pending_turn(graph, config) -> None:
    """Clear a tool node the user cancelled mid-run WITHOUT re-running it: answer any
    dangling tool_call with a 'stopped' result and route the turn to END. Used when a
    Stop landed inside a tool (esp. an ungated one like refine, which would otherwise
    re-run on the next message)."""
    snap = await graph.aget_state(config)
    msgs = (snap.values or {}).get("messages", [])
    dangling: list = []
    for m in reversed(msgs):
        if m.get("role") == "tool":
            break  # last tool_call already answered — nothing dangling
        if m.get("role") == "assistant" and m.get("tool_calls"):
            dangling = m["tool_calls"]
            break
    cancelled = [
        {"role": "tool", "tool_call_id": tc["id"],
         "content": json.dumps({"cancelled": True, "reason": "stopped by user"})}
        for tc in dangling
    ]
    # Write as if `act` produced this → after_act sees route="end" → END. No re-run.
    await graph.aupdate_state(config, {"messages": cancelled, "route": "end"}, as_node="act")


@router.post("/chat")
async def chat(body: ChatIn, request: Request):
    graph = request.app.state.graph
    cid = body.conversation_id
    config = {"configurable": {"thread_id": cid}}

    # ONE run per conversation. A prior run still in flight (a stuck/long agent the
    # user didn't wait for, or rapid-fire messages) would execute CONCURRENTLY with
    # this turn on the SAME thread and lock/corrupt the checkpointer — which shows up
    # as a new message hanging on "Thinking…" forever. Cancel it first.
    prev = _RUNNING.pop(cid, None)
    if prev and not prev.done():
        prev.cancel()
    progress_bus.clear(cid)

    # The previous turn may be left pending — either paused at an approval gate, or
    # a tool node the user cancelled with Stop mid-run. Clear it before the new turn
    # so LangGraph doesn't RESUME it. Two cases need different handling:
    #  • at a gate (has interrupts): decline it cleanly (routes to END).
    #  • a cancelled tool node ("act", no interrupt): do NOT resume — resuming
    #    re-runs an ungated tool (the refiner re-triggering on the next message).
    #    Instead satisfy its dangling tool_call with a "stopped" result and end.
    # All time-boxed so a stuck checkpointer op can never hang the new turn.
    try:
        snapshot = await asyncio.wait_for(graph.aget_state(config), timeout=15)
        if snapshot.next:
            pending = list(snapshot.next)
            has_gate = any(getattr(t, "interrupts", None) for t in (getattr(snapshot, "tasks", None) or []))
            if "act" in pending and not has_gate:
                await asyncio.wait_for(_discard_pending_turn(graph, config), timeout=15)
            else:
                await asyncio.wait_for(
                    graph.ainvoke(Command(resume={"approved": False, "abandon": True}), config), timeout=30)
    except Exception:  # noqa: BLE001 — a slow/failed cleanup must not block the new turn
        pass
    finally:
        progress_bus.clear(cid)

    # The checkpointer restores prior messages for this thread; add_messages
    # appends this new user turn on top of them — that's the memory.
    graph_input = {
        "conversation_id": cid,
        "messages": [{"role": "user", "content": body.message}],
        "reasoning_effort": body.reasoning,
        "log": [],
    }
    return StreamingResponse(_run(graph, graph_input, config, cid), media_type="text/event-stream")


@router.post("/resume")
async def resume(body: ResumeIn, request: Request):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": body.conversation_id}}
    decision = body.decision if body.decision is not None else {"approved": True}
    return StreamingResponse(
        _run(graph, Command(resume=decision), config, body.conversation_id), media_type="text/event-stream"
    )


@router.post("/interrupt")
async def interrupt(body: InterruptIn):
    """Stop an in-flight /chat or /resume for this conversation, if any.

    Idempotent and safe to call even if nothing is running (e.g. the client
    fired it speculatively) — cancels the driving task and wakes the SSE loop
    so it emits `{"type": "interrupted"}` and closes, instead of hanging until
    the next heartbeat or timing out.
    """
    from app.graph import _ACTIVE_TOOL, STOPPABLE_TOOLS, TOOL_LABEL
    from app.mcp_client import gateway as mcp_gateway

    cid = body.conversation_id
    task = _RUNNING.get(cid)
    was_running = bool(task and not task.done())

    # Staged stop: FIRST halt the running agent upstream (Agents 2/3 expose a
    # /stop — Agent 1 refiner and Agent 4 executor don't), THEN cancel the
    # orchestrator's own loop (the LLM). The stop call uses its own short-lived
    # connection, so it doesn't block behind the in-flight generate call.
    tool = _ACTIVE_TOOL.get(cid)
    if was_running and tool in STOPPABLE_TOOLS:
        progress_bus.publish(cid, {"type": "stopping", "label": TOOL_LABEL.get(tool, tool)})
        try:
            await mcp_gateway.call_tool_json("stop_agent", {"agent": tool})
        except Exception:  # noqa: BLE001 — best-effort; cancel the loop regardless
            pass

    if was_running:
        task.cancel()
        # only publish if a run is actually listening — otherwise this would
        # create and orphan a fresh queue nothing will ever clear
        progress_bus.publish(cid, {"type": "interrupted"})
    return JSONResponse({"ok": True, "was_running": was_running})


@router.get("/prompts")
async def prompts(request: Request):
    """List the gateway's prompts — source for the composer slash commands."""
    gw = request.app.state.gateway
    items = await gw.list_prompts()
    out = []
    for p in items:
        out.append({
            "name": p.name,
            "description": getattr(p, "description", None),
            "arguments": [
                {"name": a.name, "required": getattr(a, "required", False)}
                for a in (getattr(p, "arguments", None) or [])
            ],
        })
    return JSONResponse(out)


@router.get("/runs/{conversation_id}")
async def run_state(conversation_id: str, request: Request):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": conversation_id}}
    snapshot = await graph.aget_state(config)
    return JSONResponse({
        "values": snapshot.values,
        "next": list(snapshot.next),
        "interrupted": bool(snapshot.next),
    })


@router.get("/history/{conversation_id}")
async def history(conversation_id: str, request: Request):
    """The persisted chat transcript for a conversation, so the UI can reload it
    after a refresh (the memory lives in the checkpointer, keyed by this id).
    Also reports whether a run is paused awaiting a tool approval, and on what."""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": conversation_id}}
    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    pending = None
    if snapshot.next:  # paused at an interrupt — surface the gate payload
        tasks = getattr(snapshot, "tasks", None) or []
        for t in tasks:
            for it in (getattr(t, "interrupts", None) or []):
                val = getattr(it, "value", None)
                if isinstance(val, dict):
                    pending = val
    return JSONResponse({
        "conversation_id": conversation_id,
        "messages": values.get("messages", []),
        "interrupted": bool(snapshot.next),
        "pending": pending,
    })


@router.get("/health")
async def health():
    return {"ok": True}
