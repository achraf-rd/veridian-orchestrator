"""Durability probe — does a conversation's memory survive an orchestrator restart?

Two phases around a process restart:
  uv run python tests/durability.py chat    # have a short exchange
  <restart the orchestrator process>
  uv run python tests/durability.py check   # is the history still there? => Postgres

If the checkpointer is Postgres, `check` finds the prior messages after restart;
with in-memory it comes back empty.
"""

import asyncio
import json
import sys

import httpx

ORCH = "http://127.0.0.1:8200"
CID = "durable-agent-1"
MSG = "Remember this codename: BLUEJAY. Just acknowledge it."


async def _drain(c: httpx.AsyncClient, path: str, body: dict) -> None:
    async with c.stream("POST", f"{ORCH}{path}", json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                print(f"  SSE: {ev.get('type')} {ev.get('tool') or ev.get('gate') or ''}")
                if ev.get("type") in ("done", "error", "interrupted"):
                    break


async def chat() -> None:
    async with httpx.AsyncClient(timeout=180) as c:
        await _drain(c, "/chat", {"conversation_id": CID, "message": MSG})
        st = (await c.get(f"{ORCH}/history/{CID}")).json()
        print(f"after chat: {len(st['messages'])} messages persisted")


async def check() -> None:
    async with httpx.AsyncClient(timeout=30) as c:
        st = (await c.get(f"{ORCH}/history/{CID}")).json()
        msgs = st["messages"]
        blob = " ".join(m.get("content") or "" for m in msgs)
        print(f"after restart: {len(msgs)} messages; codename present={'BLUEJAY' in blob}")
        if msgs and "BLUEJAY" in blob:
            print("DURABILITY OK — conversation memory survived restart (Postgres checkpointer)")
            sys.exit(0)
        print("DURABILITY NOT CONFIRMED — memory lost (in-memory / PG unavailable)")
        sys.exit(1)


asyncio.run(chat() if sys.argv[1] == "chat" else check())
