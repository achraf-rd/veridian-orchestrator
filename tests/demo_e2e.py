"""End-to-end feature demo — drives the agent like the UI would.

  uv run python tests/demo_e2e.py analyse   # fast: analyse a requirement (Agent 1 only)
  uv run python tests/demo_e2e.py full      # slower: walk the pipeline, auto-approving gates

Requires gateway (:8100) + orchestrator (:8200) running.
"""

import asyncio
import json
import sys
import time

import httpx

ORCH = "http://127.0.0.1:8200"


async def stream(client: httpx.AsyncClient, path: str, body: dict) -> dict | None:
    """Print the decision stream; return the gate pause event if one occurred."""
    paused = None
    async with client.stream("POST", f"{ORCH}{path}", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev.get("type")
            if t == "message":
                print(f"    assistant: {ev.get('text', '')[:120]}")
            elif t == "tool_call":
                print(f"    → calling {ev.get('label') or ev.get('tool')}")
            elif t == "agent":
                extra = ev.get("verdict") or ev.get("total_scenarios") or ev.get("passed") or ""
                print(f"    ✓ {ev.get('agent')} {extra}")
            elif t == "gate" and not ev.get("result"):
                print(f"    ⏸ approval needed: {ev.get('labels') or ev.get('tools')}")
                paused = ev
            if t in ("done", "error", "interrupted"):
                break
            if t == "gate" and not ev.get("result"):
                break
    return paused


async def analyse() -> None:
    cid = f"demo-analyse-{int(time.time())}"
    print(f"=== ANALYSE (Agent 1 only)  (conv {cid}) ===")
    async with httpx.AsyncClient(timeout=180) as c:
        await stream(c, "/chat", {"conversation_id": cid, "message":
            "Analyse this requirement: The AEB shall stop the vehicle before a pedestrian "
            "crossing at 50 km/h."})
        print("=== done — only Agent 1 should have run ===")


async def full() -> None:
    cid = f"demo-full-{int(time.time())}"
    print(f"=== FULL PIPELINE (agent walks the stages)  (conv {cid}) ===")
    async with httpx.AsyncClient(timeout=900) as c:
        paused = await stream(c, "/chat", {"conversation_id": cid, "message":
            "Run the complete validation pipeline for this requirement and give me the final "
            "report: The AEB shall stop the vehicle before a pedestrian crossing at 50 km/h."})
        gates = 0
        while paused:
            gates += 1
            print(f"  >> auto-approve {paused.get('tools')}")
            paused = await stream(c, "/resume", {"conversation_id": cid, "decision": {"approved": True}})
        st = (await c.get(f"{ORCH}/runs/{cid}")).json()
        print(f"=== COMPLETE — {gates} approvals, report={st['values'].get('report')} ===")


mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
asyncio.run(full() if mode == "full" else analyse())
