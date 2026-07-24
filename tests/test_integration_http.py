"""End-to-end HTTP check — real NIM + real Agent 1 through the gateway.

Requires BOTH services running:
  gateway:      (in veridian-mcp-gateway)  uv run python server.py
  orchestrator: (here)                     uv run python server.py

Asks the agent to analyse a requirement and asserts it runs ONLY Agent 1
(refine_requirements — not gated, so it runs straight away) and replies, with
nothing cascading to later stages. Then sends a follow-up on the SAME
conversation and asserts the agent still remembers it (checkpointed memory).

Run:  uv run python tests/test_integration_http.py
"""

import asyncio
import json
import sys

import httpx

ORCH = "http://127.0.0.1:8200"
CID = "itest-agent-1"
MESSAGE = (
    "Please analyse these requirements:\n"
    "The AEB shall stop the vehicle before a pedestrian crossing at 50 km/h.\n"
    "The AEB shall react within 300 ms of detection."
)


async def drain(c: httpx.AsyncClient, path: str, body: dict) -> list[dict]:
    events: list[dict] = []
    async with c.stream("POST", f"{ORCH}{path}", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            events.append(ev)
            label = ev.get("tool") or ev.get("agent") or ev.get("gate") or (ev.get("text", "")[:60])
            print(f"  SSE: {ev.get('type'):<10} {label}")
            if ev.get("type") in ("done", "error", "interrupted"):
                break
            if ev.get("type") == "gate" and not ev.get("result"):
                break
    return events


async def main() -> None:
    async with httpx.AsyncClient(timeout=180) as c:
        assert (await c.get(f"{ORCH}/health")).json()["ok"]

        events = await drain(c, "/chat", {"conversation_id": CID, "message": MESSAGE})
        types = [e["type"] for e in events]
        assert "tool_call" in types, types
        assert any(e["type"] == "tool_call" and e["tool"] == "refine_requirements" for e in events), events
        assert any(e["type"] == "agent" and e["agent"] == "refine_requirements" for e in events), events
        assert types[-1] == "done", types
        # granularity: it must NOT have run any later stage on its own
        assert not any(e.get("type") == "agent" and e.get("agent") in
                       ("generate_test_cases", "generate_xosc", "execute_simulation") for e in events), events
        assert any(e["type"] == "message" for e in events), "should end with a plain-language reply"
        print("  -> analysed with Agent 1 only, replied, no cascade")

        # memory: a follow-up on the same conversation
        follow = await drain(c, "/chat", {"conversation_id": CID,
                                          "message": "How many of those requirements were testable?"})
        assert any(e["type"] == "message" for e in follow), follow
        print("  -> follow-up answered (memory intact)")

        state = (await c.get(f"{ORCH}/history/{CID}")).json()
        assert len(state["messages"]) >= 4, state
        print(f"GET /history -> {len(state['messages'])} messages persisted")

    print("\nINTEGRATION HTTP TEST PASSED (real NIM + real Agent 1, single-agent + memory)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        print(f"INTEGRATION TEST FAILED: {exc}")
        sys.exit(1)
