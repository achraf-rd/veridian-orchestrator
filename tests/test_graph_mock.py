"""Agent graph mechanics — offline & deterministic (LLM + gateway stubbed).

Proves the ReAct tool-calling agent independent of network/agents:
  1. a plain message → a plain reply, no tool call.
  2. "analyse this requirement" → runs ONLY Agent 1, then replies (no cascade).
  3. a gated tool (generate_test_cases) pauses for approval, resumes, runs.
  4. rejecting the gate declines the tool without running it.
  5. memory: a second /chat turn on the same thread sees the first turn.

Run:  uv run python tests/test_graph_mock.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

import app.graph as gmod  # noqa: E402


# ── stub LLM: scripted STREAMING responses (tool call, or final text) ───────
# n_agent uses stream=True, so create() returns an async iterator of chunks
# whose choices[0].delta carries either content or tool_call fragments.

class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Chunk:
    def __init__(self, delta): self.choices = [type("C", (), {"delta": delta})()]


class _TCDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index, self.id = index, id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


async def _stream_text(text):
    # a couple of content chunks, to exercise accumulation
    mid = max(1, len(text) // 2)
    for part in (text[:mid], text[mid:]):
        yield _Chunk(_Delta(content=part))


async def _stream_tool(name, args):
    yield _Chunk(_Delta(tool_calls=[_TCDelta(0, id="call_1", name=name)]))
    yield _Chunk(_Delta(tool_calls=[_TCDelta(0, arguments=json.dumps(args))]))


class FakeLLM:
    """Pops a scripted response per agent step. Each script item is either
    ("tool", name, args_dict) or ("text", reply)."""
    def __init__(self):
        self.script: list = []
        self.seen: list = []  # message lists passed in, for the memory assertion

    class _Completions:
        def __init__(self, outer): self.outer = outer

        async def create(self, *, messages, **kwargs):
            self.outer.seen.append(messages)
            kind, *rest = self.outer.script.pop(0)
            if kind == "tool":
                name, args = rest
                return _stream_tool(name, args)
            return _stream_text(rest[0])

    @property
    def chat(self):
        outer = self
        return type("Chat", (), {"completions": FakeLLM._Completions(outer)})()


class FakeGateway:
    async def call_tool_json(self, name, args, on_progress=None):
        if name == "refine_requirements":
            return {"refining_id": "ref-1", "feature": "AEB", "summary": {"total_testable": 1},
                    "testable": [{"id": "req-001", "original": "r1", "complexity": "HIGH",
                                  "conflict_flag": False, "num_scenarios": 2, "overlap_with": []}]}
        if name == "generate_test_cases":
            return {"job_id": "j2", "total_scenarios": 2,
                    "scenarios": [{"scenario_id": "TC-1", "feature_under_test": "AEB", "test_phase": "SIL",
                                   "sil_section": {"preconditions": [], "steps": []}},
                                  {"scenario_id": "TC-2", "feature_under_test": "AEB", "test_phase": "SIL",
                                   "sil_section": {"preconditions": [], "steps": []}}]}
        raise AssertionError(f"unexpected tool {name}")


def _events(chunks):
    out = []
    for chunk in chunks:
        if "__interrupt__" in chunk:
            out.append(("gate", chunk["__interrupt__"][0].value))
        else:
            for update in chunk.values():
                for ev in (update or {}).get("log", []):
                    out.append((ev.get("type"), ev))
    return out


async def _drain(graph, gi, cfg):
    return [c async for c in graph.astream(gi, cfg, stream_mode="updates")]


def _user(msg): return {"messages": [{"role": "user", "content": msg}], "log": []}


async def test_plain_reply(graph, llm):
    llm.script = [("text", "Hi! I'm Veridian's assistant. Give me a requirement to analyse.")]
    cfg = {"configurable": {"thread_id": "c-plain"}}
    ev = _events(await _drain(graph, {"conversation_id": "c-plain", **_user("hello who are you")}, cfg))
    types = [t for t, _ in ev]
    assert types == ["message"], types
    snap = await graph.aget_state(cfg)
    assert not snap.next and "refinement" not in snap.values, "no tool should have run"
    print(f"  plain reply -> {types} (no tool)")


async def test_only_agent1(graph, llm):
    # user asks to analyse → LLM calls refine only, then replies. No cascade.
    llm.script = [("tool", "refine_requirements", {"requirements": ["The AEB shall stop at 50 km/h."]}),
                  ("text", "Analysed: 1 testable requirement (req-001, HIGH).")]
    cfg = {"configurable": {"thread_id": "c-a1"}}
    ev = _events(await _drain(graph, {"conversation_id": "c-a1", **_user("analyse this: The AEB shall stop at 50 km/h.")}, cfg))
    types = [t for t, _ in ev]
    assert types == ["tool_call", "agent", "card", "message"], types
    card = next(e for t, e in ev if t == "card")
    assert card["stage"] == "nlp" and card["result"].get("refining_id") == "ref-1", card
    snap = await graph.aget_state(cfg)
    assert snap.values.get("refinement", {}).get("refining_id") == "ref-1"
    assert "test_cases" not in snap.values, "must NOT have cascaded to Agent 2"
    assert not snap.next
    print(f"  analyse -> {types} (only Agent 1 ran, no cascade, emits nlp card)")


async def test_gated_tool_pauses_and_resumes(graph, llm):
    cfg = {"configurable": {"thread_id": "c-gate"}}
    # turn 1: refine (memory seed)
    llm.script = [("tool", "refine_requirements", {"requirements": ["r"]}), ("text", "Refined.")]
    await _drain(graph, {"conversation_id": "c-gate", **_user("refine: r")}, cfg)
    # turn 2: ask to generate test cases → gated → pause
    llm.script = [("tool", "generate_test_cases", {}), ("text", "Generated 2 test cases.")]
    ev = _events(await _drain(graph, {"conversation_id": "c-gate", **_user("now generate the test cases")}, cfg))
    assert ev[-1][0] == "gate" and ev[-1][1]["gate"] == "tool", ev
    assert "generate_test_cases" in ev[-1][1]["tools"], ev
    snap = await graph.aget_state(cfg)
    assert snap.next, "should be paused at the tool-approval gate"
    assert "test_cases" not in snap.values, "must not run before approval"
    print(f"  gated tool -> paused awaiting approval on {ev[-1][1]['tools']}")
    # approve → runs
    ev = _events(await _drain(graph, Command(resume={"approved": True}), cfg))
    types = [t for t, _ in ev]
    assert "agent" in types and types[-1] == "message", types
    scen_card = next((e for t, e in ev if t == "card" and e["stage"] == "scenario"), None)
    assert scen_card and len(scen_card["result"]["testCases"]) == 2, ev
    snap = await graph.aget_state(cfg)
    assert len(snap.values.get("test_cases", [])) == 2 and not snap.next
    print(f"  approved -> {types} (2 test cases, done, emits scenario card)")


async def test_gate_rejection(graph, llm):
    cfg = {"configurable": {"thread_id": "c-reject"}}
    llm.script = [("tool", "refine_requirements", {"requirements": ["r"]}), ("text", "Refined.")]
    await _drain(graph, {"conversation_id": "c-reject", **_user("refine: r")}, cfg)
    llm.script = [("tool", "generate_test_cases", {}), ("text", "Okay, I won't run it.")]
    await _drain(graph, {"conversation_id": "c-reject", **_user("generate test cases")}, cfg)
    ev = _events(await _drain(graph, Command(resume={"approved": False}), cfg))
    assert any(t == "gate" and e.get("result") == "rejected" for t, e in ev), ev
    snap = await graph.aget_state(cfg)
    assert "test_cases" not in snap.values and not snap.next
    print("  rejected gate -> tool declined, nothing generated")


async def test_abandon_gate(graph, llm):
    """Regression: the user answers a pending gate by sending a NEW message
    instead of approving/rejecting. The pending tool must be abandoned (route to
    END, no agent re-proposal) and must NOT run on the following turn — the bug
    where a stopped Agent 2 later ran on its own."""
    cfg = {"configurable": {"thread_id": "c-abandon"}}
    llm.script = [("tool", "refine_requirements", {"requirements": ["r"]}), ("text", "Refined.")]
    await _drain(graph, {"conversation_id": "c-abandon", **_user("refine: r")}, cfg)
    # ask to generate → gated → pauses awaiting approval
    llm.script = [("tool", "generate_test_cases", {}), ("text", "unused")]
    await _drain(graph, {"conversation_id": "c-abandon", **_user("generate test cases")}, cfg)
    assert (await graph.aget_state(cfg)).next, "should be paused at the gate"
    # user moves on → the /chat guard flushes the pending gate with abandon=True
    await graph.ainvoke(Command(resume={"approved": False, "abandon": True}), cfg)
    snap = await graph.aget_state(cfg)
    assert not snap.next, "abandon must clear the pending interrupt"
    assert "test_cases" not in snap.values, "abandoned tool must NOT have run"
    # the new message now runs fresh — a plain greeting, no tool re-proposal
    llm.script = [("text", "Hello! How can I help?")]
    ev = _events(await _drain(graph, {"conversation_id": "c-abandon", **_user("hi")}, cfg))
    types = [t for t, _ in ev]
    assert types == ["message"], types
    snap = await graph.aget_state(cfg)
    assert "test_cases" not in snap.values and not snap.next, "abandoned Agent 2 must not run later"
    print("  abandoned gate -> pending tool dropped, does not run on the next turn")


async def test_discard_cancelled_tool(graph, llm):
    """Regression: a tool node the user cancelled mid-run (Stop) must NOT re-run on
    the next message — the refiner re-triggering bug. _discard_pending_turn should
    clear the pending 'act' node without executing the tool."""
    from app.routes import _discard_pending_turn
    cfg = {"configurable": {"thread_id": "c-discard"}}
    # Simulate the checkpoint left after a Stop mid-refine: the agent asked for
    # refine_requirements but 'act' never ran (next=['act'], tool_call unanswered).
    await graph.aupdate_state(cfg, {
        "conversation_id": "c-discard",
        "messages": [
            {"role": "user", "content": "refine: r"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_x", "type": "function",
                 "function": {"name": "refine_requirements", "arguments": "{}"}},
            ]},
        ],
    }, as_node="agent")
    assert "act" in list((await graph.aget_state(cfg)).next), "setup: should be pending at act"
    # Discard it — must clear the pending node WITHOUT running refine.
    await _discard_pending_turn(graph, cfg)
    snap = await graph.aget_state(cfg)
    assert not snap.next, f"still pending after discard: {snap.next}"
    assert "refinement" not in snap.values, "refine must NOT have run on discard"
    assert snap.values["messages"][-1]["role"] == "tool", "dangling tool_call must be answered"
    print("  discard -> cancelled tool node cleared, refine did not re-run")


async def test_trim_file_blocks(graph, llm):
    """Old file dumps are stripped from history (context bloat / hang); the current
    turn's file is kept so a just-attached file is still readable."""
    from app.graph import _trim_file_blocks
    msgs = [
        {"role": "user", "content": "analyse this\n\n--- Attached file: r.docx ---\nREQ-1 huge text here"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now do X\n\n--- Attached file: r2.docx ---\nCURRENT huge text"},
    ]
    out = _trim_file_blocks(msgs)
    assert "REQ-1 huge text" not in out[0]["content"] and "already read" in out[0]["content"], out[0]
    assert "analyse this" in out[0]["content"], "user's instruction is kept"
    assert "CURRENT huge text" in out[2]["content"], "current turn's file must be kept"
    print("  trim -> old file text dropped, current file + instruction kept")


async def test_memory(graph, llm):
    cfg = {"configurable": {"thread_id": "c-mem"}}
    llm.script = [("text", "Nice to meet you, Sam.")]
    await _drain(graph, {"conversation_id": "c-mem", **_user("my name is Sam")}, cfg)
    llm.script = [("text", "Your name is Sam.")]
    await _drain(graph, {"conversation_id": "c-mem", **_user("what's my name?")}, cfg)
    # the LLM on the 2nd turn must have been handed the 1st turn in its messages
    last_call_messages = llm.seen[-1]
    contents = " ".join(m.get("content") or "" for m in last_call_messages)
    assert "my name is Sam" in contents, "second turn did not see the first — memory broken"
    snap = await graph.aget_state(cfg)
    roles = [m["role"] for m in snap.values["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"], roles
    print("  memory -> second turn saw the first (checkpointed messages)")


async def main():
    llm = FakeLLM()
    gmod.client = llm
    gmod.gateway = FakeGateway()
    graph = gmod.build(MemorySaver())

    print("test_plain_reply:"); await test_plain_reply(graph, llm)
    print("test_only_agent1:"); await test_only_agent1(graph, llm)
    print("test_gated_tool:"); await test_gated_tool_pauses_and_resumes(graph, llm)
    print("test_gate_rejection:"); await test_gate_rejection(graph, llm)
    print("test_abandon_gate:"); await test_abandon_gate(graph, llm)
    print("test_discard_cancelled_tool:"); await test_discard_cancelled_tool(graph, llm)
    print("test_trim_file_blocks:"); await test_trim_file_blocks(graph, llm)
    print("test_memory:"); await test_memory(graph, llm)
    print("\nAGENT GRAPH MOCK TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
