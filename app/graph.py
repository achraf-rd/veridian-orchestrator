"""The orchestration agent — a ReAct tool-calling loop with durable memory.

This is a genuine conversational agent (like a chat assistant that can call
tools), NOT a fixed classify-then-cascade pipeline. The LLM sees the whole
conversation and the 5 Veridian agents as tools, and calls exactly the one the
user asks for — "analyse this requirement" runs ONLY Agent 1 and replies with
its output; nothing auto-cascades. Ask it to "run the whole thing" and it walks
the stages itself, pausing for approval before each consequential one.

Memory: the graph state carries the running `messages` list; the compiled
checkpointer (Postgres) snapshots it per conversation_id, so a later /chat on
the same conversation restores the full history — the LLM remembers. The
accumulated agent artifacts (refinement, test_cases, execution, ...) are
checkpointed the same way, so a follow-up turn can act on an earlier stage's
output without re-running it.

Loop:  START -> agent -> (tool call?) -> act -> agent -> ... -> (plain reply) -> END
The `act` node pauses via interrupt() before running a GATED tool (the
long/consequential generation + execution stages); quick read-y tools
(refine, evaluate) run without a gate so an "analyse this" request is instant.
"""

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app import progress_bus, transforms, perf
from app.llm import MODEL, client
from app.mcp_client import gateway


_REASONING_LEVELS = {"low", "medium", "high"}

# Tool currently running per conversation — lets /interrupt halt the right agent
# upstream. Agents 1/2/3 expose a cancel; the executor (Agent 4) does not.
_ACTIVE_TOOL: dict[str, str] = {}
STOPPABLE_TOOLS = {"refine_requirements", "generate_test_cases", "generate_xosc"}


class State(TypedDict, total=False):
    conversation_id: str
    reasoning_effort: str                     # LLM thinking effort (low|medium|high), set per /chat turn
    messages: Annotated[list, operator.add]   # OpenAI-format chat history (the memory)
    # accumulated agent artifacts — checkpointed, so later turns build on them:
    refinement: dict
    agent2_result: dict
    test_cases: list
    specs: list
    xosc: dict
    execution: dict
    report: dict
    log: Annotated[list, operator.add]        # per-turn SSE decision events (delta only)
    route: str                                # transient: where `act` goes next ("agent" | "end")


# ── the agent's tools (thin, LLM-facing) ────────────────────────────────────
# The LLM only decides WHICH agent to run; the `act` node builds each agent's
# real (structured) arguments from prior results in state, so the model never
# has to hand-craft test-case / scenario payloads it would get wrong.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "refine_requirements",
            "description": (
                "Run Agent 1 (Requirements Refiner) on raw natural-language ADAS requirements. "
                "Classifies each as testable/incomplete/duplicate, flags conflicts/overlaps, "
                "assigns complexity. Call this ONLY when the user has actually provided requirement "
                "statements (pasted text or an attached requirements file), and pass those exact "
                "statements. NEVER invent requirements, and never pass the user's conversational "
                "message (e.g. 'analyse my file', 'here are my requirements') as a requirement. If "
                "no requirements are present yet, don't call this — ask the user for them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The exact requirement statements the user provided, one per array item. Real requirement text only — never chat/instructions, never invented.",
                    }
                },
                "required": ["requirements"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_test_cases",
            "description": (
                "Run Agent 2 (Test Case Generator) on refined requirements. Produces structured "
                "SIL/HIL test cases. Normally uses the requirements refined earlier in this "
                "conversation — call with NO args for that. To generate from only SOME of them, "
                "pass `requirement_ids` (e.g. only the testable ones). Requires a prior "
                "refine_requirements — do not call for raw text. Long-running; pauses for approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requirement_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Only generate from these refined requirement ids. Omit to use all testable ones.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_xosc",
            "description": (
                "Run Agent 3 (XOSC File Generator) to produce OpenSCENARIO .xosc files. Ways to "
                "supply the work: NO args builds .xosc for every test case generated earlier; "
                "`scenario_ids` builds only a subset (or one); `overrides` is a map of scenario_id "
                "-> a free-form patch deep-merged into that scenario's spec before generation, to "
                "tweak a value and regenerate. E.g. re-run TC-req-001-01 at 20 km/h: "
                "scenario_ids=['TC-req-001-01'], overrides={'TC-req-001-01': {'ego_vehicle': "
                "{'initial_speed': 20}}}. You may also pass a full `specs` array to build directly "
                "from scenario details the user typed out, with no prior test cases. Long-running; "
                "pauses for approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Build .xosc for only these test-case ids. Omit to build all.",
                    },
                    "overrides": {
                        "type": "object",
                        "description": "Optional. Map of scenario_id -> partial spec patch (deep-merged) to tweak fields like ego_vehicle.initial_speed, environment.weather, actors, before regenerating.",
                    },
                    "specs": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional. Full ScenarioSpec objects to build directly (use when the user gives scenario details themselves and there are no prior test cases). Takes precedence over state-derived specs.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_simulation",
            "description": (
                "Run Agent 4 (Execution) — run generated .xosc scenarios in eSMini (real executor). "
                "Call with NO args to run every scenario generated earlier, or pass `scenario_ids` "
                "to run only a subset / re-run a single one. Returns per-scenario metrics (min "
                "inter-vehicle distance) + a replay recording; it does NOT decide pass/fail (Agent 5 "
                "does). Requires a prior generate_xosc. Long-running; pauses for approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Run only these scenario ids (reusing their already-generated .xosc). Omit to run all.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_results",
            "description": (
                "Run Agent 5 (Report Generator) — score the last execution into a KPI report "
                "with a pass/fail verdict. Uses the execution in this conversation. Requires a "
                "prior execution."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Consequential tools pause for the user's OK before running; the quick read-y
# ones (refine, evaluate) just run so a plain "analyse this" is friction-free.
GATED_TOOLS = {"generate_test_cases", "generate_xosc", "execute_simulation"}

TOOL_LABEL = {
    "refine_requirements": "Agent 1 · Requirements Refiner",
    "generate_test_cases": "Agent 2 · Test Case Generator",
    "generate_xosc": "Agent 3 · XOSC File Generator",
    "execute_simulation": "Agent 4 · Execution",
    "evaluate_results": "Agent 5 · Report Generator",
}

SYSTEM = (
    "You are Veridian's assistant for ADAS validation. Chat normally; call a tool only when "
    "the user asks for that action. The five agents are tools you compose freely — you are NOT "
    "a fixed pipeline:\n"
    "NEVER fabricate a tool's input. Only call refine_requirements when the user has actually "
    "given you requirement text (pasted statements or an attached requirements file), and pass "
    "those exact statements. If the user only SAYS they'll provide a file/requirements but none "
    "are present yet, ask them to paste or attach the requirements and STOP — do not call any "
    "tool, and never turn their chat message (e.g. 'analyse my file') into a requirement.\n"
    "ATTACHED FILES: a message may embed a file's extracted text, delimited by a line "
    "'--- Attached file: NAME ---'. It could be anything — requirements, test cases, a spec, notes. "
    "If the user typed an instruction alongside the file, follow it. If they attached a file with NO "
    "instruction, DO NOT call ANY tool — not refine_requirements, not anything. Just read it and reply "
    "with one or two short sentences: say what the file appears to contain and ASK what they'd like to "
    "do with it (e.g. 'This looks like 3 ADAS requirements. Want me to refine and validate them?'). "
    "Then WAIT for their answer — do not run a tool until they say so. Never echo the file's contents "
    "back — the UI already shows a file card.\n"
    "  refine_requirements (1) → generate_test_cases (2) → generate_xosc (3) → "
    "execute_simulation (4) → evaluate_results (5).\n"
    "Run EXACTLY the agents the user asked for, in the order implied, and STOP there — never "
    "run a later stage on your own. Examples: 'just make the XOSC from these requirements' = run "
    "1, 2, 3 then stop; 'validate these' = run 1 then stop; 'run the whole thing' = walk all five. "
    "Each stage needs the previous one's output; it's kept in this conversation, so you can pick up "
    "where you left off (e.g. 'now just run them' → execute_simulation using the .xosc already made).\n"
    "You can target a SUBSET and TWEAK parameters — you don't have to redo everything:\n"
    "  • generate_test_cases(requirement_ids=[...]) — only some refined requirements (e.g. only the "
    "testable ones).\n"
    "  • generate_xosc(scenario_ids=[...], overrides={id: {<patch>}}) — rebuild only some scenarios, "
    "deep-merging a patch first. To re-run TC-req-001-01 at 20 km/h: scenario_ids=['TC-req-001-01'], "
    "overrides={'TC-req-001-01': {'ego_vehicle': {'initial_speed': 20}}}. You can override any spec "
    "field (ego_vehicle, environment.weather/lighting, actors, ...).\n"
    "  • generate_xosc(specs=[...]) — build .xosc directly from scenario details the user typed out, "
    "with no prior refine/test-case step.\n"
    "  • execute_simulation(scenario_ids=[...]) — run or re-run only some scenarios.\n"
    "MANDATORY SCOPING: when the user names specific scenario(s) to re-run or change — e.g. 'rerun "
    "TC-req-001-01', 'repeat that test case at 20 km/h', 'redo scenario 3 in rain' — you MUST pass "
    "scenario_ids for EXACTLY those ids (plus overrides for any changed value). NEVER call "
    "generate_xosc or execute_simulation with no scenario_ids in that case: calling with no args "
    "rebuilds/runs EVERY scenario, which is wrong and wastes the user's time. Confirm the id you're "
    "acting on if it's ambiguous, but once it's known, scope to it. Omit scenario_ids ONLY when the "
    "user clearly means the whole set ('run them all', 'generate all the XOSC').\n"
    "If a request needs a field or capability the agents don't have (e.g. a physics parameter Agent 3 "
    "doesn't model), SAY SO plainly and stop — do NOT promise it and pause waiting. If refinement "
    "marks a requirement incomplete, tell the user exactly what to add — you can't make test cases "
    "from it.\n"
    "IMPORTANT: after a tool runs, the UI already shows a detailed result card for that stage "
    "(metrics, tables, per-item lists, and a 'Review' link). So DO NOT repeat the results — no "
    "markdown tables, no per-requirement or per-test-case lists, no full 'Refinement Result' dumps. "
    "Reply with ONE short conversational sentence (at most two) that acknowledges what happened and "
    "points to the card, e.g. \"Done — refined your requirements. Open the card above to review, then "
    "tell me when to generate the test cases.\" You remember the whole conversation."
)


# ── result → compact summary (keeps the chat history lean) ───────────────────

def _summarize_tool_result(name: str, result: dict) -> dict:
    """A small projection of an agent result for the LLM to reason over — the full
    payload (which can be tens of thousands of tokens) stays in state, not chat."""
    if name == "refine_requirements":
        testable = transforms.testable_requirements(result)
        return {
            "summary": result.get("summary", {}),
            "testable": [{"id": r.get("id"), "original": r.get("original"),
                          "complexity": r.get("complexity"), "num_scenarios": r.get("num_scenarios")}
                         for r in testable],
            "incomplete": [{"id": r.get("id"), "issues": r.get("issues_found", [])}
                           for r in (result.get("incomplete") or [])],
        }
    if name == "generate_test_cases":
        tcs = result.get("scenarios", [])
        return {"total_scenarios": result.get("total_scenarios", len(tcs)),
                "test_cases": [{"id": tc.get("scenario_id"), "feature": tc.get("feature_under_test"),
                                "phase": tc.get("test_phase")} for tc in tcs]}
    if name == "generate_xosc":
        return {"summary": result.get("summary", {})}
    if name == "execute_simulation":
        return {
            "total": result.get("total"), "passed": result.get("passed"), "failed": result.get("failed"),
            "runs": [{"id": r.get("scenario_id"), "min_distance_m": r.get("min_distance_m"),
                      "ran": bool(r.get("run_name")) and not r.get("error"), "error": r.get("error")}
                     for r in result.get("runs", [])],
        }
    if name == "evaluate_results":
        return {k: result.get(k) for k in ("verdict", "score") if k in result}
    return result


# ── tool execution — build real args from state, call the gateway ────────────

def _filter_specs(specs: list[dict], scenario_ids: list[str] | None, overrides: dict | None) -> list[dict]:
    """Filter stored specs to a subset and deep-merge per-scenario overrides — the
    path used when we regenerate from previously-built specs (no test cases)."""
    wanted = set(scenario_ids) if scenario_ids else None
    out = []
    for s in specs:
        sid = s.get("scenario_id")
        if wanted is not None and sid not in wanted:
            continue
        out.append(transforms.deep_merge(s, overrides[sid]) if overrides and sid in overrides else s)
    return out


def _merge_by_id(prev: list[dict], new: list[dict], id_key: str) -> list[dict]:
    """Overlay `new` items onto `prev` by id (update matching, keep the rest, append
    brand-new). Preserves prev order. Used to fold a partial re-run back into the
    full stage result so state keeps the whole picture."""
    new_by = {i.get(id_key): i for i in new}
    merged = [new_by.pop(i.get(id_key), i) for i in prev]
    merged.extend(new_by.values())
    return merged


def _merge_xosc(prev: dict | None, new: dict) -> dict:
    """Fold a partial generate_xosc result into the prior full xosc state by
    scenario_id and recompute the summary counts."""
    prev = prev or {}
    summary = dict(prev.get("summary") or {})
    merged = _merge_by_id(summary.get("scenarios") or [], (new.get("summary") or {}).get("scenarios") or [], "scenario_id")
    successful = sum(1 for s in merged if s.get("status") == "success")
    fallback = sum(1 for s in merged if s.get("status") == "fallback")
    failed = sum(1 for s in merged if s.get("status") == "failed")
    ok = successful + fallback
    summary.update({
        "scenarios": merged, "total_scenarios": len(merged),
        "successful": successful, "fallback": fallback, "failed": failed,
        "success_rate": f"{(ok / len(merged) * 100):.0f}%" if merged else "0%",
    })
    return {**prev, "summary": summary}


def _merge_execution(prev: dict | None, new: dict) -> dict:
    """Fold a partial execute_simulation result into the prior full execution state
    by scenario_id and recompute total/passed/failed."""
    prev = prev or {}
    merged = _merge_by_id(prev.get("runs") or [], new.get("runs") or [], "scenario_id")
    passed = sum(1 for r in merged if r.get("run_name") and not r.get("error"))
    return {**prev, "runs": merged, "total": len(merged), "passed": passed, "failed": len(merged) - passed}


async def _run_tool(name: str, args: dict, state: State) -> tuple[dict, dict]:
    """Execute one agent tool. Returns (result_for_state, state_patch). Raises
    ValueError with a user-facing message if a prerequisite is missing.

    Argument resolution per tool: an explicit payload from the LLM wins; otherwise
    the base is resolved from conversation state (optionally filtered by *_ids) and
    any `overrides` are deep-merged on top. A partial dispatch (a subset / overrides)
    merges its result back into the full stage state so downstream tools see it all;
    the RAW subset result is still returned for the card (the frontend merges too)."""
    cid = state.get("conversation_id", "")
    prog = _forward_progress(cid, name)

    if name == "refine_requirements":
        # Only refine requirements the user ACTUALLY provided. No message-scraping
        # fallback — that turned a chat line like "I'll send a file" into a bogus
        # requirement. If the LLM calls this with nothing, make it go ask the user.
        reqs = [r.strip() for r in (args.get("requirements") or []) if isinstance(r, str) and r.strip()]
        if not reqs:
            raise ValueError(
                "No requirement statements were provided. Do NOT invent any and do NOT refine the "
                "user's chat message — ask the user to paste their ADAS requirements (or attach a "
                "requirements file), then stop."
            )
        _ACTIVE_TOOL[cid] = name
        try:
            result = await gateway.call_tool_json("refine_requirements", {"requirements": reqs}, on_progress=prog)
        finally:
            _ACTIVE_TOOL.pop(cid, None)
        return result, {"refinement": result}

    if name == "generate_test_cases":
        ref = state.get("refinement")
        if not ref:
            raise ValueError("No requirements have been refined yet — run refine_requirements first.")
        if not transforms.testable_requirements(ref):
            # Refinement DID run, but found nothing testable — surface why so the
            # assistant can tell the user what to fix (not a "run refine first" lie).
            incs = ref.get("incomplete") or []
            why = "; ".join(
                f"{r.get('id', 'requirement')}: {', '.join(r.get('issues_found') or ['needs more detail'])}"
                for r in incs
            ) or "the requirement was not testable as written"
            raise ValueError(
                "Refinement produced 0 testable requirements, so there's nothing to generate test "
                f"cases from. What's missing — {why}. Ask the user to add that detail, then re-run "
                "refine_requirements."
            )
        req_ids = args.get("requirement_ids") or None
        reqs = transforms.to_agent2_requirements(ref, req_ids)
        if not reqs:
            raise ValueError(
                f"None of the requested requirement ids {req_ids} are testable — check the ids "
                "against the refinement result's testable list."
            )
        _ACTIVE_TOOL[cid] = name
        try:
            result = await gateway.call_tool_json(
                "generate_test_cases",
                {"requirements": reqs, "refining_id": ref.get("refining_id"), "feature": ref.get("feature")},
                on_progress=prog,
            )
        finally:
            _ACTIVE_TOOL.pop(cid, None)
        return result, {"agent2_result": result, "test_cases": result.get("scenarios", [])}

    if name == "generate_xosc":
        explicit = args.get("specs")
        scenario_ids = args.get("scenario_ids") or None
        overrides = args.get("overrides") or None
        if explicit:
            specs = explicit
        elif state.get("test_cases"):
            specs = transforms.to_agent3_specs(state["test_cases"], overrides=overrides, scenario_ids=scenario_ids)
        else:
            specs = _filter_specs(state.get("specs") or [], scenario_ids, overrides)
        if not specs:
            if scenario_ids:
                raise ValueError(f"No test cases match {scenario_ids} — check the ids or run generate_test_cases first.")
            raise ValueError("No test cases yet — run generate_test_cases first (or pass scenario specs directly).")
        _ACTIVE_TOOL[cid] = name
        try:
            result = await gateway.call_tool_json("generate_xosc", {"specs": specs}, on_progress=prog)
        finally:
            _ACTIVE_TOOL.pop(cid, None)
        # A subset/override regen folds into the full xosc state; a full or explicit build replaces it.
        if (scenario_ids or overrides) and not explicit:
            return result, {"xosc": _merge_xosc(state.get("xosc"), result)}
        return result, {"specs": specs, "xosc": result}

    if name == "execute_simulation":
        # The real executor runs the Agent-3 .xosc files. We run the scenarios that
        # actually have a generated .xosc, but we DON'T silently drop the rest —
        # any that Agent 3 reported without a runnable file are folded back in as
        # errored runs, so the skip is visible in the execution card + report
        # instead of a mysterious lower count (e.g. XOSC said 6, executor ran 5).
        explicit = args.get("scenarios")
        scenario_ids = args.get("scenario_ids") or None
        skipped: list[dict] = []
        if explicit:
            payload = explicit
        else:
            wanted = set(scenario_ids) if scenario_ids else None
            summary = (state.get("xosc") or {}).get("summary") or {}
            selected = [s for s in (summary.get("scenarios") or [])
                        if wanted is None or s.get("scenario_id") in wanted]
            runnable: list[dict] = []
            for s in selected:
                (runnable if (s.get("output_path") and s.get("status") != "failed") else skipped).append(s)
            payload = [{"scenario_id": s.get("scenario_id"), "xosc_path": s.get("output_path")} for s in runnable]
        if not payload:
            if scenario_ids:
                raise ValueError(f"None of {scenario_ids} have a generated .xosc to run — check the ids or run generate_xosc for them first.")
            raise ValueError("No generated .xosc scenarios to execute — run generate_xosc first.")
        result = await gateway.call_tool_json("execute_simulation", {"scenarios": payload}, on_progress=prog)
        if skipped:
            skipped_runs = [
                {"scenario_id": s.get("scenario_id"), "run_name": None,
                 "error": f"Agent 3 produced no .xosc file (generation status: {s.get('status') or 'unknown'})"}
                for s in skipped
            ]
            result = {
                **result,
                "runs": [*result.get("runs", []), *skipped_runs],
                "total": result.get("total", 0) + len(skipped_runs),
                "failed": result.get("failed", 0) + len(skipped_runs),
            }
        if scenario_ids and not explicit:
            return result, {"execution": _merge_execution(state.get("execution"), result)}
        return result, {"execution": result}

    if name == "evaluate_results":
        execution = state.get("execution")
        if not execution:
            raise ValueError("No execution to evaluate — run execute_simulation first.")
        result = await gateway.call_tool_json("evaluate_results", {"execution": execution})
        return result, {"report": result}

    raise ValueError(f"Unknown tool {name}")


def _forward_progress(conversation_id: str, node: str):
    """MCP progress notifications -> the conversation's SSE progress queue, so the
    browser sees live ticks during a minutes-long tool call instead of silence."""
    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_bus.publish(conversation_id, {
            "type": "progress", "node": node, "progress": progress, "total": total, "message": message or "",
        })
    return on_progress


# ── nodes ────────────────────────────────────────────────────────────────────

# Maps a finished tool to the rich pipeline "card" the browser renders (the
# telemetry plane — full result, not the coarse LLM summary). Sent as a `card`
# SSE event; the frontend writes it into pipelineStore so the same dashboards as
# sequential mode appear.
def _card_event(name: str, result: dict, args: dict | None = None) -> dict | None:
    # A subset/override dispatch produces a PARTIAL card (only the affected
    # scenarios). The frontend merges it into the existing card by scenario_id
    # instead of replacing — so re-running one scenario doesn't wipe the others.
    args = args or {}
    if name == "refine_requirements":
        return {"type": "card", "stage": "nlp", "result": result}
    if name == "generate_test_cases":
        return {"type": "card", "stage": "scenario",
                "result": {"testCases": result.get("scenarios", []),
                           "total_scenarios": result.get("total_scenarios", len(result.get("scenarios", [])))}}
    if name == "generate_xosc":
        ids = args.get("scenario_ids") or list((args.get("overrides") or {}).keys())
        partial = bool(ids) and not args.get("specs")
        card = {"type": "card", "stage": "xosc", "result": result.get("summary", result)}
        return {**card, "partial": True, "scenario_ids": ids} if partial else card
    if name == "execute_simulation":
        ids = args.get("scenario_ids") or []
        partial = bool(ids) and not args.get("scenarios")
        card = {"type": "card", "stage": "execution", "result": result}
        return {**card, "partial": True, "scenario_ids": ids} if partial else card
    if name == "evaluate_results":
        return {"type": "card", "stage": "report", "result": result}
    return None


def _trim_file_blocks(messages: list) -> list:
    """Drop the dumped file text from PAST user turns. A file attachment embeds the
    whole document (up to ~100k chars) in the user message; once the agent has read
    it, re-sending it every turn just bloats the context — slow calls, and the
    "Thinking…" hangs the user saw. The CURRENT (last) user turn keeps its file so a
    just-attached file is still readable this turn."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if i != last_user and m.get("role") == "user" and isinstance(content, str) and "--- Attached file:" in content:
            head = content.split("--- Attached file:", 1)[0].strip()
            match = re.search(r"--- Attached file: (.+?) ---", content)
            name = match.group(1) if match else "file"
            note = (f"{head}\n\n" if head else "") + f"[attached file '{name}' — already read earlier; content omitted]"
            out.append({**m, "content": note})
        else:
            out.append(m)
    return out


async def n_agent(state: State) -> dict:
    """One LLM step: reply, or decide to call a tool. Streams the reply token by
    token onto the conversation's SSE queue (via progress_bus) so the browser
    shows text as it's generated instead of waiting for the whole message."""
    cid = state.get("conversation_id", "")
    messages = [{"role": "system", "content": SYSTEM}, *_trim_file_blocks(state.get("messages", []))]

    # Effort chosen in the composer, checkpointed per conversation; guard against an
    # unexpected value before handing it to the API.
    effort = state.get("reasoning_effort", "low")
    if effort not in _REASONING_LEVELS:
        effort = "low"

    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        extra_body={"reasoning_effort": effort}, temperature=0.3, max_tokens=800,
        stream=True,
    )

    content_parts: list[str] = []
    first_at: float | None = None  # time-to-first-token — isolates NIM latency
    # tool_call deltas arrive fragmented across chunks — accumulate by index.
    tc_acc: dict[int, dict] = {}
    async for chunk in stream:
        if not chunk.choices:
            continue
        if first_at is None:
            first_at = time.perf_counter()
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            progress_bus.publish(cid, {"type": "token", "text": delta.content})
        for tc in getattr(delta, "tool_calls", None) or []:
            acc = tc_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                acc["id"] = tc.id
            if tc.function and tc.function.name:
                acc["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                acc["arguments"] += tc.function.arguments

    ttft = round((first_at - t0) * 1000) if first_at else None
    kind = "tool" if tc_acc else "reply"
    perf.log("llm", (time.perf_counter() - t0) * 1000,
             effort=effort, ttft=f"{ttft}ms" if ttft is not None else None,
             kind=kind, msgs=len(messages), chars=len("".join(content_parts)))

    if tc_acc:
        tool_calls = [
            {"id": a["id"] or f"call_{i}", "type": "function",
             "function": {"name": a["name"], "arguments": a["arguments"] or "{}"}}
            for i, a in sorted(tc_acc.items())
        ]
        stored = {"role": "assistant", "content": "".join(content_parts) or None, "tool_calls": tool_calls}
        log = [{"type": "tool_call", "tool": tc["function"]["name"],
                "label": TOOL_LABEL.get(tc["function"]["name"], tc["function"]["name"])} for tc in tool_calls]
        return {"messages": [stored], "log": log}

    text = "".join(content_parts) or "…"
    return {"messages": [{"role": "assistant", "content": text}],
            "log": [{"type": "message", "role": "assistant", "text": text}]}


def _last_tool_calls(state: State) -> list[dict]:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return m["tool_calls"]
    return []


async def n_act(state: State) -> dict:
    """Run the tool call(s) the agent just requested. Pauses for approval first if
    any is a gated (consequential) stage. Appends tool results to the chat."""
    tool_calls = _last_tool_calls(state)
    gated = [tc for tc in tool_calls if tc["function"]["name"] in GATED_TOOLS]

    # Approval gate — everything before interrupt() must stay side-effect-free,
    # since LangGraph re-runs the node from the top when resumed.
    if gated:
        names = [tc["function"]["name"] for tc in gated]
        decision = interrupt({"gate": "tool", "tools": names,
                              "labels": [TOOL_LABEL.get(n, n) for n in names]})
        if isinstance(decision, dict) and decision.get("approved") is False:
            declined = [{"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps({"declined": True, "reason": "user declined to run this"})}
                        for tc in tool_calls]
            # An ABANDON (the user moved on to a new message instead of answering
            # the gate) routes straight to END: we must NOT loop back to the agent,
            # or it would re-propose the very tool that was just abandoned — the bug
            # where a stopped gate later ran the agent anyway. An explicit REJECT
            # ("No, stop") loops back so the agent can acknowledge it.
            route = "end" if decision.get("abandon") else "agent"
            return {"messages": declined, "route": route,
                    "log": [{"type": "gate", "gate": "tool", "tools": names, "result": "rejected"}]}

    out_messages: list[dict] = []
    log: list[dict] = []
    # Accumulates artifact patches so a later tool in the same batch (e.g. refine
    # then generate in one agent turn) sees the earlier one's output.
    artifacts: dict = {}
    if gated:
        log.append({"type": "gate", "gate": "tool", "tools": [tc["function"]["name"] for tc in gated], "result": "approved"})

    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            result, patch = await _run_tool(name, args, {**state, **artifacts})
            summary = _summarize_tool_result(name, result)
            out_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(summary)})
            log.append({"type": "agent", "agent": name, **_log_bits(name, result)})
            card = _card_event(name, result, args)  # rich result → the browser's pipeline card
            if card:
                log.append(card)
            artifacts.update(patch)
        except Exception as exc:  # noqa: BLE001 — surface the reason to the LLM, keep the loop alive
            out_messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps({"error": str(exc)})})
            log.append({"type": "error", "detail": str(exc)})

    # Ran the tool(s) → loop back to the agent so it can summarise. Always set
    # route explicitly so a stale "end" from an earlier abandoned gate can't leak.
    return {"messages": out_messages, "log": log, "route": "agent", **artifacts}


def _log_bits(name: str, result: dict) -> dict:
    if name == "generate_test_cases":
        return {"total_scenarios": result.get("total_scenarios", len(result.get("scenarios", [])))}
    if name in ("execute_simulation",):
        return {"passed": result.get("passed"), "total": result.get("total")}
    if name == "evaluate_results":
        return {"verdict": result.get("verdict"), "score": result.get("score")}
    if name == "generate_xosc":
        return {"summary": result.get("summary", {})}
    return {"summary": result.get("summary", {})}


# ── edges ────────────────────────────────────────────────────────────────────

def after_agent(state: State) -> str:
    """The agent step either asked for a tool (last message has tool_calls) or
    gave a final reply."""
    msgs = state.get("messages", [])
    has_calls = bool(msgs) and msgs[-1].get("role") == "assistant" and bool(msgs[-1].get("tool_calls"))
    return "act" if has_calls else "end"


def after_act(state: State) -> str:
    """Normally loop back to the agent to summarise; but an abandoned gate routes
    straight to END so the agent can't re-propose the tool that was abandoned."""
    return "end" if state.get("route") == "end" else "agent"


def build(checkpointer) -> Any:
    g = StateGraph(State)
    g.add_node("agent", n_agent)
    g.add_node("act", n_act)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"act": "act", "end": END})
    g.add_conditional_edges("act", after_act, {"agent": "agent", "end": END})
    return g.compile(checkpointer=checkpointer)
