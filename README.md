# Veridian Orchestrator

The LLM supervisor ("Global Orchestrator") that drives the 5-agent ADAS pipeline. Hybrid design (plan Option A): an LLM brain classifies intent, a deterministic LangGraph enforces ordering + HITL gates + durable state, and agents are invoked as MCP tools on the gateway.

| Piece | What |
|---|---|
| **Brain** | GPT-OSS 120B via NVIDIA NIM (`https://integrate.api.nvidia.com/v1`, OpenAI-compatible) — classifies each request into `full_pipeline` / `direct_testcase` / `replay_with_mods` / `reevaluate` and extracts params |
| **Graph** | LangGraph: `classify → refine → [gate] → gen_tc → [gate] → gen_xosc → [gate] → execute → evaluate`. Gates pause via `interrupt()`; state checkpointed to Postgres after every node |
| **Tools** | MCP client to `veridian-mcp-gateway` (Streamable HTTP) — the control plane. Live per-scenario telemetry stays on the Next.js SSE routes |
| **API** | `POST /chat` (SSE decision stream), `POST /resume` (HITL approval), `GET /prompts`, `GET /runs/{id}`, `GET /health` |

## Run

The gateway must be up first (it's the tool source):

```bash
# 1) gateway  (in ../veridian-mcp-gateway)
uv run python server.py            # http://127.0.0.1:8100/mcp

# 2) orchestrator (here)
uv sync
uv run python -u server.py         # http://127.0.0.1:8200
```

Config comes from `.env` here, falling back to `../Veridian_frontend/.env` for `NIM_API_KEY` and `DATABASE_URL` (so the existing key/DB are reused). See `.env.example`.

**Windows note:** the entrypoint forces `WindowsSelectorEventLoopPolicy` and runs uvicorn with `loop="none"` — psycopg's async Postgres driver (the durable checkpointer) can't use the default Proactor loop. Without this it silently falls back to in-memory.

## Test

```bash
uv run python tests/test_classify.py         # real NIM: 4 intents classified + params extracted
uv run python tests/test_graph_mock.py       # offline: 3 HITL gates, resume, rejection, deferred, persistence
uv run python tests/test_integration_http.py # both services up: /chat -> real Agent 1 -> requirements gate
```

Durability across a restart (proves Postgres checkpointing):

```bash
uv run python tests/durability.py chat       # run to the gate
# ...restart the orchestrator process...
uv run python tests/durability.py check      # same run still paused -> survived
```

## Driving it by hand

```bash
# start a run (streams decisions until the first gate)
curl -N -X POST http://127.0.0.1:8200/chat -H 'content-type: application/json' \
  -d '{"conversation_id":"c1","message":"The AEB shall stop before a pedestrian at 50 km/h."}'

# approve the gate and continue (streams until the next gate / completion)
curl -N -X POST http://127.0.0.1:8200/resume -H 'content-type: application/json' \
  -d '{"conversation_id":"c1","decision":{"approved":true}}'

# reject instead
#   -d '{"conversation_id":"c1","decision":{"approved":false}}'
```

## Status (plan Phase 2 — done)

- ✅ `full_pipeline` runs end-to-end through real Agents 1-3 (4/5 are gateway stubs) with 3 HITL gates and durable resumable state.
- ⏳ `direct_testcase` / `replay_with_mods` / `reevaluate` are recognized and routed to a `deferred` node — deep routing needs MCP resource reads (plan Phase 3).
- ⏳ Next.js integration (orchestrate proxy, composer slash commands) is Phase 4.

> Caveat: NIM occasionally cold-starts slowly (a first request can stall ~1-2 min); the LLM client has a 90 s timeout + retries. Subsequent calls are ~1.5 s.
