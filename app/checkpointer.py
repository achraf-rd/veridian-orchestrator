"""Durable state: Postgres LangGraph checkpointer, with in-memory fallback.

Uses the same Postgres as the Next.js app (adds langgraph checkpoint tables).
Falls back to MemorySaver if DATABASE_URL is unset or the DB is unreachable,
so the orchestrator still runs in a bare dev environment.
"""

from typing import Any

from app.config import DATABASE_URL


async def make_checkpointer() -> tuple[Any, Any]:
    """Return (checkpointer, context_manager_to_close_or_None)."""
    if DATABASE_URL:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
            saver = await cm.__aenter__()
            await saver.setup()
            print("[orchestrator] Postgres checkpointer ready (durable, resumable)")
            return saver, cm
        except Exception as exc:  # noqa: BLE001 — any DB/driver failure → degrade gracefully
            print(f"[orchestrator] Postgres checkpointer unavailable ({exc}); using in-memory")

    from langgraph.checkpoint.memory import MemorySaver

    print("[orchestrator] using in-memory checkpointer (state lost on restart)")
    return MemorySaver(), None
