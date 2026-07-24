"""Context loading via MCP resources — the "Load Context from DB" node.

Reads the gateway's Postgres-backed resources (conversation summary, a single
test case, project index) instead of touching the DB directly. Keeps the
orchestrator decoupled from Prisma/schema — it only knows resource URIs.
"""

import json
from typing import Any

from app.mcp_client import gateway


async def _read(uri: str) -> dict:
    contents = await gateway.read_resource(uri)
    for item in contents or []:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {}


async def conversation_summary(conversation_id: str) -> dict:
    return await _read(f"conversation://{conversation_id}/summary")


async def test_case(conversation_id: str, tc_id: str) -> dict:
    return await _read(f"conversation://{conversation_id}/testcases/{tc_id}")


async def project_conversations(project_id: str) -> dict:
    return await _read(f"project://{project_id}/conversations")
