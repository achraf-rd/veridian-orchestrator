import os
from pathlib import Path

from dotenv import load_dotenv

# Local orchestrator .env first, then fill any gaps from the sibling
# Veridian_frontend/.env (which already holds NIM_API_KEY + DATABASE_URL).
load_dotenv()
_frontend_env = Path(__file__).resolve().parents[2] / "Veridian_frontend" / ".env"
if _frontend_env.exists():
    load_dotenv(_frontend_env, override=False)

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NIM_API_KEY = (os.getenv("NIM_API_KEY") or "").strip()
NIM_MODEL = os.getenv("NIM_MODEL", "openai/gpt-oss-120b")

MCP_GATEWAY_URL = os.getenv("MCP_GATEWAY_URL", "http://127.0.0.1:8100/mcp")
DATABASE_URL = os.getenv("DATABASE_URL")

ORCH_HOST = os.getenv("ORCH_HOST", "127.0.0.1")
ORCH_PORT = int(os.getenv("ORCH_PORT", "8200"))
