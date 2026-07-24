"""NVIDIA NIM (OpenAI-compatible) client — the orchestrator's GPT-OSS brain."""

from openai import AsyncOpenAI

from app.config import NIM_API_KEY, NIM_BASE_URL, NIM_MODEL

# timeout guards against the occasional cold-start hang; retries cover transients.
client = AsyncOpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY, timeout=90.0, max_retries=2)
MODEL = NIM_MODEL
