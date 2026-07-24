"""Lightweight timing logs — see where each turn's seconds actually go.

Writes one line per instrumented step to stdout (the orchestrator terminal):

    12:03:41 [perf] llm                 8421ms  effort=low ttft=8210ms kind=reply msgs=3 chars=64
    12:03:41 [perf] mcp:refine_...       412ms  connect=120ms
    12:03:49 [perf] turn                8955ms  conv=abc kind=message

`ttft` (time-to-first-token) isolates model/NIM latency from generation; a big
`ttft` with `effort=low` means the NIM endpoint itself is slow, not the pipeline.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

logger = logging.getLogger("veridian.perf")


def setup_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [perf] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def log(event: str, ms: float, **fields) -> None:
    extra = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    logger.info(f"{event:<20} {ms:7.0f}ms  {extra}")


@asynccontextmanager
async def timed(event: str, **fields):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log(event, (time.perf_counter() - t0) * 1000, **fields)
