#!/bin/sh
set -e
# Bind all interfaces so Traefik can reach us (code default is local-only).
export ORCH_HOST=0.0.0.0
export ORCH_PORT="${ORCH_PORT:-8200}"
exec uv run python server.py
