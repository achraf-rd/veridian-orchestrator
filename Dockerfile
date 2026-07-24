FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependency manifests first for layer caching.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY . .
RUN chmod +x entrypoint.sh

# FastAPI decision stream (/chat, /resume, ...).
EXPOSE 8200
ENTRYPOINT ["./entrypoint.sh"]
