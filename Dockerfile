# ==============================================================================
# Dockerfile — WololoRAG-MCP (Standalone MCP Search Microservice)
# ==============================================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md* ./
COPY src ./src
COPY config ./config

RUN pip install --upgrade pip && pip install .

EXPOSE 8080
CMD ["python", "-m", "src.mcp_server.server", "--transport", "streamable-http"]
