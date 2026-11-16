# WololoRAG-MCP — Standalone MCP Server (Read / Search Service)

A standalone **Model Context Protocol (MCP)** microservice that connects large language models (**Antigravity**, **Gemini CLI**, **Claude Desktop**) to the **Wololo.net** RAG knowledge base stored in **Qdrant** and **PostgreSQL / SQLite**.

Following **CQRS (Command Query Responsibility Segregation)**, this repository contains **zero** scraping or text-chunking logic — it is strictly dedicated to semantic search and metadata retrieval:
- Validated **Pydantic** configuration for embeddings, databases, search parameters, and optional auth
- Read-only **SQLAlchemy** repository (`PostgreSQL` or `SQLite` fallback)
- Read-only **Qdrant** vector search client + provider-matched embedding factory (`local` / `mistral`)
- **LangGraph** `RetrievalStateGraph` (`embed_query` -> `search_qdrant` -> `hydrate_with_sql` -> `format_context`)
- `MCPServer` exposing read-only tools (`search_wololo_knowledge`, `get_article_details`, `get_database_status`) + `/health` HTTP route and `OptionalTokenAuthMiddleware`

---

## 3-Repository Architecture

| Repository | Responsibility | GitHub |
| :--- | :--- | :--- |
| **`WololoRAG-Infra`** | Shared Docker database stack (`PostgreSQL` + `Adminer` + `Qdrant`) | https://github.com/older0007/WololoRAG-Infra |
| **`WololoRAG`** | **Indexer Worker (Write Side)**: crawls Wololo.net, chunks text, embeds & writes vectors + SQL state | https://github.com/older0007/WololoRAG |
| **`WololoRAG-MCP`** *(this repo)* | **MCP Server (Read Side)**: serves semantic search & article retrieval to LLM clients | https://github.com/older0007/WololoRAG-MCP |

---

## Setup & Database Connection

### 1. Environment Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### 2. Database Configuration (`.env`)

#### Option A: Connect to `WololoRAG-Infra` (PostgreSQL + Qdrant on `localhost`)
Copy `.env.example` to `.env` and uncomment:
```env
DATABASE_URL=postgresql+psycopg://wololo:wololo_pass@localhost:5432/wololo_rag
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
```

#### Option B: Local Files from Sibling `WololoRAG` Repo (Default)
When `.env` is omitted, `WololoRAG-MCP` automatically reads `../WololoRAG/data/wololo_state.db` and `../WololoRAG/data/qdrant`.

---

## Running the MCP Server

```bash
# Streamable HTTP transport (recommended for Antigravity, endpoint: http://localhost:8080/mcp)
.venv/bin/python -m src.mcp_server.server --transport streamable-http

# Legacy SSE transport (endpoint: http://localhost:8080/sse)
.venv/bin/python -m src.mcp_server.server --transport sse

# Stdio transport (for local process-based MCP clients)
.venv/bin/python -m src.mcp_server.server --transport stdio
```

Check service health in your browser or curl:
```bash
curl http://localhost:8080/health
```

---

## Connecting to Antigravity / Gemini CLI (`~/.gemini/config/mcp_config.json`)

```json
{
  "mcpServers": {
    "wololo-rag": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### Optional API Key / Bearer Token Authentication
Authentication is **disabled by default** (`mcp_server.auth.enabled: false` in [`config/app_config.yaml`](config/app_config.yaml)). If you expose this server over a tunnel (e.g. `ngrok` or `cloudflared`), set `enabled: true` and provide `MCP_SECRET_TOKEN` in `.env` — every MCP request will require `X-MCP-Key: <token>` or `Authorization: Bearer <token>`.

---

## Tests

```bash
.venv/bin/pytest -v
```
