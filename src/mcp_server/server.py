"""
Standalone MCP Read/Search Microservice (src/mcp_server/server.py).

Architectural Role (CQRS Read Side):
    This microservice exposes ONLY read & search tools to external LLMs
    (Antigravity, Gemini CLI, Claude Desktop). Data ingestion/scraping happens
    exclusively in the separate `WololoRAG` Indexer repository.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import FullSettings, load_settings
from src.db.repository import ArticleRepository
from src.graphs.retrieval import build_retrieval_graph
from src.mcp_server.auth import OptionalTokenAuthMiddleware
from src.vector.embeddings import create_embedding_model, get_collection_name
from src.vector.qdrant_store import QdrantStoreManager


class McpRuntime:
    """Shared dependency container initialized once for the MCP server process."""

    def __init__(self, settings: FullSettings) -> None:
        self.settings = settings
        self.active_provider = settings.app.embedding.provider
        self.collection_name = get_collection_name(settings.app)
        self.repo = ArticleRepository(settings.app.database.sqlalchemy_url)
        self.qdrant = QdrantStoreManager(
            app_config=settings.app,
            vector_config=settings.vector,
            collection_name=self.collection_name,
        )
        self._embeddings = None
        self._retrieval_graph = None

    def get_embeddings(self):
        """Lazily load query embedding weights on first search call."""
        if self._embeddings is None:
            self._embeddings = create_embedding_model(self.settings.app)
        return self._embeddings

    def get_retrieval_graph(self):
        """
        Compile and cache the LangGraph RetrievalStateGraph once per runtime
        instead of recompiling on every search request.
        """
        if self._retrieval_graph is None:
            self._retrieval_graph = build_retrieval_graph(
                settings=self.settings,
                repo=self.repo,
                embeddings=self.get_embeddings(),
                qdrant_store=self.qdrant,
            )
        return self._retrieval_graph

    def close(self) -> None:
        self.qdrant.close()


def create_mcp_server(settings: FullSettings) -> tuple[MCPServer, McpRuntime]:
    """Build the `MCPServer` instance and register all read-only RAG tools."""
    runtime = McpRuntime(settings)

    @asynccontextmanager
    async def server_lifespan(_: MCPServer) -> AsyncIterator[None]:
        try:
            yield
        finally:
            runtime.close()

    mcp = MCPServer(
        name="WololoRAG",
        instructions=(
            "Read-only RAG knowledge base over Wololo.net articles (PS5, PS4, PS Vita, "
            "Switch, homebrew, and retro gaming news). Use `search_wololo_knowledge` to "
            "search indexed chunks by semantic similarity, `get_article_details` for full "
            "article text, and `get_database_status` to inspect SQL + Qdrant metrics."
        ),
        lifespan=server_lifespan,
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(_: Request) -> Response:
        stats = runtime.repo.get_stats(runtime.active_provider)
        return JSONResponse(
            {
                "status": "ok",
                "service": "wololo-mcp-server",
                "sql_backend": runtime.repo.backend_name,
                "qdrant_mode": settings.app.database.qdrant_mode,
                "provider": runtime.active_provider,
                "collection": runtime.collection_name,
                "auth_enabled": settings.app.mcp_server.auth.enabled,
                "articles_in_sql": stats.total_in_sql,
                "articles_indexed": stats.indexed_for_active_provider,
            }
        )

    @mcp.tool()
    def search_wololo_knowledge(
        query: str,
        topic: str | None = None,
        top_k: int = 5,
        published_after: str | None = None,
    ) -> str:
        """
        Search Wololo.net articles by meaning using Qdrant hybrid search + SQL hydration.

        Args:
            query: Natural-language search query (e.g. "PS5 firmware 9.00 exploit").
            topic: Optional category/topic filter (e.g. "PS5", "PS4", "Switch", "PS Vita").
            top_k: Maximum number of unique articles to return (default: 5).
            published_after: Optional ISO date filter (e.g. "2024-01-01").
        """
        graph = runtime.get_retrieval_graph()
        result_state = graph.invoke(
            {
                "query": query,
                "topic_filter": topic,
                "top_k": top_k,
                "published_after": published_after,
            }
        )
        return result_state.get("formatted_context", "No results found.")

    @mcp.tool()
    def get_article_details(wp_post_id: int) -> dict[str, Any]:
        """
        Retrieve public metadata and full cleaned article text for a specific
        Wololo.net WordPress post ID.
        """
        records = runtime.repo.get_by_ids([wp_post_id])
        row = records.get(wp_post_id)
        if row is None:
            return {"found": False, "wp_post_id": wp_post_id}

        return {
            "found": True,
            "wp_post_id": row.wp_post_id,
            "title": row.title,
            "url": row.url,
            "topic": row.topic,
            "status": row.status,
            "chunk_count": row.chunk_count,
            "published_at": row.published_at.isoformat() if row.published_at else None,
            "remote_modified_at": (
                row.remote_modified_at.isoformat() if row.remote_modified_at else None
            ),
            "cleaned_content": row.cleaned_content,
        }

    @mcp.tool()
    def get_database_status() -> dict[str, Any]:
        """
        Inspect current synchronization counts across SQL (PostgreSQL/SQLite) and Qdrant.
        """
        stats = runtime.repo.get_stats(runtime.active_provider)
        qdrant_vectors = runtime.qdrant.count_vectors()

        return {
            "active_provider": runtime.active_provider,
            "sql_backend": runtime.repo.backend_name,
            "qdrant_mode": runtime.settings.app.database.qdrant_mode,
            "collection_name": runtime.collection_name,
            "qdrant_vectors_count": qdrant_vectors,
            "sql_stats": stats.model_dump(),
        }

    return mcp, runtime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WololoRAG Standalone MCP Microservice")
    parser.add_argument("--app-config", type=str, default=None, help="Path to app_config.yaml")
    parser.add_argument("--vector-config", type=str, default=None, help="Path to vector_config.yaml")
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable-http", "stdio"],
        default=None,
        help="Override transport from app_config.yaml ('streamable-http', 'sse', or 'stdio')",
    )
    parser.add_argument("--host", type=str, default=None, help="Override bind host")
    parser.add_argument("--port", type=int, default=None, help="Override bind port")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(
        app_config_path=args.app_config,
        vector_config_path=args.vector_config,
    )

    transport = args.transport or settings.app.mcp_server.transport
    host = args.host or settings.app.mcp_server.host
    port = args.port or settings.app.mcp_server.port

    mcp, runtime = create_mcp_server(settings)

    if transport == "stdio":
        try:
            mcp.run(transport="stdio")
        finally:
            runtime.close()
        return 0

    if transport == "streamable-http":
        starlette_app = mcp.streamable_http_app(host=host)
    else:
        starlette_app = mcp.sse_app(host=host)

    starlette_app.add_middleware(
        OptionalTokenAuthMiddleware,
        auth_config=settings.app.mcp_server.auth,
    )
    uvicorn.run(starlette_app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
