"""
Unit & Integration Tests for WololoRAG-MCP (Read/Search Microservice).

Verifies:
1. Pydantic configuration & DATABASE_URL / QDRANT_URL env overrides.
2. LangGraph RetrievalStateGraph against Qdrant + SQL hydration.
3. OptionalTokenAuthMiddleware (`enabled=False` vs `enabled=True`).
4. MCPServer `/health` endpoint on both `sse_app` and `streamable_http_app`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anyio
import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.config import AppConfig, DatabaseConfig, EmbeddingConfig, FullSettings, VectorConfig
from src.db.models import ArticleRecord, ArticleStatus
from src.db.repository import ArticleRepository
from src.graphs.retrieval import run_retrieval_query
from src.mcp_server.auth import OptionalTokenAuthMiddleware
from src.mcp_server.server import create_mcp_server
from src.vector.qdrant_store import QdrantStoreManager


class DeterministicFakeEmbeddings(Embeddings):
    """Lightweight deterministic mock embeddings for offline search verification."""

    def _encode(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [
            float(lowered.count("ps5") + 0.1),
            float(lowered.count("exploit") + lowered.count("jailbreak") + 0.1),
            float(lowered.count("switch") + 0.1),
            float(lowered.count("vita") + 0.1),
        ]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def _make_test_settings(tmp_path: Path) -> FullSettings:
    return FullSettings(
        app=AppConfig(
            embedding={"provider": "local"},
            database={
                "sqlite_path": str(tmp_path / "mcp_test.db"),
                "qdrant_mode": "local",
                "qdrant_local_path": str(tmp_path / "qdrant"),
                "collection_prefix": "test_mcp",
            },
        ),
        vector=VectorConfig(
            search={
                "top_k": 3,
                "score_threshold": 0.1,
                "distance_metric": "Cosine",
            }
        ),
    )


def test_pydantic_mistral_requires_key_and_db_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        EmbeddingConfig(provider="mistral")

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://wololo:pass@localhost:5432/wololo_rag")
    monkeypatch.setenv("QDRANT_MODE", "server")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    db_cfg = DatabaseConfig()
    assert db_cfg.sqlalchemy_url.startswith("postgresql+psycopg://")
    assert db_cfg.qdrant_mode == "server"


def test_retrieval_langgraph_queries_qdrant_and_hydrates_sql(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    repo = ArticleRepository(settings.app.database.sqlalchemy_url)
    embeddings = DeterministicFakeEmbeddings()

    # Seed SQLite row as written by the Indexer service
    with repo._session() as session:
        session.add(
            ArticleRecord(
                wp_post_id=49055,
                title="PS5 Byepervisor Exploit Released",
                url="https://wololo.net/ps5-byepervisor",
                topic="PS5",
                published_at=datetime(2024, 10, 26, tzinfo=timezone.utc),
                cleaned_content="PS5 hypervisor exploit details for firmware 2.xx.",
                status=ArticleStatus.INDEXED.value,
                vector_ids=["v1"],
                chunk_count=1,
                embedding_provider="local",
            )
        )
        session.commit()

    # Seed Qdrant collection directly (simulating Indexer writes)
    client = QdrantClient(path=settings.app.database.qdrant_local_path)
    try:
        client.create_collection(
            collection_name="test_mcp_local",
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )
        vec = embeddings.embed_query("PS5 hypervisor exploit")
        client.upsert(
            collection_name="test_mcp_local",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "wp_post_id": 49055,
                        "title": "PS5 Byepervisor Exploit Released",
                        "url": "https://wololo.net/ps5-byepervisor",
                        "topic": "PS5",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "token_count": 12,
                        "text": "PS5 hypervisor exploit details for firmware 2.xx.",
                    },
                )
            ],
        )
    finally:
        client.close()

    qdrant_store = QdrantStoreManager(settings.app, settings.vector, "test_mcp_local")
    try:
        state = run_retrieval_query(
            query="PS5 exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant_store,
            topic_filter="PS5",
        )
        hits = state["enriched_hits"]
        assert len(hits) == 1
        assert hits[0]["wp_post_id"] == 49055
        assert hits[0]["sql_status"] == ArticleStatus.INDEXED.value
        assert "https://wololo.net/ps5-byepervisor" in state["formatted_context"]
    finally:
        qdrant_store.close()


def test_mcp_optional_auth_middleware() -> None:
    async def ping(_request):
        return JSONResponse({"ok": True})

    async def health(_request):
        return JSONResponse({"status": "ok"})

    app_open = Starlette(routes=[Route("/mcp", ping), Route("/health", health)])
    cfg_open = AppConfig(mcp_server={"auth": {"enabled": False, "secret_token": "topsecret"}}).mcp_server.auth
    app_open.add_middleware(OptionalTokenAuthMiddleware, auth_config=cfg_open)
    assert TestClient(app_open).get("/mcp").status_code == 200

    app_locked = Starlette(routes=[Route("/mcp", ping), Route("/health", health)])
    cfg_locked = AppConfig(mcp_server={"auth": {"enabled": True, "secret_token": "topsecret"}}).mcp_server.auth
    app_locked.add_middleware(OptionalTokenAuthMiddleware, auth_config=cfg_locked)
    client = TestClient(app_locked)

    assert client.get("/health").status_code == 200
    assert client.get("/mcp").status_code == 401
    assert client.get("/mcp", headers={"X-MCP-Key": "wrong"}).status_code == 401
    assert client.get("/mcp", headers={"X-MCP-Key": "topsecret"}).status_code == 200
    assert client.get("/mcp", headers={"Authorization": "Bearer topsecret"}).status_code == 200


def test_mcp_server_health_both_transports(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    mcp, runtime = create_mcp_server(settings)
    try:
        for app in (mcp.sse_app(host="127.0.0.1"), mcp.streamable_http_app(host="127.0.0.1")):
            app.add_middleware(
                OptionalTokenAuthMiddleware,
                auth_config=settings.app.mcp_server.auth,
            )
            resp = TestClient(app).get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["sql_backend"] == "SQLite"
    finally:
        runtime.close()


def test_mcp_public_dto_hides_internal_fields_and_caches_graph(tmp_path: Path) -> None:
    """
    Verify:
    1. get_article_details omits internal DB columns (`vector_ids`, `error_str`, `embedding_provider`).
    2. McpRuntime.get_retrieval_graph() caches the compiled graph instance across calls.
    """
    settings = _make_test_settings(tmp_path)
    mcp, runtime = create_mcp_server(settings)
    runtime._embeddings = DeterministicFakeEmbeddings()

    try:
        with runtime.repo._session() as session:
            session.add(
                ArticleRecord(
                    wp_post_id=777,
                    title="PS5 Internal Field Check",
                    url="https://wololo.net/777",
                    topic="PS5",
                    published_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
                    cleaned_content="Public article text for LLM.",
                    status=ArticleStatus.INDEXED.value,
                    vector_ids=["secret-uuid-1"],
                    chunk_count=1,
                    embedding_provider="local",
                    error_str="internal debug stacktrace",
                )
            )
            session.commit()

        # Verify graph compilation is cached
        g1 = runtime.get_retrieval_graph()
        g2 = runtime.get_retrieval_graph()
        assert g1 is g2

        # Call real MCP tool through MCPServer
        call_res = anyio.run(mcp.call_tool, "get_article_details", {"wp_post_id": 777})
        payload = json.loads(call_res.content[0].text)
        assert payload["found"] is True
        assert payload["cleaned_content"] == "Public article text for LLM."
        assert "vector_ids" not in payload
        assert "error_str" not in payload
        assert "embedding_provider" not in payload
    finally:
        runtime.close()


def test_mcp_dedup_date_filter_and_full_article_mode(tmp_path: Path) -> None:
    """Verify max_chunks_per_article=1, published_after, and full_article context_mode."""
    settings = _make_test_settings(tmp_path)
    settings.vector.search.max_chunks_per_article = 1
    settings.vector.search.context_mode = "full_article"
    repo = ArticleRepository(settings.app.database.sqlalchemy_url)
    embeddings = DeterministicFakeEmbeddings()

    with repo._session() as session:
        session.add(
            ArticleRecord(
                wp_post_id=11,
                title="Old Multi-Chunk Post",
                url="https://wololo.net/11",
                topic="PS5",
                published_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
                cleaned_content="Old PS5 exploit FULL_SQL_BODY_OLD",
                status=ArticleStatus.INDEXED.value,
                vector_ids=["a", "b"],
                chunk_count=2,
            )
        )
        session.add(
            ArticleRecord(
                wp_post_id=22,
                title="Recent Post",
                url="https://wololo.net/22",
                topic="PS5",
                published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
                cleaned_content="New PS5 exploit FULL_SQL_BODY_NEW",
                status=ArticleStatus.INDEXED.value,
                vector_ids=["c"],
                chunk_count=1,
            )
        )
        session.commit()

    client = QdrantClient(path=settings.app.database.qdrant_local_path)
    try:
        client.create_collection(
            collection_name="test_mcp_local",
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )
        vec = embeddings.embed_query("PS5 exploit")
        client.upsert(
            collection_name="test_mcp_local",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "wp_post_id": 11,
                        "title": "Old Multi-Chunk Post",
                        "url": "https://wololo.net/11",
                        "topic": "PS5",
                        "published_at": "2022-01-01T00:00:00+00:00",
                        "chunk_index": 0,
                        "total_chunks": 2,
                        "text": "Old chunk 1 PS5 exploit",
                    },
                ),
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "wp_post_id": 11,
                        "title": "Old Multi-Chunk Post",
                        "url": "https://wololo.net/11",
                        "topic": "PS5",
                        "published_at": "2022-01-01T00:00:00+00:00",
                        "chunk_index": 1,
                        "total_chunks": 2,
                        "text": "Old chunk 2 PS5 exploit",
                    },
                ),
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "wp_post_id": 22,
                        "title": "Recent Post",
                        "url": "https://wololo.net/22",
                        "topic": "PS5",
                        "published_at": "2026-03-01T00:00:00+00:00",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "text": "New chunk PS5 exploit",
                    },
                ),
            ],
        )
    finally:
        client.close()

    qdrant_store = QdrantStoreManager(settings.app, settings.vector, "test_mcp_local")
    try:
        # Dedup ensures post 11 appears at most once
        state_all = run_retrieval_query(
            query="PS5 exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant_store,
        )
        ids_all = [h["wp_post_id"] for h in state_all["enriched_hits"]]
        assert ids_all.count(11) == 1
        assert ids_all.count(22) == 1
        assert "FULL_SQL_BODY_NEW" in state_all["formatted_context"]

        # Date filter keeps only 2026 post
        state_recent = run_retrieval_query(
            query="PS5 exploit",
            settings=settings,
            repo=repo,
            embeddings=embeddings,
            qdrant_store=qdrant_store,
            published_after="2025-01-01",
        )
        assert [h["wp_post_id"] for h in state_recent["enriched_hits"]] == [22]
    finally:
        qdrant_store.close()
