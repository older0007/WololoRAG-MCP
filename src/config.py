"""
Pydantic configuration loader and validator for WololoRAG-MCP (src/config.py).

Reads `config/app_config.yaml`, `config/vector_config.yaml`, and `.env`
so the server can point to either `WololoRAG-Infra` (PostgreSQL + Qdrant Server)
or local files in `../WololoRAG/data/`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class LocalEmbeddingConfig(BaseModel):
    """Settings for the local HuggingFace embedding model (must match Indexer)."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="cpu")


class MistralEmbeddingConfig(BaseModel):
    """Settings for Mistral's cloud embedding API (must match Indexer)."""

    api_key: str = Field(default="")
    model_name: str = Field(default="mistral-embed")
    timeout_sec: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)


class EmbeddingConfig(BaseModel):
    """Strategy selector between 'local' and 'mistral' query embeddings."""

    provider: Literal["local", "mistral"] = "local"
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)
    mistral: MistralEmbeddingConfig = Field(default_factory=MistralEmbeddingConfig)

    @model_validator(mode="after")
    def resolve_and_validate_mistral_key(self) -> "EmbeddingConfig":
        env_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not self.mistral.api_key.strip() and env_key:
            self.mistral.api_key = env_key

        if self.provider == "mistral" and not self.mistral.api_key.strip():
            raise ValueError(
                "Configuration error: embedding.provider is set to 'mistral', "
                "but neither embedding.mistral.api_key nor MISTRAL_API_KEY is set."
            )
        return self


class DatabaseConfig(BaseModel):
    """Connection settings to the shared SQL (PostgreSQL/SQLite) and Qdrant databases."""

    database_url: str = Field(default="")
    sqlite_path: str = Field(default="../WololoRAG/data/wololo_state.db")
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: str = Field(default="../WololoRAG/data/qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    collection_prefix: str = Field(default="wololo_articles")

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "DatabaseConfig":
        """Allow `.env` variables to route the MCP service to `WololoRAG-Infra`."""
        if env_db_url := os.getenv("DATABASE_URL", "").strip():
            self.database_url = env_db_url
        if env_sqlite := os.getenv("SQLITE_PATH", "").strip():
            self.sqlite_path = env_sqlite
        if env_mode := os.getenv("QDRANT_MODE", "").strip():
            if env_mode in ("local", "server"):
                self.qdrant_mode = env_mode  # type: ignore[assignment]
        if env_url := os.getenv("QDRANT_URL", "").strip():
            self.qdrant_url = env_url
        if env_local_path := os.getenv("QDRANT_LOCAL_PATH", "").strip():
            self.qdrant_local_path = env_local_path
        return self

    @property
    def sqlalchemy_url(self) -> str:
        """Return explicit `database_url` if set, otherwise fall back to `sqlite_path`."""
        if self.database_url.strip():
            return self.database_url.strip()
        return self.sqlite_path


class McpAuthConfig(BaseModel):
    """Optional API key / Bearer token protection for the MCP HTTP endpoints."""

    enabled: bool = False
    api_key_header: str = Field(default="X-MCP-Key")
    secret_token: str = Field(default="dev-mcp-secret-token")

    @model_validator(mode="after")
    def resolve_env_secret(self) -> "McpAuthConfig":
        env_token = os.getenv("MCP_SECRET_TOKEN", "").strip()
        if env_token:
            self.secret_token = env_token
        if self.enabled and not self.secret_token.strip():
            raise ValueError(
                "Configuration error: mcp_server.auth.enabled is true, "
                "but secret_token / MCP_SECRET_TOKEN is empty."
            )
        return self


class McpServerConfig(BaseModel):
    """Network and transport settings for the MCP microservice."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, gt=0, le=65535)
    transport: Literal["sse", "streamable-http", "stdio"] = "streamable-http"
    auth: McpAuthConfig = Field(default_factory=McpAuthConfig)


class AppConfig(BaseModel):
    """Root schema for config/app_config.yaml."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mcp_server: McpServerConfig = Field(default_factory=McpServerConfig)


class SearchConfig(BaseModel):
    """Semantic search, deduplication, and formatting parameters matching config/vector_config.yaml."""

    top_k: int = Field(default=5, gt=0, le=50)
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    distance_metric: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
    max_chunks_per_article: int = Field(default=1, ge=1, le=10)
    hybrid_keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    context_mode: Literal["best_chunk", "full_article"] = "best_chunk"


class VectorConfig(BaseModel):
    """Root schema for config/vector_config.yaml."""

    search: SearchConfig = Field(default_factory=SearchConfig)


class FullSettings(BaseModel):
    """Combined validated settings passed to the MCP server runtime."""

    app: AppConfig
    vector: VectorConfig


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must be a key-value mapping.")
    return data


def load_settings(
    app_config_path: str | Path | None = None,
    vector_config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
) -> FullSettings:
    """Load `.env` and validate both YAML config files via Pydantic."""
    load_dotenv(dotenv_path=dotenv_path)

    project_root = Path(__file__).resolve().parents[1]
    resolved_app_path = Path(
        app_config_path or os.getenv("APP_CONFIG_PATH") or (project_root / "config" / "app_config.yaml")
    )
    resolved_vector_path = Path(
        vector_config_path or os.getenv("VECTOR_CONFIG_PATH") or (project_root / "config" / "vector_config.yaml")
    )

    settings = FullSettings(
        app=AppConfig.model_validate(_read_yaml(resolved_app_path)),
        vector=VectorConfig.model_validate(_read_yaml(resolved_vector_path)),
    )

    # Resolve relative fallback paths against WololoRAG-MCP project root so `stdio`
    # launches from outside this directory still locate sibling `../WololoRAG/data/`.
    sqlite_path = Path(settings.app.database.sqlite_path)
    if not sqlite_path.is_absolute():
        settings.app.database.sqlite_path = str((project_root / sqlite_path).resolve())

    qdrant_local = Path(settings.app.database.qdrant_local_path)
    if not qdrant_local.is_absolute():
        settings.app.database.qdrant_local_path = str((project_root / qdrant_local).resolve())

    return settings
