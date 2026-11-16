"""
Embedding provider factory for query encoding (src/vector/embeddings.py).

Instantiates the same embedding backend configured for the Indexer so query
vectors match the target Qdrant collection dimensionality and vector space.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.config import AppConfig


def get_collection_name(app_config: AppConfig) -> str:
    """Build the provider-specific Qdrant collection name (must match Indexer)."""
    prefix = app_config.database.collection_prefix.strip()
    provider = app_config.embedding.provider.strip()
    return f"{prefix}_{provider}"


def create_embedding_model(app_config: AppConfig) -> Embeddings:
    """Instantiate the configured LangChain `Embeddings` implementation."""
    provider = app_config.embedding.provider

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        local_cfg = app_config.embedding.local
        return HuggingFaceEmbeddings(
            model_name=local_cfg.model_name,
            model_kwargs={"device": local_cfg.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        from pydantic import SecretStr

        mistral_cfg = app_config.embedding.mistral
        return MistralAIEmbeddings(
            model=mistral_cfg.model_name,
            api_key=SecretStr(mistral_cfg.api_key),
            max_retries=mistral_cfg.max_retries,
            timeout=mistral_cfg.timeout_sec,
        )

    raise ValueError(f"Unsupported embedding provider: {provider!r}")
