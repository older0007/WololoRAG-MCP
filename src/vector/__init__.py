"""
Vector search layer for WololoRAG-MCP: query embeddings + Qdrant read client.
"""

from src.vector.embeddings import create_embedding_model, get_collection_name
from src.vector.qdrant_store import QdrantStoreManager

__all__ = [
    "QdrantStoreManager",
    "create_embedding_model",
    "get_collection_name",
]
