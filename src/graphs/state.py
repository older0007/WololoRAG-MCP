"""
Typed state schema for the Retrieval LangGraph (src/graphs/state.py).
"""

from __future__ import annotations

from typing import Any, TypedDict


class RetrievalState(TypedDict, total=False):
    """Shared state passed through each node in the RAG Retrieval Graph."""

    query: str
    topic_filter: str | None
    top_k: int | None
    published_after: str | None
    query_vector: list[float]
    raw_hits: list[dict[str, Any]]
    enriched_hits: list[dict[str, Any]]
    formatted_context: str
