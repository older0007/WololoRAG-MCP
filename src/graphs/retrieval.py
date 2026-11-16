"""
LangGraph Retrieval StateGraph (src/graphs/retrieval.py).

Graph Topology:
    START -> embed_query -> search_qdrant -> hydrate_with_sql -> format_context -> END
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph

from src.config import FullSettings
from src.db.repository import ArticleRepository
from src.graphs.state import RetrievalState
from src.vector.qdrant_store import QdrantStoreManager


def build_retrieval_graph(
    settings: FullSettings,
    repo: ArticleRepository,
    embeddings: Embeddings,
    qdrant_store: QdrantStoreManager,
):
    """Build and compile the semantic RAG Retrieval StateGraph."""

    def embed_query(state: RetrievalState) -> dict:
        query_text = state["query"].strip()
        if not query_text:
            return {"query_vector": []}
        return {"query_vector": embeddings.embed_query(query_text)}

    def search_qdrant(state: RetrievalState) -> dict:
        query_vec = state.get("query_vector", [])
        if not query_vec:
            return {"raw_hits": []}

        hits = qdrant_store.search(
            query_vector=query_vec,
            top_k=state.get("top_k") or settings.vector.search.top_k,
            score_threshold=settings.vector.search.score_threshold,
            topic_filter=state.get("topic_filter"),
            query_text=state.get("query", ""),
            published_after=state.get("published_after"),
        )
        return {"raw_hits": hits}

    def hydrate_with_sql(state: RetrievalState) -> dict:
        raw_hits = state.get("raw_hits", [])
        if not raw_hits:
            return {"enriched_hits": []}

        post_ids = [
            int(hit["wp_post_id"])
            for hit in raw_hits
            if hit.get("wp_post_id") is not None
        ]
        sql_records = repo.get_by_ids(post_ids)

        enriched: list[dict[str, Any]] = []
        for hit in raw_hits:
            post_id = hit.get("wp_post_id")
            sql_row = sql_records.get(int(post_id)) if post_id is not None else None
            enriched.append(
                {
                    **hit,
                    "sql_status": sql_row.status if sql_row else "UNKNOWN",
                    "published_at": (
                        sql_row.published_at.isoformat()
                        if sql_row and sql_row.published_at
                        else hit.get("published_at")
                    ),
                    "remote_modified_at": (
                        sql_row.remote_modified_at.isoformat()
                        if sql_row and sql_row.remote_modified_at
                        else None
                    ),
                    "full_content": sql_row.cleaned_content if sql_row else None,
                }
            )
        return {"enriched_hits": enriched}

    def format_context(state: RetrievalState) -> dict:
        hits = state.get("enriched_hits", [])
        if not hits:
            return {
                "formatted_context": "No matching Wololo.net articles found above the similarity threshold."
            }

        context_mode = settings.vector.search.context_mode
        sections: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            pub_display = hit.get("published_at") or "unknown"
            header = (
                f"[{idx}] {hit['title']} "
                f"(Topic: {hit['topic']} | Post ID: {hit['wp_post_id']} | "
                f"Published: {pub_display[:10]} | "
                f"Chunk: {hit['chunk_index'] + 1}/{hit['total_chunks']} | "
                f"Score: {hit['score']:.3f} | Status: {hit['sql_status']})"
            )
            url_line = f"URL: {hit['url']}"

            if context_mode == "full_article" and hit.get("full_content"):
                body = (
                    f"[Mode: full_article — matched chunk {hit['chunk_index'] + 1}/{hit['total_chunks']}]\n\n"
                    f"Title: {hit['title']}\nTopic: {hit['topic']}\n\n"
                    f"{str(hit['full_content']).strip()}"
                )
            else:
                body = (
                    f"{hit['text'].strip()}\n\n"
                    f"(Tip: call get_article_details(wp_post_id={hit['wp_post_id']}) for full article text)"
                )

            sections.append(f"{header}\n{url_line}\n\n{body}")

        return {"formatted_context": "\n\n---\n\n".join(sections)}

    graph = StateGraph(RetrievalState)
    graph.add_node("embed_query", embed_query)
    graph.add_node("search_qdrant", search_qdrant)
    graph.add_node("hydrate_with_sql", hydrate_with_sql)
    graph.add_node("format_context", format_context)

    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "search_qdrant")
    graph.add_edge("search_qdrant", "hydrate_with_sql")
    graph.add_edge("hydrate_with_sql", "format_context")
    graph.add_edge("format_context", END)

    return graph.compile()


def run_retrieval_query(
    query: str,
    settings: FullSettings,
    repo: ArticleRepository,
    embeddings: Embeddings,
    qdrant_store: QdrantStoreManager,
    topic_filter: str | None = None,
    top_k: int | None = None,
    published_after: str | None = None,
) -> RetrievalState:
    compiled = build_retrieval_graph(settings, repo, embeddings, qdrant_store)
    return compiled.invoke(
        {
            "query": query,
            "topic_filter": topic_filter,
            "top_k": top_k,
            "published_after": published_after,
        }
    )
