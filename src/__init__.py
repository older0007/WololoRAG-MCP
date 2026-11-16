"""
WololoRAG-MCP root package.

This microservice is strictly responsible for the READ / QUERY side (CQRS):
- `src/config.py`: Pydantic configuration for embeddings, databases, search, and MCP auth.
- `src/db/`: Read-only SQLAlchemy layer (PostgreSQL or SQLite).
- `src/vector/`: Embedding factory + read-only Qdrant search manager.
- `src/graphs/`: LangGraph RetrievalStateGraph.
- `src/mcp_server/`: MCPServer HTTP/SSE/stdio endpoints + OptionalTokenAuthMiddleware.
"""
