"""
Read-only Repository over PostgreSQL or SQLite for WololoRAG-MCP (src/db/repository.py).

Notice this repository exposes ONLY read/query methods (`get_stats`, `get_by_ids`,
`get_failed_articles`) because CQRS separates writes (`WololoRAG` Indexer) from
reads (`WololoRAG-MCP` Search Server).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import ArticleRecord, ArticleStatus, Base


class DatabaseStats(BaseModel):
    """Snapshot of SQL article synchronization metrics."""

    total_in_sql: int = 0
    discovered_count: int = 0
    indexed_count: int = 0
    indexed_for_active_provider: int = 0
    indexed_other_provider: int = 0
    outdated_count: int = 0
    failed_count: int = 0


class ArticleRepository:
    """Thread-safe read repository supporting both PostgreSQL and SQLite."""

    def __init__(self, db_target: str | Path) -> None:
        target_str = str(db_target).strip()
        if "://" in target_str:
            self.db_url = target_str
            if self.db_url.startswith("sqlite:///") and not self.db_url.startswith("sqlite:///:memory:"):
                raw_path = Path(self.db_url.removeprefix("sqlite:///"))
                raw_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            db_path = Path(target_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_url = f"sqlite:///{db_path}"

        is_sqlite = self.db_url.startswith("sqlite:")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine = create_engine(
            self.db_url,
            connect_args=connect_args,
            echo=False,
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        # Ensure table exists safely if MCP starts before the Indexer runs.
        Base.metadata.create_all(self.engine)

    @property
    def backend_name(self) -> str:
        if self.db_url.startswith("postgresql"):
            return "PostgreSQL"
        return "SQLite"

    def _session(self) -> Session:
        return self._session_factory()

    def get_stats(self, active_provider: str) -> DatabaseStats:
        """Aggregate row counts by sync status and embedding provider."""
        with self._session() as session:
            rows = session.execute(
                select(ArticleRecord.status, ArticleRecord.embedding_provider, func.count())
                .group_by(ArticleRecord.status, ArticleRecord.embedding_provider)
            ).all()

        stats = DatabaseStats()
        for status_val, provider_val, count in rows:
            stats.total_in_sql += count
            if status_val == ArticleStatus.DISCOVERED.value:
                stats.discovered_count += count
            elif status_val == ArticleStatus.INDEXED.value:
                stats.indexed_count += count
                if provider_val == active_provider:
                    stats.indexed_for_active_provider += count
                else:
                    stats.indexed_other_provider += count
            elif status_val == ArticleStatus.OUTDATED.value:
                stats.outdated_count += count
            elif status_val == ArticleStatus.FAILED.value:
                stats.failed_count += count

        return stats

    def get_by_ids(self, wp_post_ids: Iterable[int]) -> dict[int, ArticleRecord]:
        """Hydrate multiple articles by WordPress post ID in a single SQL query."""
        ids = list(set(wp_post_ids))
        if not ids:
            return {}
        with self._session() as session:
            stmt = select(ArticleRecord).where(ArticleRecord.wp_post_id.in_(ids))
            rows = session.scalars(stmt).all()
            return {row.wp_post_id: row for row in rows}

    def get_failed_articles(self, limit: int = 20) -> list[ArticleRecord]:
        """Return recent FAILED rows with `error_str` for diagnostics."""
        with self._session() as session:
            stmt = (
                select(ArticleRecord)
                .where(ArticleRecord.status == ArticleStatus.FAILED.value)
                .order_by(ArticleRecord.updated_at.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())
