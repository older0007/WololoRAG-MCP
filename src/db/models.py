"""
SQLAlchemy ORM model mirroring the `articles` table written by the Indexer service
(src/db/models.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArticleStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    INDEXED = "INDEXED"
    OUTDATED = "OUTDATED"
    FAILED = "FAILED"


class Base(DeclarativeBase):
    """Base ORM class."""


class ArticleRecord(Base):
    """Read-model mapping for the shared `articles` SQL table."""

    __tablename__ = "articles"

    wp_post_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    topic: Mapped[str] = mapped_column(String(128), nullable=False, default="General", index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    remote_modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cleaned_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ArticleStatus.DISCOVERED.value,
        index=True,
    )
    vector_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_str: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
