import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class SourceStatus(str, enum.Enum):
    PENDING = "PENDING"
    CLASSIFYING = "CLASSIFYING"
    PARSING = "PARSING"
    NEEDS_OCR = "NEEDS_OCR"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    READY = "READY"
    FAILED = "FAILED"


class ElementType(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"


class Notebook(Base):
    __tablename__ = "notebooks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    sources: Mapped[list["Source"]] = relationship(
        "Source", back_populates="notebook", cascade="all, delete-orphan"
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    notebook_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("notebooks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), default="application/pdf")
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[SourceStatus] = mapped_column(
        SQLEnum(SourceStatus), default=SourceStatus.PENDING, index=True, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    figures_pending: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    figures_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    figures_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ocr_pages_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ingestion_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_sources_notebook_is_active", "notebook_id", "is_active"),
    )

    notebook: Mapped["Notebook"] = relationship("Notebook", back_populates="sources")
    elements: Mapped[list["Element"]] = relationship(
        "Element", back_populates="source", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="source", cascade="all, delete-orphan"
    )


class Element(Base):
    __tablename__ = "elements"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sources.id", ondelete="CASCADE"), index=True, nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    type: Mapped[ElementType] = mapped_column(SQLEnum(ElementType), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    caption_status: Mapped[str | None] = mapped_column(String(50), default="pending")
    structured_chart_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ocr_applied: Mapped[bool] = mapped_column(default=False, nullable=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parser_version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    source: Mapped["Source"] = relationship("Source", back_populates="elements")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sources.id", ondelete="CASCADE"), index=True, nullable=False
    )
    notebook_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    element_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("elements.id", ondelete="SET NULL"), nullable=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embeddable_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    embedding_model_version: Mapped[str] = mapped_column(String(50), default="bge-m3-v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    source: Mapped["Source"] = relationship("Source", back_populates="chunks")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    notebook_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("notebooks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="complete", nullable=False)  # "complete" | "partial" | "failed"
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
