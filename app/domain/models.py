import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    String,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    Text,
    JSON,
    BigInteger,
    Float,
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

    sources: Mapped[List["Source"]] = relationship(
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
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    notebook: Mapped["Notebook"] = relationship("Notebook", back_populates="sources")
    elements: Mapped[List["Element"]] = relationship(
        "Element", back_populates="source", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["Chunk"]] = relationship(
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
    bbox: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    type: Mapped[ElementType] = mapped_column(SQLEnum(ElementType), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    caption_status: Mapped[Optional[str]] = mapped_column(String(50), default="pending")
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
    element_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("elements.id", ondelete="SET NULL"), nullable=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embeddable_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bbox: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    embedding_model_version: Mapped[str] = mapped_column(String(50), default="bge-m3-v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    source: Mapped["Source"] = relationship("Source", back_populates="chunks")
