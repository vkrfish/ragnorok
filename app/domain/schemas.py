from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import ElementType, SourceStatus


class TenantScope(BaseModel):
    user_id: str
    notebook_id: str


# Notebook Schemas
class NotebookCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, json_schema_extra={"example": "My Research Notebook"})


class NotebookResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ImagePart(BaseModel):
    bytes: bytes
    mime_type: str = "image/png"


# Source Schemas
class SourceResponse(BaseModel):
    id: str
    notebook_id: str
    user_id: str
    filename: str
    content_hash: str
    file_path: str
    mime_type: str
    file_size: int
    status: SourceStatus
    error_message: str | None = None
    figures_pending: int = 0
    figures_completed: int = 0
    figures_failed: int = 0
    ocr_pages_count: int = 0
    ingestion_cost_usd: float = 0.0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Element & Chunk Schemas
class ElementResponse(BaseModel):
    id: str
    source_id: str
    page_number: int
    bbox: dict[str, Any] | None = None
    type: ElementType
    content: str
    summary: str | None = None
    image_path: str | None = None
    image_hash: str | None = None
    caption_status: str | None = None
    structured_chart_json: dict[str, Any] | None = None
    ocr_applied: bool = False
    ocr_confidence: float | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    parser_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkResponse(BaseModel):
    id: str
    source_id: str
    notebook_id: str
    user_id: str
    element_id: str | None = None
    page_number: int
    content: str
    embeddable_text: str
    token_count: int
    heading_path: str | None = None
    bbox: dict[str, Any] | None = None
    embedding_model_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ListSourcesResponse(BaseModel):
    sources: list[SourceResponse]
    total: int


# Chat & Citation Schemas
class Citation(BaseModel):
    handle: str                 # e.g., "[1]"
    source_id: str
    page: int
    element_id: str | None = None
    bbox: dict[str, Any] | None = None
    snippet: str


class ChatQueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    relevance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class ChatQueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    conversation_id: str | None = None
    is_refusal: bool = False
    refusal_reason: str | None = None
    prompt_version: str = "qa_v1"
    model_name: str = "gemini-3.5-flash-lite"
    latencies_ms: dict[str, float] = Field(default_factory=dict)
