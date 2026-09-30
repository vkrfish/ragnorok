from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.domain.models import SourceStatus, ElementType


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
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Element & Chunk Schemas
class ElementResponse(BaseModel):
    id: str
    source_id: str
    page_number: int
    bbox: Optional[Dict[str, Any]] = None
    type: ElementType
    content: str
    summary: Optional[str] = None
    image_path: Optional[str] = None
    caption_status: Optional[str] = None
    parser_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkResponse(BaseModel):
    id: str
    source_id: str
    notebook_id: str
    user_id: str
    element_id: Optional[str] = None
    page_number: int
    content: str
    embeddable_text: str
    token_count: int
    heading_path: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    embedding_model_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ListSourcesResponse(BaseModel):
    sources: List[SourceResponse]
    total: int


# Chat & Citation Schemas
class Citation(BaseModel):
    handle: str                 # e.g., "[1]"
    source_id: str
    page: int
    element_id: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    snippet: str


class ChatQueryRequest(BaseModel):
    query: str
    relevance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class ChatQueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    prompt_version: str = "qa_v1"
    model_name: str = "claude-3-5-sonnet"
