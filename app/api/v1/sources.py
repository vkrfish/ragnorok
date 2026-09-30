import hashlib
import io
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Header, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.adapters.factory import get_storage_adapter
from app.adapters.base import StorageAdapter
from app.core.db import get_db
from app.core.errors import NotFoundError, RagnorkException
from app.core.logging import logger
from app.domain.models import Notebook, Source, SourceStatus
from app.domain.schemas import ListSourcesResponse, SourceResponse

router = APIRouter()


def get_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-ID")) -> str:
    """Extract user_id for tenant isolation boundary."""
    return x_user_id or "default_user"


@router.post(
    "/{notebook_id}/sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_source(
    notebook_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
    storage: StorageAdapter = Depends(get_storage_adapter),
):
    """Upload a source PDF asynchronously and idempotently (deduplicated by content_hash)."""
    # 1. Verify notebook existence & tenant ownership
    nb_result = await db.execute(
        select(Notebook).where(
            Notebook.id == notebook_id,
            Notebook.user_id == user_id,
        )
    )
    notebook = nb_result.scalar_one_or_none()
    if not notebook:
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")

    # 2. Read file & compute SHA-256 hash in 64KB chunks
    sha256_hash = hashlib.sha256()
    file_bytes_io = io.BytesIO()
    file_size = 0

    while chunk := await file.read(64 * 1024):
        sha256_hash.update(chunk)
        file_bytes_io.write(chunk)
        file_size += len(chunk)

    content_hash = sha256_hash.hexdigest()
    file_bytes_io.seek(0)

    # 3. Idempotency Check: search for existing source with same content_hash in notebook
    existing_result = await db.execute(
        select(Source).where(
            Source.notebook_id == notebook_id,
            Source.user_id == user_id,
            Source.content_hash == content_hash,
        )
    )
    existing_source = existing_result.scalar_one_or_none()

    if existing_source:
        logger.info(
            f"Idempotent upload hit: content_hash={content_hash} already exists as source_id={existing_source.id}"
        )
        return existing_source

    # 4. Generate source ID and save file using configured storage adapter
    filename = file.filename or "uploaded_document.pdf"
    source_id = str(uuid.uuid4())
    
    file_path = await storage.save(
        file_obj=file_bytes_io,
        notebook_id=notebook_id,
        source_id=source_id,
        filename=filename,
        content_type=file.content_type or "application/pdf",
    )

    # 5. Persist Source record in DB with PENDING status
    source = Source(
        id=source_id,
        notebook_id=notebook_id,
        user_id=user_id,
        filename=filename,
        content_hash=content_hash,
        file_path=file_path,
        mime_type=file.content_type or "application/pdf",
        file_size=file_size,
        status=SourceStatus.PENDING,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)

    logger.info(
        f"Created source_id={source.id} for notebook_id={notebook_id}, status={source.status}",
        extra={"source_id": source.id, "stage": "ingestion_upload"},
    )

    # 6. Run Ingestion Pipeline (parse layout, save elements/chunks, embed, upsert to vector store)
    try:
        from app.ingestion.pipeline import IngestionPipeline
        from app.workers.tasks import shared_embedder, shared_vector_store

        pipeline = IngestionPipeline(
            db=db,
            storage=storage,
            embedder=shared_embedder,
            vector_store=shared_vector_store,
        )
        await pipeline.process_source(source.id)
        await db.refresh(source)
    except Exception as e:
        logger.error(f"Automatic ingestion failed for source {source.id}: {e}")

    return source


@router.get("/{notebook_id}/sources", response_model=ListSourcesResponse)
async def list_sources(
    notebook_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """List all sources in a notebook."""
    result = await db.execute(
        select(Source).where(
            Source.notebook_id == notebook_id,
            Source.user_id == user_id,
        )
    )
    sources = result.scalars().all()
    return ListSourcesResponse(sources=list(sources), total=len(sources))


@router.get("/{notebook_id}/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    notebook_id: str,
    source_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Get source status and details by source_id."""
    result = await db.execute(
        select(Source).where(
            Source.id == source_id,
            Source.notebook_id == notebook_id,
            Source.user_id == user_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise NotFoundError(f"Source '{source_id}' not found in notebook '{notebook_id}'.")
    return source
