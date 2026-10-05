import asyncio
import hashlib
import io
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import StorageAdapter
from app.adapters.factory import get_storage_adapter
from app.core.db import get_db
from app.core.errors import NotFoundError
from app.core.logging import logger
from app.domain.models import Element, Notebook, Source, SourceStatus
from app.domain.schemas import (
    ElementResponse,
    ListSourcesResponse,
    PageElementResponse,
    PageMetadataResponse,
    SourceResponse,
    SourceUpdateRequest,
)

router = APIRouter()


def get_user_id(x_user_id: str | None = Header(None, alias="X-User-ID")) -> str:
    """Extract user_id for tenant isolation boundary."""
    return x_user_id or "default_user"


@router.post(
    "/notebooks/{notebook_id}/sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_source(
    notebook_id: str,
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
    storage: StorageAdapter = Depends(get_storage_adapter),
):
    """Upload a source PDF asynchronously and idempotently (deduplicated by content_hash and Idempotency-Key)."""
    # 1. Verify notebook existence & tenant ownership (return 404 for other users' resources)
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
        is_active=True,
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


@router.get("/notebooks/{notebook_id}/sources", response_model=ListSourcesResponse)
async def list_sources(
    notebook_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """List all sources in a notebook with full status metadata."""
    # Verify notebook ownership
    nb_res = await db.execute(select(Notebook).where(Notebook.id == notebook_id, Notebook.user_id == user_id))
    if not nb_res.scalar_one_or_none():
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")

    result = await db.execute(
        select(Source).where(
            Source.notebook_id == notebook_id,
            Source.user_id == user_id,
        )
    )
    sources = list(result.scalars().all())
    return ListSourcesResponse(
        sources=[SourceResponse.model_validate(s) for s in sources],
        total=len(sources),
    )


@router.get("/notebooks/{notebook_id}/sources/{source_id}", response_model=SourceResponse)
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


@router.patch("/notebooks/{notebook_id}/sources/{source_id}", response_model=SourceResponse)
async def update_source_active_state(
    notebook_id: str,
    source_id: str,
    payload: SourceUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Idempotent update to enable or disable source in active retrieval context."""
    result = await db.execute(
        select(Source).where(
            Source.id == source_id,
            Source.notebook_id == notebook_id,
            Source.user_id == user_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise NotFoundError(f"Source '{source_id}' not found.")

    source.is_active = payload.is_active
    await db.commit()
    await db.refresh(source)
    logger.info(f"Updated source_id={source_id} is_active={source.is_active}")
    return source


@router.delete("/notebooks/{notebook_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    notebook_id: str,
    source_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
    storage: StorageAdapter = Depends(get_storage_adapter),
):
    """Cascading delete source, elements, chunks, and cached page images."""
    result = await db.execute(
        select(Source).where(
            Source.id == source_id,
            Source.notebook_id == notebook_id,
            Source.user_id == user_id,
        )
    )
    source = result.scalar_one_or_none()
    if not source:
        raise NotFoundError(f"Source '{source_id}' not found.")

    # Delete storage file
    try:
        await storage.delete(source.file_path)
    except Exception as e:
        logger.warning(f"Failed to delete storage file {source.file_path}: {e}")

    await db.delete(source)
    await db.commit()
    return FastAPIResponse(status_code=status.HTTP_204_NO_CONTENT)


# standalone route for element details
@router.get("/sources/{source_id}/elements/{element_id}", response_model=ElementResponse)
async def get_element_detail(
    source_id: str,
    element_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Get element detail with bounding box and summary."""
    result = await db.execute(
        select(Element)
        .join(Source, Element.source_id == Source.id)
        .where(Element.id == element_id, Source.id == source_id, Source.user_id == user_id)
    )
    elem = result.scalar_one_or_none()
    if not elem:
        raise NotFoundError(f"Element '{element_id}' not found.")
    return elem


# Split Page Metadata Endpoint (JSON)
@router.get("/sources/{source_id}/pages/{page_number}", response_model=PageMetadataResponse)
async def get_page_metadata(
    source_id: str,
    page_number: int,
    dpi: int = Query(150, ge=72, le=300),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
    storage: StorageAdapter = Depends(get_storage_adapter),
):
    """Get page metadata JSON: PDF point dimensions, rotation, image pixel size, DPI, and layout elements with bboxes."""
    src_res = await db.execute(select(Source).where(Source.id == source_id, Source.user_id == user_id))
    source = src_res.scalar_one_or_none()
    if not source:
        raise NotFoundError(f"Source '{source_id}' not found.")

    file_bytes = await storage.get(source.file_path)
    if not file_bytes:
        raise NotFoundError("Source PDF file bytes not found.")

    # Render PDF page metadata using PyMuPDF in thread pool
    def _get_meta() -> dict[str, Any]:
        import fitz  # type: ignore[import-untyped]
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if page_number < 1 or page_number > len(doc):
            raise NotFoundError(f"Page {page_number} out of range (1..{len(doc)}).")
        page = doc[page_number - 1]
        rect = page.rect
        pix = page.get_pixmap(dpi=dpi)
        return {
            "width_points": float(rect.width),
            "height_points": float(rect.height),
            "rotation": int(page.rotation),
            "image_width_px": int(pix.width),
            "image_height_px": int(pix.height),
        }

    meta = await asyncio.to_thread(_get_meta)

    elem_res = await db.execute(
        select(Element).where(Element.source_id == source_id, Element.page_number == page_number)
    )
    elements = list(elem_res.scalars().all())

    page_elements = [
        PageElementResponse(
            id=e.id,
            type=e.type,
            content=e.content,
            bbox=e.bbox,
            summary=e.summary,
            image_path=e.image_path,
        )
        for e in elements
    ]

    return PageMetadataResponse(
        source_id=source_id,
        page_number=page_number,
        width_points=meta["width_points"],
        height_points=meta["height_points"],
        rotation=meta["rotation"],
        image_width_px=meta["image_width_px"],
        image_height_px=meta["image_height_px"],
        dpi=dpi,
        elements=page_elements,
    )


# Split Page Image Endpoint (PNG)
@router.get("/sources/{source_id}/pages/{page_number}/image")
async def get_page_image(
    source_id: str,
    page_number: int,
    request: Request,
    dpi: int = Query(150, ge=72, le=300),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
    storage: StorageAdapter = Depends(get_storage_adapter),
):
    """Render and serve rendered page image PNG (cached via Storage interface with ETag)."""
    src_res = await db.execute(select(Source).where(Source.id == source_id, Source.user_id == user_id))
    source = src_res.scalar_one_or_none()
    if not source:
        raise NotFoundError(f"Source '{source_id}' not found.")

    cache_key = f"page_cache/{source_id}_p{page_number}_dpi{dpi}.png"

    # Check Storage Cache
    try:
        cached_bytes = await storage.get(cache_key)
        if cached_bytes:
            etag = f'"{hashlib.md5(cached_bytes).hexdigest()}"'
            if request.headers.get("if-none-match") == etag:
                return FastAPIResponse(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag, "Cache-Control": "private, max-age=86400"})
            return FastAPIResponse(
                content=cached_bytes,
                media_type="image/png",
                headers={
                    "Cache-Control": "private, max-age=86400",
                    "ETag": etag,
                },
            )
    except Exception:
        pass

    file_bytes = await storage.get(source.file_path)
    if not file_bytes:
        raise NotFoundError("Source PDF file bytes not found.")

    def _render_png() -> bytes:
        import fitz  # type: ignore[import-untyped]
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if page_number < 1 or page_number > len(doc):
            raise NotFoundError(f"Page {page_number} out of range.")
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    png_bytes = await asyncio.to_thread(_render_png)

    # Save to storage cache asynchronously
    try:
        await storage.save(
            file_obj=io.BytesIO(png_bytes),
            notebook_id=source.notebook_id,
            source_id=source_id,
            filename=f"p{page_number}_dpi{dpi}.png",
            content_type="image/png",
        )
    except Exception as e:
        logger.warning(f"Page image cache write failed: {e}")

    etag = f'"{hashlib.md5(png_bytes).hexdigest()}"'
    if request.headers.get("if-none-match") == etag:
        return FastAPIResponse(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag, "Cache-Control": "private, max-age=86400"})

    return FastAPIResponse(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "private, max-age=86400",
            "ETag": etag,
        },
    )
