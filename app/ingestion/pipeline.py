import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.adapters.base import StorageAdapter
from app.adapters.embedder import Embedder
from app.adapters.vector_store import VectorStore
from app.core.logging import logger
from app.domain.models import Source, SourceStatus, Element, Chunk
from app.ingestion.parser import PyMuPDFParser
from app.ingestion.chunker import SectionAwareChunker


class IngestionPipeline:
    """Async Ingestion Pipeline running per-stage processing and DB persistence."""

    def __init__(
        self,
        db: AsyncSession,
        storage: StorageAdapter,
        embedder: Embedder,
        vector_store: VectorStore,
        parser: Optional[PyMuPDFParser] = None,
        chunker: Optional[SectionAwareChunker] = None,
    ):
        self.db = db
        self.storage = storage
        self.embedder = embedder
        self.vector_store = vector_store
        self.parser = parser or PyMuPDFParser()
        self.chunker = chunker or SectionAwareChunker()

    async def process_source(self, source_id: str) -> SourceStatus:
        # 1. Fetch Source from DB
        result = await self.db.execute(select(Source).where(Source.id == source_id))
        source = result.scalar_one_or_none()
        if not source:
            raise ValueError(f"Source '{source_id}' not found.")

        try:
            # Stage 1: CLASSIFYING & PARSING
            source.status = SourceStatus.PARSING
            await self.db.commit()

            file_bytes = await self.storage.get(source.file_path)
            parse_result = self.parser.parse(file_bytes, source.filename)

            if parse_result.needs_ocr:
                source.status = SourceStatus.NEEDS_OCR
                source.error_message = "Document requires OCR parsing."
                await self.db.commit()
                logger.info(f"Source {source_id} flagged as NEEDS_OCR")
                return SourceStatus.NEEDS_OCR

            # Clear any stale elements or chunks for idempotency / re-run
            existing_elements = await self.db.execute(
                select(Element).where(Element.source_id == source_id)
            )
            for el in existing_elements.scalars().all():
                await self.db.delete(el)

            existing_chunks = await self.db.execute(
                select(Chunk).where(Chunk.source_id == source_id)
            )
            for ch in existing_chunks.scalars().all():
                await self.db.delete(ch)

            await self.db.commit()

            # Save Elements to DB
            db_elements: List[Element] = []
            for pe in parse_result.elements:
                elem = Element(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    page_number=pe.page_number,
                    bbox=pe.bbox,
                    type=pe.type,
                    content=pe.content,
                    summary=pe.summary if pe.summary else None,
                    caption_status=pe.caption_status,
                )
                self.db.add(elem)
                db_elements.append(elem)

            await self.db.commit()
            for elem in db_elements:
                await self.db.refresh(elem)

            # Stage 2: CHUNKING
            source.status = SourceStatus.CHUNKING
            await self.db.commit()

            chunks_data = self.chunker.chunk_elements(parse_result.elements)
            db_chunks: List[Chunk] = []
            embeddable_chunks_data: List[dict] = []
            texts_to_embed: List[str] = []

            for cd in chunks_data:
                matching_elem_id = (
                    db_elements[cd.element_index].id
                    if cd.element_index < len(db_elements)
                    else None
                )
                chunk_id = str(uuid.uuid4())
                chunk_orm = Chunk(
                    id=chunk_id,
                    source_id=source_id,
                    notebook_id=source.notebook_id,
                    user_id=source.user_id,
                    element_id=matching_elem_id,
                    page_number=cd.page_number,
                    content=cd.content,
                    embeddable_text=cd.embeddable_text,
                    token_count=cd.token_count,
                    heading_path=cd.heading_path,
                    bbox=cd.bbox,
                )
                self.db.add(chunk_orm)
                db_chunks.append(chunk_orm)

                if cd.is_embeddable and cd.embeddable_text.strip():
                    embeddable_chunks_data.append({
                        "id": chunk_id,
                        "source_id": source_id,
                        "element_id": matching_elem_id,
                        "page_number": cd.page_number,
                        "notebook_id": source.notebook_id,
                        "user_id": source.user_id,
                    })
                    texts_to_embed.append(cd.embeddable_text)

            await self.db.commit()

            # Stage 3: EMBEDDING & VECTOR STORE UPSERT
            source.status = SourceStatus.EMBEDDING
            await self.db.commit()

            if texts_to_embed:
                vectors = await self.embedder.embed_batch(texts_to_embed)
                await self.vector_store.upsert_chunks(
                    chunks=embeddable_chunks_data,
                    vectors=vectors,
                )

            # Stage 4: READY
            source.status = SourceStatus.READY
            source.error_message = None
            await self.db.commit()
            logger.info(f"Ingestion pipeline completed for source {source_id}: READY")
            return SourceStatus.READY

        except Exception as e:
            source.status = SourceStatus.FAILED
            source.error_message = str(e)
            await self.db.commit()
            logger.error(f"Ingestion pipeline failed for source {source_id}: {str(e)}")
            return SourceStatus.FAILED
