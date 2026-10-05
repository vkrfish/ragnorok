import asyncio
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import StorageAdapter
from app.adapters.embedder import Embedder
from app.adapters.sparse_embedder import SparseEmbedder
from app.adapters.vector_store import VectorStore
from app.core.config import settings
from app.core.logging import logger
from app.domain.models import Chunk, Element, ElementType, Source, SourceStatus
from app.ingestion.chunker import SectionAwareChunker
from app.ingestion.docling_parser import DoclingParser
from app.ingestion.figure_processor import FigureProcessor
from app.ingestion.ocr_processor import OCRProcessor
from app.ingestion.parser import ParsedElement, ParseResult, PyMuPDFParser
from app.ingestion.table_processor import TableProcessor


class IngestionPipeline:
    """Multi-stage async ingestion pipeline with per-page OCR, non-inferential table summaries,
    and bounded concurrency figure captioning.
    """

    def __init__(
        self,
        db: AsyncSession,
        storage: StorageAdapter,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_embedder: SparseEmbedder | None = None,
        parser: Any | None = None,
        chunker: SectionAwareChunker | None = None,
        ocr_processor: OCRProcessor | None = None,
        figure_processor: FigureProcessor | None = None,
        table_processor: TableProcessor | None = None,
    ):
        self.db = db
        self.storage = storage
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_embedder = sparse_embedder or SparseEmbedder()

        if parser:
            self.parser = parser
        elif settings.PARSER_BACKEND == "docling":
            self.parser = DoclingParser()
        else:
            self.parser = PyMuPDFParser()

        self.chunker = chunker or SectionAwareChunker()
        self.ocr_processor = ocr_processor or OCRProcessor()
        self.figure_processor = figure_processor or FigureProcessor(storage=storage)
        self.table_processor = table_processor or TableProcessor()

    async def process_source(self, source_id: str) -> SourceStatus:
        # 1. Fetch Source
        result = await self.db.execute(select(Source).where(Source.id == source_id))
        source = result.scalar_one_or_none()
        if not source:
            raise ValueError(f"Source '{source_id}' not found.")

        try:
            # Stage 1: PARSING Text & Tables
            source.status = SourceStatus.PARSING
            await self.db.commit()

            file_bytes = await self.storage.get(source.file_path)
            parse_result: ParseResult = self.parser.parse(file_bytes, source.filename)

            # Clear stale elements or chunks for idempotency
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

            # Save Parsed Elements to DB
            db_elements: list[Element] = []
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
                    ocr_applied=pe.ocr_applied,
                    ocr_confidence=pe.ocr_confidence,
                )
                self.db.add(elem)
                db_elements.append(elem)

            await self.db.commit()
            for elem in db_elements:
                await self.db.refresh(elem)

            # Stage 2: PER-PAGE OCR (for low-text pages)
            if parse_result.needs_ocr_pages:
                source.status = SourceStatus.NEEDS_OCR
                await self.db.commit()

                ocr_elements_added = 0
                for p_num in parse_result.needs_ocr_pages:
                    ocr_parsed = await self.ocr_processor.process_page(file_bytes, p_num)
                    for ope in ocr_parsed:
                        elem = Element(
                            id=str(uuid.uuid4()),
                            source_id=source_id,
                            page_number=ope.page_number,
                            bbox=ope.bbox,
                            type=ope.type,
                            content=ope.content,
                            ocr_applied=ope.ocr_applied,
                            ocr_confidence=ope.ocr_confidence,
                        )
                        self.db.add(elem)
                        db_elements.append(elem)
                        ocr_elements_added += 1

                source.ocr_pages_count = len(parse_result.needs_ocr_pages)
                await self.db.commit()
                for elem in db_elements:
                    await self.db.refresh(elem)

            # Stage 3: CHUNKING & EMBEDDING TEXT & TABLES (Immediate searchability)
            source.status = SourceStatus.CHUNKING
            await self.db.commit()

            parsed_list_for_chunking = [
                ParsedElement(
                    page_number=e.page_number,
                    bbox=e.bbox or {},
                    type=e.type,
                    content=e.content,
                    summary=e.summary or "",
                    caption_status=e.caption_status or "pending",
                )
                for e in db_elements
            ]

            chunks_data = self.chunker.chunk_elements(parsed_list_for_chunking)
            embeddable_chunks_data: list[dict] = []
            texts_to_embed: list[str] = []

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

            source.status = SourceStatus.EMBEDDING
            await self.db.commit()

            if texts_to_embed:
                vectors = await self.embedder.embed_batch(texts_to_embed)
                sparse_vecs = self.sparse_embedder.embed_batch(texts_to_embed)
                await self.vector_store.upsert_chunks(
                    chunks=embeddable_chunks_data,
                    vectors=vectors,
                    sparse_vectors=sparse_vecs,
                )

            # Document is searchable early for text & tables!
            source.status = SourceStatus.READY
            await self.db.commit()
            logger.info(f"Source {source_id} text and tables indexed into Qdrant successfully.")

            # Stage 4: ASYNC BOUNDED CONCURRENCY FIGURE CAPTIONING
            figure_elements = [e for e in db_elements if e.type == ElementType.FIGURE][: settings.MAX_FIGURES_PER_DOC]
            source.figures_pending = len(figure_elements)
            source.figures_completed = 0
            source.figures_failed = 0
            await self.db.commit()

            if figure_elements:
                semaphore = asyncio.Semaphore(settings.MAX_CAPTION_CONCURRENCY)

                async def process_single_figure(fig_elem: Element):
                    async with semaphore:
                        try:
                            # 1. Crop region
                            crop_bytes, image_hash = self.figure_processor.crop_figure_region(
                                pdf_bytes=file_bytes,
                                page_number=fig_elem.page_number,
                                bbox=fig_elem.bbox or {"x0": 0.0, "y0": 0.0, "x1": 500.0, "y1": 500.0},
                            )

                            # 2. Save crop image through StorageAdapter interface
                            image_path = await self.figure_processor.save_crop_image(
                                notebook_id=source.notebook_id,
                                source_id=source_id,
                                element_id=fig_elem.id,
                                crop_bytes=crop_bytes,
                            )

                            # 3. Vision LLM captioning & structured chart extraction
                            pdf_caption = fig_elem.summary or ""
                            caption_text, chart_json, model_name, prompt_ver = await self.figure_processor.process_figure_caption(
                                crop_bytes=crop_bytes,
                                image_hash=image_hash,
                                pdf_caption=pdf_caption,
                            )

                            fig_elem.summary = caption_text
                            fig_elem.image_path = image_path
                            fig_elem.image_hash = image_hash
                            fig_elem.structured_chart_json = chart_json
                            fig_elem.model_name = model_name
                            fig_elem.prompt_version = prompt_ver
                            fig_elem.caption_status = "completed"

                            source.figures_completed += 1

                        except Exception as e:
                            logger.error(f"Figure {fig_elem.id} captioning failed: {e}")
                            fig_elem.caption_status = "failed"
                            source.figures_failed += 1
                        finally:
                            source.figures_pending = max(0, source.figures_pending - 1)

                # Launch figure tasks concurrently
                await asyncio.gather(*(process_single_figure(fe) for fe in figure_elements))
                await self.db.commit()

                # Re-embed completed figures into Qdrant
                completed_figures = [fe for fe in figure_elements if fe.caption_status == "completed"]
                if completed_figures:
                    fig_parsed = [
                        ParsedElement(
                            page_number=fe.page_number,
                            bbox=fe.bbox or {},
                            type=ElementType.FIGURE,
                            content=fe.content,
                            summary=fe.summary or "",
                            caption_status="completed",
                        )
                        for fe in completed_figures
                    ]
                    fig_chunks = self.chunker.chunk_elements(fig_parsed)
                    fig_embeddable = []
                    fig_texts = []
                    for idx, fc in enumerate(fig_chunks):
                        if fc.is_embeddable and fc.embeddable_text.strip():
                            fe_id = completed_figures[idx].id if idx < len(completed_figures) else None
                            c_id = str(uuid.uuid4())
                            fig_embeddable.append({
                                "id": c_id,
                                "source_id": source_id,
                                "element_id": fe_id,
                                "page_number": fc.page_number,
                                "notebook_id": source.notebook_id,
                                "user_id": source.user_id,
                            })
                            fig_texts.append(fc.embeddable_text)

                    if fig_texts:
                        fig_vecs = await self.embedder.embed_batch(fig_texts)
                        fig_sparse_vecs = self.sparse_embedder.embed_batch(fig_texts)
                        await self.vector_store.upsert_chunks(
                            chunks=fig_embeddable,
                            vectors=fig_vecs,
                            sparse_vectors=fig_sparse_vecs,
                        )

            logger.info(f"Ingestion pipeline fully completed for source {source_id}: READY")
            return SourceStatus.READY

        except Exception as e:
            source.status = SourceStatus.FAILED
            source.error_message = str(e)
            await self.db.commit()
            logger.error(f"Ingestion pipeline failed for source {source_id}: {e!s}")
            return SourceStatus.FAILED
