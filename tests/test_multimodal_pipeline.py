import io
import os

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.adapters.embedder import FakeEmbedder
from app.adapters.llm import FakeLLMGenerator
from app.adapters.local_storage import LocalStorageAdapter
from app.adapters.vector_store import InMemoryVectorStore
from app.core.db import Base
from app.domain.models import Element, ElementType, Notebook, Source, SourceStatus
from app.ingestion.figure_processor import FigureProcessor
from app.ingestion.pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_multimodal_pipeline_execution():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        notebook = Notebook(id="nb-mm-1", user_id="user-1", title="Multimodal Notebook")
        db.add(notebook)
        await db.commit()

        storage = LocalStorageAdapter(base_dir="./media")
        pdf_path = os.path.join("sample_pdfs", "vector_chart.pdf")
        with open(pdf_path, "rb") as f:  # noqa: ASYNC230
            pdf_bytes = f.read()

        stored_file_path = await storage.save(
            file_obj=io.BytesIO(pdf_bytes),
            notebook_id="nb-mm-1",
            source_id="src-mm-1",
            filename="vector_chart.pdf",
        )

        source = Source(
            id="src-mm-1",
            notebook_id="nb-mm-1",
            user_id="user-1",
            filename="vector_chart.pdf",
            content_hash="hash-vector-123",
            file_path=stored_file_path,
            file_size=len(pdf_bytes),
            status=SourceStatus.PENDING,
        )
        db.add(source)
        await db.commit()

        embedder = FakeEmbedder()
        vector_store = InMemoryVectorStore()
        llm = FakeLLMGenerator()
        fig_processor = FigureProcessor(storage=storage, llm_generator=llm)

        pipeline = IngestionPipeline(
            db=db,
            storage=storage,
            embedder=embedder,
            vector_store=vector_store,
            figure_processor=fig_processor,
        )

        final_status = await pipeline.process_source("src-mm-1")
        assert final_status == SourceStatus.READY

        # Verify element records in DB
        res = await db.execute(select_elements("src-mm-1"))
        elements = list(res.scalars().all())
        assert len(elements) > 0

        # Check figure element provenance & crop saving
        fig_elems = [e for e in elements if e.type == ElementType.FIGURE]
        if fig_elems:
            fig = fig_elems[0]
            assert fig.caption_status == "completed"
            assert fig.image_path is not None
            assert await storage.exists(fig.image_path)


def select_elements(source_id: str):
    from sqlalchemy import select
    return select(Element).where(Element.source_id == source_id)
