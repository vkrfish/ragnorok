from typing import Any

from app.adapters.embedder import FakeEmbedder
from app.adapters.factory import get_storage_adapter
from app.adapters.vector_store import InMemoryVectorStore
from app.core.db import AsyncSessionLocal
from app.core.logging import logger
from app.ingestion.pipeline import IngestionPipeline

# Module-level singletons for async worker task
shared_vector_store = InMemoryVectorStore()
shared_embedder = FakeEmbedder()


async def process_source_task(ctx: dict[str, Any], source_id: str) -> str:
    """Async background task for running source ingestion pipeline."""
    logger.info(f"Worker picked up ingestion task for source_id={source_id}")
    async with AsyncSessionLocal() as session:
        storage = get_storage_adapter()
        pipeline = IngestionPipeline(
            db=session,
            storage=storage,
            embedder=shared_embedder,
            vector_store=shared_vector_store,
        )
        status = await pipeline.process_source(source_id)
        return status.value
