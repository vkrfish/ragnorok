import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.domain.models import (
    Chunk,
    Notebook,
    Source,
    SourceStatus,
)
from app.retrieval.context_expansion import ParentContextExpander


@pytest.mark.asyncio
async def test_parent_context_expansion_and_document_ordering():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        nb = Notebook(id="nb-ctx", user_id="user-ctx", title="Parent Context Test")
        src = Source(id="src-ctx", notebook_id="nb-ctx", user_id="user-ctx", filename="doc.pdf", content_hash="h1", file_path="media/doc.pdf", file_size=100, status=SourceStatus.READY)
        db.add_all([nb, src])
        await db.commit()

        # Page 2 chunk (retrieved first in score)
        c2 = Chunk(id="c2", source_id="src-ctx", notebook_id="nb-ctx", user_id="user-ctx", page_number=2, content="Page 2 text snippet", embeddable_text="Page 2 text snippet", token_count=10, bbox={"y0": 100.0})
        # Page 1 chunk (retrieved second in score)
        c1 = Chunk(id="c1", source_id="src-ctx", notebook_id="nb-ctx", user_id="user-ctx", page_number=1, content="Page 1 text snippet", embeddable_text="Page 1 text snippet", token_count=10, bbox={"y0": 50.0})
        db.add_all([c2, c1])
        await db.commit()

        expander = ParentContextExpander()
        blocks, _ctx_map = await expander.expand_context(
            retrieved_chunk_ids=["c2", "c1"],
            notebook_id="nb-ctx",
            user_id="user-ctx",
            db=db,
            max_tokens=2000,
        )

        assert len(blocks) == 2
        # Final context ordered by document position (Page 1 before Page 2)!
        assert blocks[0]["page"] == 1
        assert blocks[1]["page"] == 2
        assert blocks[0]["handle"] == "[1]"
        assert blocks[1]["handle"] == "[2]"
