import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Chunk, Element, ElementType, Source
from app.workers.tasks import shared_embedder, shared_vector_store


@pytest.mark.asyncio
async def test_sse_streaming_event_order_and_parity(client: AsyncClient, db_session: AsyncSession):
    # Create notebook
    res = await client.post(
        "/v1/notebooks",
        json={"title": "SSE Parity Notebook"},
        headers={"X-User-ID": "user_sse"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    # Create active READY source & element
    source = Source(
        notebook_id=nb_id,
        user_id="user_sse",
        filename="sse_doc.pdf",
        content_hash="hash_sse_1",
        file_path="/tmp/sse.pdf",
        mime_type="application/pdf",
        file_size=100,
        status="READY",
        is_active=True
    )
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)

    element = Element(
        source_id=source.id,
        type=ElementType.TEXT,
        content="Photosynthesis converts solar energy into chemical energy.",
        page_number=1,
    )
    db_session.add(element)
    await db_session.commit()
    await db_session.refresh(element)

    chunk = Chunk(
        id=element.id,
        source_id=source.id,
        notebook_id=nb_id,
        user_id="user_sse",
        element_id=element.id,
        page_number=1,
        content=element.content,
        embeddable_text=element.content,
        token_count=10,
    )
    db_session.add(chunk)
    await db_session.commit()

    vec = await shared_embedder.embed_text(element.content)
    await shared_vector_store.upsert_chunks(
        chunks=[{
            "id": element.id,
            "user_id": "user_sse",
            "notebook_id": nb_id,
            "source_id": source.id,
            "content": element.content,
            "page_number": 1
        }],
        vectors=[vec]
    )

    # 1. Non-streamed response
    sync_res = await client.post(
        f"/v1/notebooks/{nb_id}/chat",
        json={"query": "Explain photosynthesis", "relevance_threshold": 0.0},
        headers={"X-User-ID": "user_sse"}
    )
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert not sync_data.get("is_refusal", False)

    # 2. Streamed response
    async with client.stream(
        "POST",
        f"/v1/notebooks/{nb_id}/chat/stream",
        json={"query": "Explain photosynthesis", "relevance_threshold": 0.0},
        headers={"X-User-ID": "user_sse"}
    ) as stream_res:
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]

        events = []
        async for line in stream_res.aiter_lines():
            if line.startswith("event: "):
                events.append(line.replace("event: ", "").strip())

        # Verify event order: retrieval -> token (optional multiples) -> citations -> done
        assert "retrieval" in events
        assert "token" in events
        assert "citations" in events
        assert "done" in events
        assert events[0] == "retrieval"
        assert events[-1] == "done"


@pytest.mark.asyncio
async def test_sse_zero_sources_refusal(client: AsyncClient, db_session: AsyncSession):
    res = await client.post(
        "/v1/notebooks",
        json={"title": "SSE Refusal Notebook"},
        headers={"X-User-ID": "user_sse_refusal"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    async with client.stream(
        "POST",
        f"/v1/notebooks/{nb_id}/chat/stream",
        json={"query": "Anything here?"},
        headers={"X-User-ID": "user_sse_refusal"}
    ) as stream_res:
        assert stream_res.status_code == 200

        events = []
        async for line in stream_res.aiter_lines():
            if line.startswith("event: "):
                events.append(line.replace("event: ", "").strip())

        assert "refusal" in events
        assert "done" in events
        assert events[0] == "refusal"
        assert events[-1] == "done"
