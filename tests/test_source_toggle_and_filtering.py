import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Chunk, Element, ElementType, Source
from app.retrieval.search import RetrievalService
from app.workers.tasks import shared_embedder, shared_vector_store


@pytest.mark.asyncio
async def test_source_toggle_idempotent(client: AsyncClient, db_session: AsyncSession):
    # Create notebook
    res = await client.post(
        "/v1/notebooks",
        json={"title": "Toggle Test Notebook"},
        headers={"X-User-ID": "user_toggle"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    # Create source in DB directly
    source = Source(
        notebook_id=nb_id,
        user_id="user_toggle",
        filename="toggle_doc.pdf",
        content_hash="hash_toggle_1",
        file_path="/tmp/toggle.pdf",
        mime_type="application/pdf",
        file_size=100,
        status="READY",
        is_active=True
    )
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    sid = source.id

    # Toggle to inactive
    res = await client.patch(
        f"/v1/notebooks/{nb_id}/sources/{sid}",
        json={"is_active": False},
        headers={"X-User-ID": "user_toggle"}
    )
    assert res.status_code == 200
    assert res.json()["is_active"] is False

    # Repeat toggle to inactive (idempotent)
    res = await client.patch(
        f"/v1/notebooks/{nb_id}/sources/{sid}",
        json={"is_active": False},
        headers={"X-User-ID": "user_toggle"}
    )
    assert res.status_code == 200
    assert res.json()["is_active"] is False

    # Toggle back to active
    res = await client.patch(
        f"/v1/notebooks/{nb_id}/sources/{sid}",
        json={"is_active": True},
        headers={"X-User-ID": "user_toggle"}
    )
    assert res.status_code == 200
    assert res.json()["is_active"] is True


@pytest.mark.asyncio
async def test_zero_active_sources_refusal(client: AsyncClient, db_session: AsyncSession):
    # Create notebook
    res = await client.post(
        "/v1/notebooks",
        json={"title": "Zero Active Notebook"},
        headers={"X-User-ID": "user_zero"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    # Chat with 0 sources -> should return refusal without calling LLM
    chat_res = await client.post(
        f"/v1/notebooks/{nb_id}/chat",
        json={"query": "What is in the document?"},
        headers={"X-User-ID": "user_zero"}
    )
    assert chat_res.status_code == 200
    body = chat_res.json()
    assert body["is_refusal"] is True
    assert body["refusal_reason"] == "NO_ACTIVE_SOURCES"
    assert body["citations"] == []


@pytest.mark.asyncio
async def test_toggle_affects_retrieval(client: AsyncClient, db_session: AsyncSession):
    res = await client.post(
        "/v1/notebooks",
        json={"title": "Retrieval Filter Notebook"},
        headers={"X-User-ID": "user_filter"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    source = Source(
        notebook_id=nb_id,
        user_id="user_filter",
        filename="filter_doc.pdf",
        content_hash="hash_filter_1",
        file_path="/tmp/filter.pdf",
        mime_type="application/pdf",
        file_size=100,
        status="READY",
        is_active=True
    )
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    sid = source.id

    element = Element(
        source_id=sid,
        type=ElementType.TEXT,
        content="Quantum computing uses qubits for superdense coding.",
        page_number=1,
    )
    db_session.add(element)
    await db_session.commit()
    await db_session.refresh(element)

    chunk = Chunk(
        id=element.id,
        source_id=sid,
        notebook_id=nb_id,
        user_id="user_filter",
        element_id=element.id,
        page_number=1,
        content=element.content,
        embeddable_text=element.content,
        token_count=10,
    )
    db_session.add(chunk)
    await db_session.commit()

    # Upsert element to vector store
    vec = await shared_embedder.embed_text("Quantum computing uses qubits for superdense coding.")
    await shared_vector_store.upsert_chunks(
        chunks=[{
            "id": element.id,
            "user_id": "user_filter",
            "notebook_id": nb_id,
            "source_id": sid,
            "content": element.content,
            "page_number": 1
        }],
        vectors=[vec]
    )

    from app.domain.schemas import TenantScope

    retrieval_service = RetrievalService(embedder=shared_embedder, vector_store=shared_vector_store)

    # 1. Active: context retrieved
    blocks_active, _, _, _ = await retrieval_service.retrieve_context(
        query="quantum computing",
        tenant=TenantScope(user_id="user_filter", notebook_id=nb_id),
        db=db_session,
        min_score=0.0
    )
    assert len(blocks_active) > 0

    # 2. Toggle inactive in DB
    source.is_active = False
    await db_session.commit()

    # Context excluded
    blocks_inactive, _, _, _ = await retrieval_service.retrieve_context(
        query="quantum computing",
        tenant=TenantScope(user_id="user_filter", notebook_id=nb_id),
        db=db_session,
        min_score=0.0
    )
    assert len(blocks_inactive) == 0
