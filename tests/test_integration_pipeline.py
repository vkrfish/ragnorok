import io
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.adapters.factory import get_storage_adapter
from app.adapters.local_storage import LocalStorageAdapter
from app.ingestion.pipeline import IngestionPipeline
from app.workers.tasks import shared_embedder, shared_vector_store


@pytest.mark.asyncio
async def test_full_pdf_to_cited_answer_integration(client: AsyncClient, db_session: AsyncSession):
    headers = {"X-User-ID": "test_user_integration"}

    # 1. Create Notebook
    create_nb_res = await client.post(
        "/api/v1/notebooks/",
        json={"title": "Integration Test Notebook"},
        headers=headers,
    )
    assert create_nb_res.status_code == 201
    notebook_id = create_nb_res.json()["id"]

    # 2. Upload PDF
    pdf_content = b"Ragnork Integration Document.\nPostgres is the system of record for all chunks."
    files = {"file": ("test_doc.pdf", io.BytesIO(pdf_content), "application/pdf")}

    upload_res = await client.post(
        f"/api/v1/notebooks/{notebook_id}/sources",
        files=files,
        headers=headers,
    )
    assert upload_res.status_code == 201
    source = upload_res.json()
    source_id = source["id"]

    # Duplicate Upload Test (Idempotency)
    dupe_res = await client.post(
        f"/api/v1/notebooks/{notebook_id}/sources",
        files={"file": ("test_doc_dupe.pdf", io.BytesIO(pdf_content), "application/pdf")},
        headers=headers,
    )
    assert dupe_res.status_code == 201
    assert dupe_res.json()["id"] == source_id

    # 3. Process Ingestion Pipeline
    storage = app.dependency_overrides[get_storage_adapter]()
    pipeline = IngestionPipeline(
        db=db_session,
        storage=storage,
        embedder=shared_embedder,
        vector_store=shared_vector_store,
    )
    final_status = await pipeline.process_source(source_id)
    assert final_status == "READY" or final_status.value == "READY"

    # 4. Chat Q&A Request
    chat_res = await client.post(
        f"/api/v1/notebooks/{notebook_id}/chat",
        json={"query": "What is the system of record?", "relevance_threshold": 0.0},
        headers=headers,
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()

    assert chat_data["is_refusal"] is False
    assert len(chat_data["citations"]) > 0
    citation = chat_data["citations"][0]
    assert citation["source_id"] == source_id
    assert citation["page"] == 1
    assert "snippet" in citation

    # 5. Refusal Path Test (High relevance threshold with no matching chunks)
    refusal_res = await client.post(
        f"/api/v1/notebooks/{notebook_id}/chat",
        json={"query": "Unrelated topic", "relevance_threshold": 0.999},
        headers=headers,
    )
    assert refusal_res.status_code == 200
    assert refusal_res.json()["is_refusal"] is True
