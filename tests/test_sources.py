import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_notebook_and_upload_source(client: AsyncClient):
    headers = {"X-User-ID": "user-test-123"}

    # 1. Create Notebook
    create_nb_res = await client.post(
        "/api/v1/notebooks/",
        json={"title": "Test Research Notebook"},
        headers=headers,
    )
    assert create_nb_res.status_code == 201
    notebook = create_nb_res.json()
    notebook_id = notebook["id"]
    assert notebook["user_id"] == "user-test-123"

    # 2. Upload Source PDF
    pdf_content = b"%PDF-1.4 Mock document text and tables"
    files = {"file": ("research_paper.pdf", io.BytesIO(pdf_content), "application/pdf")}

    upload_res = await client.post(
        f"/api/v1/notebooks/{notebook_id}/sources",
        files=files,
        headers=headers,
    )
    assert upload_res.status_code == 201
    source = upload_res.json()
    assert source["notebook_id"] == notebook_id
    assert source["filename"] == "research_paper.pdf"
    assert source["status"] in ("PENDING", "READY")
    assert source["user_id"] == "user-test-123"

    # 3. Idempotent upload test (upload exact same file again)
    files_dupe = {"file": ("research_paper_copy.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_dupe_res = await client.post(
        f"/api/v1/notebooks/{notebook_id}/sources",
        files=files_dupe,
        headers=headers,
    )
    assert upload_dupe_res.status_code == 201
    dupe_source = upload_dupe_res.json()
    # Must return the SAME source object (idempotency key: content_hash)
    assert dupe_source["id"] == source["id"]

    # 4. List Sources
    list_res = await client.get(
        f"/api/v1/notebooks/{notebook_id}/sources",
        headers=headers,
    )
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] == 1
    assert list_data["sources"][0]["id"] == source["id"]

    # 5. Get Source details
    get_res = await client.get(
        f"/api/v1/notebooks/{notebook_id}/sources/{source['id']}",
        headers=headers,
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == source["id"]
