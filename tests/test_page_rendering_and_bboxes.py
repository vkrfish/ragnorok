import fitz
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Element, ElementType, Source


@pytest.fixture
def create_sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample_page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter size
    page.insert_text((50, 50), "Sample Page Content for Testing")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.mark.asyncio
async def test_page_metadata_and_image_rendering(client: AsyncClient, db_session: AsyncSession, create_sample_pdf):
    # Create notebook
    res = await client.post(
        "/v1/notebooks",
        json={"title": "Page Test Notebook"},
        headers={"X-User-ID": "user_page"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    source = Source(
        notebook_id=nb_id,
        user_id="user_page",
        filename="sample_page.pdf",
        content_hash="hash_page_1",
        file_path=create_sample_pdf,
        mime_type="application/pdf",
        file_size=1024,
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
        content="Sample Page Content for Testing",
        page_number=1,
        bbox={"x0": 50.0, "y0": 50.0, "x1": 250.0, "y1": 70.0}
    )
    db_session.add(element)
    await db_session.commit()
    await db_session.refresh(element)

    # 1. GET page JSON metadata
    meta_res = await client.get(
        f"/v1/sources/{sid}/pages/1",
        headers={"X-User-ID": "user_page"}
    )
    assert meta_res.status_code == 200
    meta = meta_res.json()
    assert meta["page_number"] == 1
    assert meta["width_points"] == 612.0
    assert meta["height_points"] == 792.0
    assert meta["rotation"] == 0
    assert len(meta["elements"]) == 1
    assert meta["elements"][0]["id"] == element.id
    assert meta["elements"][0]["bbox"] == {"x0": 50.0, "y0": 50.0, "x1": 250.0, "y1": 70.0}

    # 2. GET page PNG image
    img_res = await client.get(
        f"/v1/sources/{sid}/pages/1/image?dpi=150",
        headers={"X-User-ID": "user_page"}
    )
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] == "image/png"
    assert "ETag" in img_res.headers
    assert "Cache-Control" in img_res.headers
    assert img_res.headers["Cache-Control"] == "private, max-age=86400"
    etag = img_res.headers["ETag"]

    # 3. Test ETag 304 Not Modified
    etag_res = await client.get(
        f"/v1/sources/{sid}/pages/1/image?dpi=150",
        headers={"X-User-ID": "user_page", "If-None-Match": etag}
    )
    assert etag_res.status_code == 304

    # 4. Invalid page number -> 404
    inv_page = await client.get(
        f"/v1/sources/{sid}/pages/99",
        headers={"X-User-ID": "user_page"}
    )
    assert inv_page.status_code == 404
