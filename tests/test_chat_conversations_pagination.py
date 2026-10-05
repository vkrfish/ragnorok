import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import ChatMessage, Element, ElementType, Source


@pytest.mark.asyncio
async def test_conversations_and_history_pagination(client: AsyncClient, db_session: AsyncSession):
    # Create notebook
    res = await client.post(
        "/v1/notebooks",
        json={"title": "Pagination Notebook"},
        headers={"X-User-ID": "user_pagi"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    # Create chat messages across 3 conversations
    conv_ids = ["conv-1", "conv-2", "conv-3"]
    for cid in conv_ids:
        msg1 = ChatMessage(
            user_id="user_pagi",
            notebook_id=nb_id,
            conversation_id=cid,
            role="user",
            content=f"Hello in {cid}",
            status="complete"
        )
        msg2 = ChatMessage(
            user_id="user_pagi",
            notebook_id=nb_id,
            conversation_id=cid,
            role="assistant",
            content=f"Response in {cid}",
            status="complete"
        )
        db_session.add(msg1)
        db_session.add(msg2)
    await db_session.commit()

    # 1. List conversations with page limit=2
    conv_res = await client.get(
        f"/v1/notebooks/{nb_id}/conversations?limit=2",
        headers={"X-User-ID": "user_pagi"}
    )
    assert conv_res.status_code == 200
    data = conv_res.json()
    assert len(data["items"]) == 2
    assert data["has_more"] is True
    assert data["next_cursor"] is not None

    # Fetch next page
    next_cursor = data["next_cursor"]
    next_res = await client.get(
        f"/v1/notebooks/{nb_id}/conversations?limit=2&cursor={next_cursor}",
        headers={"X-User-ID": "user_pagi"}
    )
    assert next_res.status_code == 200
    next_data = next_res.json()
    assert len(next_data["items"]) == 1
    assert next_data["has_more"] is False

    # 2. Get specific conversation history
    hist_res = await client.get(
        f"/v1/notebooks/{nb_id}/conversations/conv-1",
        headers={"X-User-ID": "user_pagi"}
    )
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert len(hist_data["items"]) == 2
    assert hist_data["items"][0]["role"] == "user"
    assert hist_data["items"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_element_details(client: AsyncClient, db_session: AsyncSession):
    res = await client.post(
        "/v1/notebooks",
        json={"title": "Element Notebook"},
        headers={"X-User-ID": "user_elem"}
    )
    assert res.status_code == 201
    nb_id = res.json()["id"]

    source = Source(
        notebook_id=nb_id,
        user_id="user_elem",
        filename="elem_doc.pdf",
        content_hash="hash_elem_1",
        file_path="/tmp/elem.pdf",
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
        type=ElementType.TABLE,
        content="Table cell content",
        page_number=2,
        bbox={"x0": 10.0, "y0": 20.0, "x1": 100.0, "y1": 200.0}
    )
    db_session.add(element)
    await db_session.commit()
    await db_session.refresh(element)

    elem_res = await client.get(
        f"/v1/sources/{source.id}/elements/{element.id}",
        headers={"X-User-ID": "user_elem"}
    )
    assert elem_res.status_code == 200
    edata = elem_res.json()
    assert edata["id"] == element.id
    assert edata["type"] == "table"
    assert edata["content"] == "Table cell content"
    assert edata["page_number"] == 2
    assert edata["bbox"] == {"x0": 10.0, "y0": 20.0, "x1": 100.0, "y1": 200.0}
