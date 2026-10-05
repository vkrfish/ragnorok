import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import ChatMessage, Element, ElementType, Source


@pytest.mark.asyncio
async def test_cross_tenant_isolation_404(client: AsyncClient, db_session: AsyncSession):
    # Owner User A creates notebook & source
    res_a = await client.post(
        "/v1/notebooks",
        json={"title": "Owner A Notebook"},
        headers={"X-User-ID": "user_owner_a"}
    )
    assert res_a.status_code == 201
    nb_id_a = res_a.json()["id"]

    source_a = Source(
        notebook_id=nb_id_a,
        user_id="user_owner_a",
        filename="owner_a_doc.pdf",
        content_hash="hash_owner_a",
        file_path="/tmp/owner_a.pdf",
        mime_type="application/pdf",
        file_size=100,
        status="READY",
        is_active=True
    )
    db_session.add(source_a)
    await db_session.commit()
    await db_session.refresh(source_a)
    sid_a = source_a.id

    element_a = Element(
        source_id=sid_a,
        type=ElementType.TEXT,
        content="Owner A Secret Data",
        page_number=1,
    )
    db_session.add(element_a)

    msg_a = ChatMessage(
        user_id="user_owner_a",
        notebook_id=nb_id_a,
        conversation_id="conv_a",
        role="user",
        content="Owner A query",
        status="complete"
    )
    db_session.add(msg_a)
    await db_session.commit()
    await db_session.refresh(element_a)
    eid_a = element_a.id

    # User B attempts cross-tenant access -> MUST return 404 (Not 403) to prevent resource existence leaks

    # 1. Source toggle
    res = await client.patch(
        f"/v1/notebooks/{nb_id_a}/sources/{sid_a}",
        json={"is_active": False},
        headers={"X-User-ID": "user_intruder_b"}
    )
    assert res.status_code == 404

    # 2. Element details
    res = await client.get(
        f"/v1/sources/{sid_a}/elements/{eid_a}",
        headers={"X-User-ID": "user_intruder_b"}
    )
    assert res.status_code == 404

    # 3. Page metadata
    res = await client.get(
        f"/v1/sources/{sid_a}/pages/1",
        headers={"X-User-ID": "user_intruder_b"}
    )
    assert res.status_code == 404

    # 4. Page image
    res = await client.get(
        f"/v1/sources/{sid_a}/pages/1/image",
        headers={"X-User-ID": "user_intruder_b"}
    )
    assert res.status_code == 404

    # 5. Chat conversations list
    res = await client.get(
        f"/v1/notebooks/{nb_id_a}/conversations",
        headers={"X-User-ID": "user_intruder_b"}
    )
    assert res.status_code == 404

    # 6. Chat message history
    res = await client.get(
        f"/v1/notebooks/{nb_id_a}/conversations/conv_a",
        headers={"X-User-ID": "user_intruder_b"}
    )
    assert res.status_code == 404
