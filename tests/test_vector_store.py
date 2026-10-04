import pytest

from app.adapters.vector_store import InMemoryVectorStore
from app.domain.schemas import TenantScope


@pytest.mark.asyncio
async def test_vector_store_tenant_isolation():
    vs = InMemoryVectorStore()
    tenant_a = TenantScope(user_id="user_a", notebook_id="nb_a")
    tenant_b = TenantScope(user_id="user_b", notebook_id="nb_b")

    # Upsert chunk for tenant A
    chunks_a = [{
        "id": "c_a",
        "source_id": "s_a",
        "element_id": "e_a",
        "page_number": 1,
        "notebook_id": tenant_a.notebook_id,
        "user_id": tenant_a.user_id,
    }]
    vec_a = [[1.0, 0.0, 0.0]]
    await vs.upsert_chunks(chunks_a, vec_a)

    # Search with tenant A scope -> finds chunk
    res_a = await vs.search(query_vector=[1.0, 0.0, 0.0], tenant=tenant_a)
    assert len(res_a) == 1
    assert res_a[0].chunk_id == "c_a"

    # Search with tenant B scope -> finds zero chunks (Tenant Isolation!)
    res_b = await vs.search(query_vector=[1.0, 0.0, 0.0], tenant=tenant_b)
    assert len(res_b) == 0
