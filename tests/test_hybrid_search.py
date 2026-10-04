import pytest

from app.adapters.sparse_embedder import SparseEmbedder
from app.adapters.vector_store import InMemoryVectorStore
from app.domain.schemas import TenantScope
from app.retrieval.fusion import ReciprocalRankFusion


@pytest.mark.asyncio
async def test_hybrid_search_exact_match_terms():
    sparse_embedder = SparseEmbedder()
    vector_store = InMemoryVectorStore()

    tenant = TenantScope(user_id="user-hybrid", notebook_id="nb-hybrid")

    chunks = [
        {"id": "c1", "source_id": "s1", "element_id": "e1", "page_number": 1, "notebook_id": tenant.notebook_id, "user_id": tenant.user_id},
        {"id": "c2", "source_id": "s1", "element_id": "e2", "page_number": 2, "notebook_id": tenant.notebook_id, "user_id": tenant.user_id},
    ]

    texts = [
        "Financial report revenue for Q3-2024 reached 50M for product SKU-4471.",
        "General corporate overview and employee guidelines.",
    ]

    dense_vecs = [[0.1] * 384, [0.2] * 384]
    sparse_vecs = sparse_embedder.embed_batch(texts)

    await vector_store.upsert_chunks(chunks, dense_vecs, sparse_vecs)

    # Search exact term SKU-4471
    query_sparse = sparse_embedder.embed_text("SKU-4471 Q3-2024")
    sparse_res = await vector_store.search_sparse(query_sparse, tenant=tenant, top_k=5)

    assert len(sparse_res) >= 1
    assert sparse_res[0].chunk_id == "c1"

    # Test RRF fusion
    fusion = ReciprocalRankFusion(k=60)
    dense_res = await vector_store.search(dense_vecs[0], tenant=tenant, top_k=5)
    fused = fusion.fuse(dense_res, sparse_res, top_k=5)

    assert len(fused) > 0
    assert fused[0]["chunk_id"] == "c1"
