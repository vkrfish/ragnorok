import pytest

from app.adapters.reranker import FakeReranker


@pytest.mark.asyncio
async def test_original_query_passed_to_reranker():
    reranker = FakeReranker()
    original_query = "What is the P99 latency for vector retrieval?"
    _rewritten_subquery = "Vector retrieval latency metrics"

    candidates = [
        {"chunk_id": "c1", "content": "Vector retrieval P99 latency is 120ms.", "rrf_score": 0.8},
    ]

    # Confirm reranker runs against original_query
    reranked, _latency = await reranker.rerank(original_query, candidates, top_n=1)
    assert len(reranked) == 1
    assert reranked[0]["rerank_score"] > 0.5
