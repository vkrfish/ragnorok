import pytest

from app.adapters.reranker import FakeReranker
from app.core.config import settings


@pytest.mark.asyncio
async def test_reranker_driven_refusal_threshold():
    reranker = FakeReranker()
    query = "What is the stock price of Apple in 2026? refusal test"

    candidates = [
        {"chunk_id": "c1", "content": "Postgres is the system of record.", "rrf_score": 0.05},
        {"chunk_id": "c2", "content": "Qdrant is used for dense vector indexing.", "rrf_score": 0.04},
    ]

    reranked, _latency = await reranker.rerank(query, candidates, top_n=2)
    assert len(reranked) > 0
    top_score = reranked[0]["rerank_score"]

    # Refusal condition check
    is_refusal = top_score < settings.RERANK_REFUSAL_THRESHOLD
    assert is_refusal is True, f"Expected top score {top_score} to be below refusal threshold {settings.RERANK_REFUSAL_THRESHOLD}"
