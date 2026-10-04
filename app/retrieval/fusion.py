from typing import Any

from app.adapters.vector_store import VectorSearchResult
from app.core.config import settings


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) for combining dense vector and sparse BM25 search results."""

    def __init__(self, k: int = 60):
        self.k = k or settings.RRF_K

    def fuse(
        self,
        dense_results: list[VectorSearchResult],
        sparse_results: list[VectorSearchResult],
        top_k: int = 35,
    ) -> list[dict[str, Any]]:
        """Fuse dense and sparse search rankings using reciprocal rank scoring."""
        rrf_scores: dict[str, float] = {}
        item_map: dict[str, VectorSearchResult] = {}

        # Process dense rankings
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk_id
            item_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))

        # Process sparse rankings
        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk_id
            if cid not in item_map:
                item_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))

        fused: list[dict[str, Any]] = []
        for cid, score in rrf_scores.items():
            res = item_map[cid]
            fused.append({
                "chunk_id": res.chunk_id,
                "source_id": res.source_id,
                "element_id": res.element_id,
                "page_number": res.page_number,
                "rrf_score": score,
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]
