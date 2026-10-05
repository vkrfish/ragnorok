from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from app.adapters.sparse_embedder import SparseVector
from app.domain.schemas import TenantScope


class VectorSearchResult(BaseModel):
    chunk_id: str
    source_id: str
    element_id: str | None = None
    page_number: int
    score: float


class VectorStore(ABC):
    """Abstract VectorStore interface enforcing mandatory TenantScope filtering."""

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        """Upsert dense and optional sparse chunk embeddings with tenant filter fields."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        tenant: TenantScope,
        top_k: int = 5,
        min_score: float = 0.0,
        source_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Dense vector search filtered strictly by TenantScope and optional active source_ids."""

    @abstractmethod
    async def search_sparse(
        self,
        query_sparse: SparseVector,
        tenant: TenantScope,
        top_k: int = 5,
        source_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Sparse BM25 vector search filtered strictly by TenantScope and optional active source_ids."""


class InMemoryVectorStore(VectorStore):
    """In-memory VectorStore supporting both dense and sparse vector search with TenantScope."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = chunk["id"]
            sparse_vec = sparse_vectors[idx] if sparse_vectors and idx < len(sparse_vectors) else None

            # Remove pre-existing point with same chunk_id for deterministic re-embedding
            self._records = [r for r in self._records if r["chunk_id"] != chunk_id]
            self._records.append({
                "chunk_id": chunk_id,
                "source_id": chunk["source_id"],
                "element_id": chunk.get("element_id"),
                "page_number": chunk["page_number"],
                "notebook_id": chunk["notebook_id"],
                "user_id": chunk["user_id"],
                "vector": vector,
                "sparse_vector": sparse_vec,
            })

    async def search(
        self,
        query_vector: list[float],
        tenant: TenantScope,
        top_k: int = 5,
        min_score: float = 0.0,
        source_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        filtered = [
            r for r in self._records
            if r["notebook_id"] == tenant.notebook_id
            and (source_ids is None or r["source_id"] in source_ids)
        ]
        if not filtered:
            return []

        results: list[VectorSearchResult] = []
        for r in filtered:
            vec = r["vector"]
            dot = sum(a * b for a, b in zip(query_vector, vec))
            norm_a = sum(a * a for a in query_vector) ** 0.5
            norm_b = sum(b * b for b in vec) ** 0.5
            sim = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

            if sim >= min_score:
                results.append(
                    VectorSearchResult(
                        chunk_id=r["chunk_id"],
                        source_id=r["source_id"],
                        element_id=r.get("element_id"),
                        page_number=r["page_number"],
                        score=float(sim),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def search_sparse(
        self,
        query_sparse: SparseVector,
        tenant: TenantScope,
        top_k: int = 5,
        source_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        filtered = [
            r for r in self._records
            if r["notebook_id"] == tenant.notebook_id
            and (source_ids is None or r["source_id"] in source_ids)
        ]
        if not filtered or not query_sparse.indices:
            return []

        q_dict = dict(zip(query_sparse.indices, query_sparse.values))
        results: list[VectorSearchResult] = []

        for r in filtered:
            s_vec: SparseVector | None = r.get("sparse_vector")
            if not s_vec or not s_vec.indices:
                continue

            doc_dict = dict(zip(s_vec.indices, s_vec.values))
            score = sum(q_dict[idx] * doc_dict[idx] for idx in q_dict if idx in doc_dict)

            if score > 0:
                results.append(
                    VectorSearchResult(
                        chunk_id=r["chunk_id"],
                        source_id=r["source_id"],
                        element_id=r.get("element_id"),
                        page_number=r["page_number"],
                        score=float(score),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
