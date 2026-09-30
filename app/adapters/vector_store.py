from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from app.domain.schemas import TenantScope


class VectorSearchResult(BaseModel):
    chunk_id: str
    source_id: str
    element_id: Optional[str] = None
    page_number: int
    score: float


class VectorStore(ABC):
    """Abstract VectorStore interface enforcing mandatory TenantScope filtering."""

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> None:
        """Upsert chunk embeddings with minimal IDs and filter fields in payload."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        tenant: TenantScope,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Dense vector search filtered strictly by TenantScope."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory VectorStore for fast unit & integration tests enforcing TenantScope."""

    def __init__(self):
        # Storage schema: list of dicts {chunk_id, source_id, element_id, page_number, notebook_id, user_id, vector}
        self._records: List[Dict[str, Any]] = []

    async def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._records.append({
                "chunk_id": chunk["id"],
                "source_id": chunk["source_id"],
                "element_id": chunk.get("element_id"),
                "page_number": chunk["page_number"],
                "notebook_id": chunk["notebook_id"],
                "user_id": chunk["user_id"],
                "vector": vector,
            })

    async def search(
        self,
        query_vector: List[float],
        tenant: TenantScope,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[VectorSearchResult]:
        # Filter strictly by tenant scope (notebook_id AND user_id)
        filtered = [
            r for r in self._records
            if r["notebook_id"] == tenant.notebook_id and r["user_id"] == tenant.user_id
        ]

        if not filtered:
            return []

        # Cosine similarity calculation
        results: List[VectorSearchResult] = []
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
