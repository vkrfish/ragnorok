from typing import Dict, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.adapters.embedder import Embedder
from app.adapters.vector_store import VectorStore
from app.domain.models import Chunk
from app.domain.schemas import TenantScope


class RetrievalService:
    """Retrieval service wrapping vector store search with tenant scope and Postgres context expansion."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def _cosine_sim(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    async def retrieve_context(
        self,
        query: str,
        tenant: TenantScope,
        db: AsyncSession,
        top_k: int = 5,
        min_score: float = 0.4,
    ) -> Tuple[List[dict], Dict[str, dict]]:
        """
        Embeds query, performs vector search with tenant scope,
        expands context from Postgres DB (System of Record), and builds formatted context blocks.
        """
        # 1. Embed query
        query_vec = await self.embedder.embed_text(query)

        # 2. Vector search strictly filtered by TenantScope
        search_results = await self.vector_store.search(
            query_vector=query_vec,
            tenant=tenant,
            top_k=top_k,
            min_score=min_score,
        )

        chunk_score_map: Dict[str, float] = {}
        chunks_to_use: List[Chunk] = []

        if search_results:
            chunk_ids = [sr.chunk_id for sr in search_results]
            chunk_score_map = {sr.chunk_id: sr.score for sr in search_results}

            db_res = await db.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
            chunks_by_id = {c.id: c for c in db_res.scalars().all()}
            chunks_to_use = [chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id]
        else:
            # Fallback to DB (System of Record) if in-memory index was cleared
            db_res = await db.execute(
                select(Chunk).where(
                    Chunk.notebook_id == tenant.notebook_id,
                    Chunk.user_id == tenant.user_id,
                )
            )
            all_tenant_chunks = list(db_res.scalars().all())

            if all_tenant_chunks:
                # Compute similarity for fallback chunks
                chunk_vectors = await self.embedder.embed_batch(
                    [c.embeddable_text for c in all_tenant_chunks]
                )
                
                # Re-upsert into vector store for future searches
                chunk_dicts = [
                    {
                        "id": c.id,
                        "source_id": c.source_id,
                        "element_id": c.element_id,
                        "page_number": c.page_number,
                        "notebook_id": c.notebook_id,
                        "user_id": c.user_id,
                    }
                    for c in all_tenant_chunks
                ]
                await self.vector_store.upsert_chunks(chunk_dicts, chunk_vectors)

                # Filter by min_score
                scored_chunks = []
                for c, v in zip(all_tenant_chunks, chunk_vectors):
                    sim = self._cosine_sim(query_vec, v)
                    if sim >= min_score:
                        scored_chunks.append((c, sim))

                scored_chunks.sort(key=lambda x: x[1], reverse=True)
                chunks_to_use = [item[0] for item in scored_chunks[:top_k]]
                chunk_score_map = {item[0].id: item[1] for item in scored_chunks[:top_k]}

        if not chunks_to_use:
            return [], {}

        retrieved_blocks: List[dict] = []
        context_map: Dict[str, dict] = {}

        for idx, chunk in enumerate(chunks_to_use[:top_k], start=1):
            handle = f"[{idx}]"
            score = chunk_score_map.get(chunk.id, 1.0)
            block_info = {
                "handle": handle,
                "chunk_id": chunk.id,
                "source_id": chunk.source_id,
                "element_id": chunk.element_id,
                "page": chunk.page_number,
                "bbox": chunk.bbox,
                "content": chunk.content,
                "heading_path": chunk.heading_path,
                "score": score,
            }

            retrieved_blocks.append(block_info)
            context_map[handle] = block_info

        return retrieved_blocks, context_map
