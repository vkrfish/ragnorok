import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.embedder import Embedder
from app.adapters.reranker import Reranker, get_reranker
from app.adapters.sparse_embedder import SparseEmbedder
from app.adapters.vector_store import VectorSearchResult, VectorStore
from app.core.config import settings
from app.core.logging import logger
from app.domain.schemas import TenantScope
from app.retrieval.context_expansion import ParentContextExpander
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.rewriter import QueryRewriter, QueryRewriteResult


class RetrievalService:
    """Full Phase 4 Hybrid Retrieval Service: Query Rewriting -> Qdrant Dense + Sparse -> RRF Fusion -> Cross-Encoder Reranking -> Refusal Check -> Parent Context Expansion."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        expander: ParentContextExpander | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_embedder = sparse_embedder or SparseEmbedder()
        self.reranker = reranker or get_reranker()
        self.rewriter = rewriter or QueryRewriter()
        self.expander = expander or ParentContextExpander()
        self.fusion = ReciprocalRankFusion(k=settings.RRF_K)

    async def retrieve_context(
        self,
        query: str,
        tenant: TenantScope,
        db: AsyncSession,
        chat_history: list[dict[str, str]] | None = None,
        notebook_title: str = "Research Notebook",
        top_k: int = 5,
        min_score: float = 0.4,
        override_source_ids: list[str] | None = None,
    ) -> tuple[list[dict], dict[str, dict], dict[str, float], bool]:
        """
        Full Hybrid Retrieval Pipeline.
        Returns: (retrieved_blocks, context_map, latencies_ms, is_refusal)
        """
        latencies: dict[str, float] = {}

        # 0. Snapshot active READY source_ids from Postgres (Source of Truth)
        from sqlalchemy import select
        from app.domain.models import Source, SourceStatus

        stmt = select(Source.id).where(
            Source.notebook_id == tenant.notebook_id,
            Source.is_active.is_(True),
            Source.status == SourceStatus.READY,
        )
        res = await db.execute(stmt)
        db_active_ids = list(res.scalars().all())

        clean_overrides = (
            [sid for sid in override_source_ids if sid and sid.strip().lower() not in ("string", "null", "none")]
            if override_source_ids is not None
            else []
        )

        if clean_overrides:
            active_source_ids = [sid for sid in db_active_ids if sid in clean_overrides]
        else:
            active_source_ids = db_active_ids

        if not active_source_ids:
            logger.info(f"Zero active/ready sources found for notebook {tenant.notebook_id}. Refusing without LLM call.")
            return [], {}, {"rewrite_ms": 0.0, "dense_retrieval_ms": 0.0, "sparse_retrieval_ms": 0.0, "rrf_fusion_ms": 0.0, "rerank_ms": 0.0}, True

        # 1. Query Understanding & Rewriting (Preserves original query)
        t0 = time.time()
        rewrite_res: QueryRewriteResult = await self.rewriter.rewrite(
            user_query=query,
            chat_history=chat_history,
            notebook_title=notebook_title,
        )
        latencies["rewrite_ms"] = (time.time() - t0) * 1000

        search_queries = rewrite_res.sub_queries or [query]

        # 2. Dense Vector Search across all query variants with active_source_ids filter
        t1 = time.time()
        all_dense_results: list[VectorSearchResult] = []
        for q_var in search_queries:
            q_vec = await self.embedder.embed_text(q_var)
            d_res = await self.vector_store.search(
                query_vector=q_vec,
                tenant=tenant,
                top_k=settings.MAX_RERANK_CANDIDATES,
                source_ids=active_source_ids,
            )
            all_dense_results.extend(d_res)
        latencies["dense_retrieval_ms"] = (time.time() - t1) * 1000

        # 3. Qdrant Sparse BM25 Search across all query variants with active_source_ids filter
        t2 = time.time()
        all_sparse_results: list[VectorSearchResult] = []
        for q_var in search_queries:
            s_vec = self.sparse_embedder.embed_text(q_var)
            s_res = await self.vector_store.search_sparse(
                query_sparse=s_vec,
                tenant=tenant,
                top_k=settings.MAX_RERANK_CANDIDATES,
                source_ids=active_source_ids,
            )
            all_sparse_results.extend(s_res)
        latencies["sparse_retrieval_ms"] = (time.time() - t2) * 1000

        # 4. Reciprocal Rank Fusion (RRF)
        t3 = time.time()
        fused_candidates = self.fusion.fuse(
            dense_results=all_dense_results,
            sparse_results=all_sparse_results,
            top_k=settings.MAX_RERANK_CANDIDATES,
        )
        latencies["rrf_fusion_ms"] = (time.time() - t3) * 1000

        if not fused_candidates:
            # Fallback to DB (System of Record) if index was reset
            return await self._fallback_db_retrieval(query, tenant, db, top_k, latencies, active_source_ids)

        # Expand candidate content for reranking
        candidate_ids = [c["chunk_id"] for c in fused_candidates]
        from sqlalchemy import select

        from app.domain.models import Chunk
        db_res = await db.execute(select(Chunk).where(Chunk.id.in_(candidate_ids)))
        chunks_by_id = {c.id: c for c in db_res.scalars().all()}

        candidates_for_rerank = []
        for fc in fused_candidates:
            cid = fc["chunk_id"]
            if cid in chunks_by_id:
                chunk_orm = chunks_by_id[cid]
                candidates_for_rerank.append({
                    "chunk_id": cid,
                    "source_id": chunk_orm.source_id,
                    "element_id": chunk_orm.element_id,
                    "page_number": chunk_orm.page_number,
                    "content": chunk_orm.content,
                    "rrf_score": fc["rrf_score"],
                })

        # 5. Cross-Encoder Reranking against the ORIGINAL user query
        top_reranked, rerank_ms = await self.reranker.rerank(
            query=query,  # Always use ORIGINAL user query for reranking!
            candidates=candidates_for_rerank,
            top_n=top_k,
        )
        latencies["rerank_ms"] = rerank_ms

        if not top_reranked:
            return [], {}, latencies, True

        # 6. Reranker-Driven Refusal Threshold Check
        top_score = top_reranked[0].get("rerank_score", 0.0)
        effective_threshold = min_score if min_score is not None else settings.RERANK_REFUSAL_THRESHOLD
        is_refusal = top_score < effective_threshold
        if is_refusal:
            logger.info(f"Reranker top score {top_score:.4f} < threshold {effective_threshold}. Triggering refusal.")
            return [], {}, latencies, True

        # 7. Parent Context Expansion & Document Order Assembly
        t4 = time.time()
        reranked_ids = [c["chunk_id"] for c in top_reranked]
        retrieved_blocks, context_map = await self.expander.expand_context(
            retrieved_chunk_ids=reranked_ids,
            notebook_id=tenant.notebook_id,
            user_id=tenant.user_id,
            db=db,
            max_tokens=settings.MAX_PARENT_CONTEXT_TOKENS,
            active_source_ids=active_source_ids,
        )
        latencies["parent_expansion_ms"] = (time.time() - t4) * 1000

        return retrieved_blocks, context_map, latencies, False

    async def _fallback_db_retrieval(
        self,
        query: str,
        tenant: TenantScope,
        db: AsyncSession,
        top_k: int,
        latencies: dict[str, float],
        active_source_ids: list[str] | None = None,
    ) -> tuple[list[dict], dict[str, dict], dict[str, float], bool]:
        """System of Record Postgres DB fallback if in-memory index was reset."""
        from sqlalchemy import select

        from app.domain.models import Chunk
        stmt = select(Chunk).where(
            Chunk.notebook_id == tenant.notebook_id,
        )
        if active_source_ids is not None:
            stmt = stmt.where(Chunk.source_id.in_(active_source_ids))

        db_res = await db.execute(stmt)
        all_chunks = list(db_res.scalars().all())
        if not all_chunks:
            return [], {}, latencies, True

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
            for c in all_chunks
        ]
        vectors = await self.embedder.embed_batch([c.embeddable_text for c in all_chunks])
        sparse_vecs = self.sparse_embedder.embed_batch([c.embeddable_text for c in all_chunks])
        await self.vector_store.upsert_chunks(chunk_dicts, vectors, sparse_vecs)

        candidate_ids = [c.id for c in all_chunks[:10]]
        blocks, ctx_map = await self.expander.expand_context(
            candidate_ids, tenant.notebook_id, tenant.user_id, db, active_source_ids=active_source_ids
        )
        return blocks, ctx_map, latencies, False
