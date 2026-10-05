from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.domain.models import Chunk, Element, ElementType


class ParentContextExpander:
    """Parent Context Expander with token budget management and document-position context ordering."""

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def expand_context(
        self,
        retrieved_chunk_ids: list[str],
        notebook_id: str,
        user_id: str,
        db: AsyncSession,
        max_tokens: int = 2000,
        active_source_ids: list[str] | None = None,
    ) -> tuple[list[dict], dict[str, dict]]:
        """Expand retrieved chunks to parent context with token budget & document order."""
        if not retrieved_chunk_ids:
            return [], {}

        # 1. Fetch retrieved chunks from Postgres DB (System of Record)
        db_res = await db.execute(select(Chunk).where(Chunk.id.in_(retrieved_chunk_ids)))
        chunks_by_id = {c.id: c for c in db_res.scalars().all()}
        target_chunks = [
            chunks_by_id[cid] for cid in retrieved_chunk_ids
            if cid in chunks_by_id and (active_source_ids is None or chunks_by_id[cid].source_id in active_source_ids)
        ]

        if not target_chunks:
            return [], {}

        # Sort chunks by document position (page_number, bbox y0)
        target_chunks.sort(key=lambda c: (c.page_number, (c.bbox or {}).get("y0", 0.0)))

        # Fetch parent elements for table & figure context expansion
        element_ids = [c.element_id for c in target_chunks if c.element_id]
        elements_by_id: dict[str, Element] = {}
        if element_ids:
            elem_res = await db.execute(select(Element).where(Element.id.in_(element_ids)))
            elements_by_id = {e.id: e for e in elem_res.scalars().all()}

        expanded_blocks: list[dict] = []
        context_map: dict[str, dict] = {}
        total_tokens = 0

        for idx, chunk in enumerate(target_chunks, start=1):
            handle = f"[{idx}]"
            elem = elements_by_id.get(chunk.element_id) if chunk.element_id else None

            # Priority 1: Full raw Markdown table
            if elem and elem.type == ElementType.TABLE:
                expanded_content = f"[Table (Page {chunk.page_number})]\n{elem.content}"
            # Priority 2: Section-aware chunk text with heading path
            else:
                heading = f"[Section: {chunk.heading_path}]\n" if chunk.heading_path else ""
                expanded_content = f"{heading}{chunk.content}"

            tok_count = self.estimate_tokens(expanded_content)
            if total_tokens + tok_count > max_tokens and expanded_blocks:
                logger.info(f"Parent context expansion reached token budget limit ({total_tokens}/{max_tokens})")
                break

            total_tokens += tok_count
            block_info = {
                "handle": handle,
                "chunk_id": chunk.id,
                "source_id": chunk.source_id,
                "element_id": chunk.element_id,
                "page": chunk.page_number,
                "bbox": chunk.bbox,
                "content": expanded_content,
                "heading_path": chunk.heading_path,
                "score": getattr(chunk, "rerank_score", 1.0),
                "image_path": elem.image_path if elem and elem.type == ElementType.FIGURE else None,
            }

            expanded_blocks.append(block_info)
            context_map[handle] = block_info

        return expanded_blocks, context_map
