import time
import uuid

from fastapi import APIRouter, Depends, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.factory import get_storage_adapter
from app.adapters.llm import LLMGenerator, get_llm_generator
from app.core.config import settings
from app.core.db import get_db
from app.core.errors import ForbiddenError, NotFoundError
from app.domain.models import ChatMessage, Element, ElementType, Notebook
from app.domain.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    ImagePart,
    TenantScope,
)
from app.generation.citation import CitationResolver
from app.generation.prompts import load_prompt_template
from app.retrieval.search import RetrievalService
from app.workers.tasks import shared_embedder, shared_vector_store

router = APIRouter()


def get_user_id(x_user_id: str | None = Header(None, alias="X-User-ID")) -> str:
    return x_user_id or "default_user"


@router.post(
    "/{notebook_id}/chat",
    response_model=ChatQueryResponse,
    status_code=status.HTTP_200_OK,
)
async def chat_query(
    notebook_id: str,
    payload: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Phase 4 Grounded Q&A endpoint with tenant isolation, multi-turn history, query rewriting, reranking, and latency recording."""
    # 1. Verify notebook existence & tenant ownership
    nb_result = await db.execute(
        select(Notebook).where(
            Notebook.id == notebook_id,
            Notebook.user_id == user_id,
        )
    )
    notebook = nb_result.scalar_one_or_none()
    if not notebook:
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")

    tenant = TenantScope(user_id=user_id, notebook_id=notebook_id)
    conversation_id = payload.conversation_id or str(uuid.uuid4())

    # 2. Multi-turn Chat History & Tenant Ownership Verification on conversation_id
    existing_messages_res = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
    )
    existing_messages = list(existing_messages_res.scalars().all())

    # Strict Tenant Check on Conversation History
    if existing_messages:
        first_msg = existing_messages[0]
        if first_msg.user_id != user_id or first_msg.notebook_id != notebook_id:
            raise ForbiddenError("Access denied: conversation_id belongs to a different tenant.")

    chat_history_dicts = [
        {"role": m.role, "content": m.content}
        for m in existing_messages[-settings.MAX_CHAT_HISTORY_TURNS :]
    ]

    # 3. Hybrid Retrieval Pipeline (Query Rewrite -> Qdrant Dense + Sparse -> RRF -> Rerank -> Refusal Check -> Parent Context)
    retrieval_service = RetrievalService(
        embedder=shared_embedder,
        vector_store=shared_vector_store,
    )
    retrieved_blocks, context_map, latencies_ms, is_refusal = await retrieval_service.retrieve_context(
        query=payload.query,
        tenant=tenant,
        db=db,
        chat_history=chat_history_dicts,
        notebook_title=notebook.title,
        top_k=5,
        min_score=payload.relevance_threshold,
    )

    # Reranker-Driven Refusal Path
    if is_refusal or not retrieved_blocks:
        refusal_answer = "I am sorry, but the provided sources do not contain relevant information to answer your query."
        # Persist conversation
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query))
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=refusal_answer))
        await db.commit()

        return ChatQueryResponse(
            answer=refusal_answer,
            citations=[],
            conversation_id=conversation_id,
            is_refusal=True,
            refusal_reason="RERANKER_REFUSAL_BELOW_THRESHOLD",
            prompt_version="qa_v1",
            model_name=settings.GEMINI_MODEL,
            latencies_ms=latencies_ms,
        )

    # 4. Attach figure crop images for retrieved figure elements
    storage_adapter = get_storage_adapter()
    crop_images: list[ImagePart] = []

    element_ids = [b["element_id"] for b in retrieved_blocks if b.get("element_id")]
    if element_ids:
        elem_res = await db.execute(select(Element).where(Element.id.in_(element_ids)))
        elements_by_id = {e.id: e for e in elem_res.scalars().all()}

        for b in retrieved_blocks:
            eid = b.get("element_id")
            if eid and eid in elements_by_id:
                elem = elements_by_id[eid]
                if elem.type == ElementType.FIGURE and elem.image_path:
                    try:
                        img_bytes = await storage_adapter.get(elem.image_path)
                        if img_bytes:
                            crop_images.append(ImagePart(bytes=img_bytes, mime_type="image/png"))
                    except Exception:
                        pass

    # 5. Assemble prompt with untrusted data delimiters
    context_text = "\n\n".join(
        f"--- Block {b['handle']} (Source: {b['source_id']}, Page: {b['page']}) ---\n{b['content']}"
        for b in retrieved_blocks
    )

    template = load_prompt_template(version="qa_v1")
    prompt = template.format(context_blocks=context_text, user_query=payload.query)

    # 6. Generate answer via LLM
    t_gen = time.time()
    llm: LLMGenerator = get_llm_generator()
    llm_resp = await llm.generate(prompt=prompt, images=crop_images if crop_images else None)
    latencies_ms["llm_generation_ms"] = (time.time() - t_gen) * 1000

    # 7. Resolve citations
    resolver = CitationResolver()
    cleaned_answer, resolved_citations, is_valid = resolver.resolve_and_validate(
        llm_text=llm_resp.text,
        retrieved_context_map=context_map,
    )

    if not is_valid or not resolved_citations:
        refusal_answer = "I am sorry, but the retrieved sources did not support a grounded answer."
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query))
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=refusal_answer))
        await db.commit()

        return ChatQueryResponse(
            answer=refusal_answer,
            citations=[],
            conversation_id=conversation_id,
            is_refusal=True,
            refusal_reason="ZERO_VALID_CITATIONS",
            prompt_version=llm_resp.prompt_version,
            model_name=llm_resp.model_name,
            latencies_ms=latencies_ms,
        )

    # Persist chat history
    citations_json = [c.model_dump() for c in resolved_citations]
    db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query))
    db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=cleaned_answer, citations=citations_json))
    await db.commit()

    return ChatQueryResponse(
        answer=cleaned_answer,
        citations=resolved_citations,
        conversation_id=conversation_id,
        is_refusal=False,
        prompt_version=llm_resp.prompt_version,
        model_name=llm_resp.model_name,
        latencies_ms=latencies_ms,
    )
