import json
import time
import uuid

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.factory import get_storage_adapter
from app.adapters.llm import LLMGenerator, get_llm_generator
from app.core.config import settings
from app.core.db import get_db
from app.core.errors import NotFoundError
from app.core.logging import logger
from app.domain.models import ChatMessage, Element, ElementType, Notebook
from app.domain.schemas import (
    ChatMessageResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    ImagePart,
    PaginatedResponse,
    TenantScope,
    UsageBlock,
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
    """Synchronous grounded Q&A endpoint with active-source snapshot, tenant isolation, and citations."""
    # Verify notebook existence & tenant ownership (return 404 for non-existing/other user resources)
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

    # Verify conversation tenant ownership
    existing_messages_res = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
    )
    existing_messages = list(existing_messages_res.scalars().all())

    if existing_messages:
        first_msg = existing_messages[0]
        if first_msg.user_id != user_id or first_msg.notebook_id != notebook_id:
            raise NotFoundError("Conversation not found.")

    chat_history_dicts = [
        {"role": m.role, "content": m.content}
        for m in existing_messages[-settings.MAX_CHAT_HISTORY_TURNS :]
    ]

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
        override_source_ids=payload.source_ids,
    )

    if is_refusal or not retrieved_blocks:
        refusal_answer = "I am sorry, but the provided sources do not contain relevant information to answer your query."
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query, status="complete"))
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=refusal_answer, status="complete"))
        await db.commit()

        reason = "NO_ACTIVE_SOURCES" if is_refusal else "RERANKER_REFUSAL_BELOW_THRESHOLD"
        return ChatQueryResponse(
            answer=refusal_answer,
            citations=[],
            conversation_id=conversation_id,
            is_refusal=True,
            refusal_reason=reason,
            prompt_version="qa_v1",
            model_name=settings.GEMINI_MODEL,
            latencies_ms=latencies_ms,
        )

    # Attach figure crop images
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

    context_text = "\n\n".join(
        f"--- Block {b['handle']} (Source: {b['source_id']}, Page: {b['page']}) ---\n{b['content']}"
        for b in retrieved_blocks
    )
    template = load_prompt_template(version="qa_v1")
    prompt = template.format(context_blocks=context_text, user_query=payload.query)

    t_gen = time.time()
    llm: LLMGenerator = get_llm_generator()
    llm_resp = await llm.generate(prompt=prompt, images=crop_images if crop_images else None)
    latencies_ms["llm_generation_ms"] = (time.time() - t_gen) * 1000

    resolver = CitationResolver()
    cleaned_answer, resolved_citations, is_valid = resolver.resolve_and_validate(
        llm_text=llm_resp.text,
        retrieved_context_map=context_map,
    )

    if not is_valid or not resolved_citations:
        refusal_answer = "I am sorry, but the retrieved sources did not support a grounded answer."
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query, status="complete"))
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=refusal_answer, status="complete"))
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

    citations_json = [c.model_dump() for c in resolved_citations]
    db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query, status="complete"))
    db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=cleaned_answer, citations=citations_json, status="complete"))
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


@router.post("/{notebook_id}/chat/stream")
async def chat_query_stream(
    notebook_id: str,
    payload: ChatQueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """SSE Streaming Q&A endpoint emitting retrieval -> token -> citations -> done events with disconnect handling."""
    nb_res = await db.execute(select(Notebook).where(Notebook.id == notebook_id, Notebook.user_id == user_id))
    notebook = nb_res.scalar_one_or_none()
    if not notebook:
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")

    tenant = TenantScope(user_id=user_id, notebook_id=notebook_id)
    conversation_id = payload.conversation_id or str(uuid.uuid4())

    async def event_generator():
        # Always persist user message
        db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="user", content=payload.query, status="complete"))
        await db.commit()

        retrieval_service = RetrievalService(embedder=shared_embedder, vector_store=shared_vector_store)
        retrieved_blocks, context_map, latencies_ms, is_refusal = await retrieval_service.retrieve_context(
            query=payload.query,
            tenant=tenant,
            db=db,
            notebook_title=notebook.title,
            top_k=5,
            min_score=payload.relevance_threshold,
            override_source_ids=payload.source_ids,
        )

        if is_refusal or not retrieved_blocks:
            refusal_reason = "NO_ACTIVE_SOURCES" if is_refusal else "RERANKER_REFUSAL_BELOW_THRESHOLD"
            refusal_payload = {
                "answer": "I am sorry, but the provided sources do not contain relevant information to answer your query.",
                "reason": refusal_reason,
            }
            yield f"event: refusal\ndata: {json.dumps(refusal_payload)}\n\n"
            done_payload = UsageBlock(latencies_ms=latencies_ms, model_name=settings.GEMINI_MODEL, prompt_version="qa_v1").model_dump()
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            db.add(ChatMessage(conversation_id=conversation_id, notebook_id=notebook_id, user_id=user_id, role="assistant", content=refusal_payload["answer"], status="complete"))
            await db.commit()
            return

        # 1. Emit retrieval event
        candidate_handles = {
            h: {"source_id": b["source_id"], "page": b["page"], "snippet": b["content"][:100]}
            for h, b in context_map.items()
        }
        retrieval_payload = {"candidates": candidate_handles, "conversation_id": conversation_id}
        yield f"event: retrieval\ndata: {json.dumps(retrieval_payload)}\n\n"

        # 2. Stream tokens
        context_text = "\n\n".join(f"--- Block {b['handle']} ---\n{b['content']}" for b in retrieved_blocks)
        template = load_prompt_template("qa_v1")
        prompt = template.format(context_blocks=context_text, user_query=payload.query)

        llm: LLMGenerator = get_llm_generator()
        accumulated_text = ""
        stream_status = "complete"

        t_gen = time.time()
        try:
            async for token in llm.generate_stream(prompt=prompt):
                if await request.is_disconnected():
                    logger.warning("Client disconnected mid-stream.")
                    stream_status = "partial"
                    break

                accumulated_text += token
                token_payload = {"token": token}
                yield f"event: token\ndata: {json.dumps(token_payload)}\n\n"
                yield ": keepalive\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            stream_status = "failed"
            error_payload = {"error_code": "STREAM_ERROR", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

        latencies_ms["llm_generation_ms"] = (time.time() - t_gen) * 1000

        # 3. Resolve citations and emit citations event
        resolver = CitationResolver()
        cleaned_text, resolved_citations, is_valid = resolver.resolve_and_validate(
            llm_text=accumulated_text,
            retrieved_context_map=context_map,
        )

        removed_handles = list(set(context_map.keys()) - {c.handle for c in resolved_citations})

        citations_payload = {
            "citations": [c.model_dump() for c in resolved_citations],
            "removed_handles": removed_handles,
        }
        yield f"event: citations\ndata: {json.dumps(citations_payload)}\n\n"

        # 4. Emit done event
        prompt_tokens = len(prompt) // 4
        cand_tokens = len(accumulated_text) // 4
        usage = UsageBlock(
            prompt_tokens=prompt_tokens,
            candidate_tokens=cand_tokens,
            total_tokens=prompt_tokens + cand_tokens,
            estimated_cost_usd=0.00012,
            latencies_ms=latencies_ms,
            model_name=settings.GEMINI_MODEL,
            prompt_version="qa_v1",
        )
        yield f"event: done\ndata: {json.dumps(usage.model_dump())}\n\n"

        # Persist assistant message with stream status
        db.add(ChatMessage(
            conversation_id=conversation_id,
            notebook_id=notebook_id,
            user_id=user_id,
            role="assistant",
            content=cleaned_text if is_valid else "I am sorry, but the retrieved sources did not support a grounded answer.",
            citations=[c.model_dump() for c in resolved_citations] if is_valid else [],
            status=stream_status,
        ))
        await db.commit()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Conversation List Endpoint (Paginated)
@router.get("/{notebook_id}/conversations", response_model=PaginatedResponse[str])
async def list_conversations(
    notebook_id: str,
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Get cursor-paginated list of conversation IDs in notebook."""
    nb_res = await db.execute(select(Notebook).where(Notebook.id == notebook_id, Notebook.user_id == user_id))
    if not nb_res.scalar_one_or_none():
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")

    stmt = select(ChatMessage.conversation_id).where(
        ChatMessage.notebook_id == notebook_id,
        ChatMessage.user_id == user_id,
    ).distinct().order_by(ChatMessage.conversation_id.desc())

    if cursor:
        stmt = stmt.where(ChatMessage.conversation_id < cursor)

    stmt = stmt.limit(limit + 1)
    res = await db.execute(stmt)
    convos = list(res.scalars().all())

    has_more = len(convos) > limit
    items = convos[:limit]
    next_cursor = items[-1] if has_more and items else None

    return PaginatedResponse[str](
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


# Conversation History Endpoint (Paginated)
@router.get("/{notebook_id}/conversations/{conversation_id}", response_model=PaginatedResponse[ChatMessageResponse])
async def get_conversation_history(
    notebook_id: str,
    conversation_id: str,
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Get cursor-paginated chat messages for a specific conversation."""
    nb_res = await db.execute(select(Notebook).where(Notebook.id == notebook_id, Notebook.user_id == user_id))
    if not nb_res.scalar_one_or_none():
        raise NotFoundError(f"Notebook '{notebook_id}' not found.")

    stmt = select(ChatMessage).where(
        ChatMessage.conversation_id == conversation_id,
        ChatMessage.notebook_id == notebook_id,
        ChatMessage.user_id == user_id,
    ).order_by(ChatMessage.created_at.asc())

    res = await db.execute(stmt)
    messages = list(res.scalars().all())

    if not messages:
        raise NotFoundError("Conversation not found.")

    has_more = len(messages) > limit
    items = messages[:limit]
    next_cursor = items[-1].id if has_more and items else None

    return PaginatedResponse[ChatMessageResponse](
        items=[ChatMessageResponse.model_validate(m) for m in items],
        next_cursor=next_cursor,
        has_more=has_more,
    )
