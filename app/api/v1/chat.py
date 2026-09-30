from typing import Optional
from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.adapters.embedder import FakeEmbedder
from app.adapters.vector_store import InMemoryVectorStore
from app.adapters.llm import get_llm_generator, LLMGenerator
from app.core.db import get_db
from app.core.errors import NotFoundError
from app.domain.models import Notebook
from app.domain.schemas import ChatQueryRequest, ChatQueryResponse, TenantScope
from app.generation.citation import CitationResolver
from app.generation.prompts import load_prompt_template
from app.retrieval.search import RetrievalService
from app.workers.tasks import shared_embedder, shared_vector_store

router = APIRouter()


def get_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-ID")) -> str:
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
    """Grounded Q&A chat endpoint with strict tenant filtering, citation resolution, and refusal path."""
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

    # 2. Retrieve context using tenant scope
    retrieval_service = RetrievalService(
        embedder=shared_embedder,
        vector_store=shared_vector_store,
    )
    retrieved_blocks, context_map = await retrieval_service.retrieve_context(
        query=payload.query,
        tenant=tenant,
        db=db,
        top_k=5,
        min_score=payload.relevance_threshold,
    )

    # Refusal Path if zero context blocks retrieved above relevance threshold
    if not retrieved_blocks:
        return ChatQueryResponse(
            answer="I am sorry, but the provided sources do not contain relevant information to answer your query.",
            citations=[],
            is_refusal=True,
            refusal_reason="NO_RELEVANT_CONTEXT_BELOW_THRESHOLD",
            prompt_version="qa_v1",
            model_name="claude-3-5-sonnet",
        )

    # 3. Assemble prompt with untrusted data delimiters
    context_text = "\n\n".join(
        f"--- Block {b['handle']} (Source: {b['source_id']}, Page: {b['page']}) ---\n{b['content']}"
        for b in retrieved_blocks
    )

    template = load_prompt_template(version="qa_v1")
    prompt = template.format(context_blocks=context_text, user_query=payload.query)

    # 4. Generate answer via LLM
    llm: LLMGenerator = get_llm_generator()
    llm_resp = await llm.generate(prompt=prompt)

    # 5. Resolve short handles & validate citations
    resolver = CitationResolver()
    cleaned_answer, resolved_citations, is_valid = resolver.resolve_and_validate(
        llm_text=llm_resp.text,
        retrieved_context_map=context_map,
    )

    if not is_valid or not resolved_citations:
        return ChatQueryResponse(
            answer="I am sorry, but the retrieved sources did not support a grounded answer.",
            citations=[],
            is_refusal=True,
            refusal_reason="ZERO_VALID_CITATIONS",
            prompt_version=llm_resp.prompt_version,
            model_name=llm_resp.model_name,
        )

    return ChatQueryResponse(
        answer=cleaned_answer,
        citations=resolved_citations,
        is_refusal=False,
        prompt_version=llm_resp.prompt_version,
        model_name=llm_resp.model_name,
    )
