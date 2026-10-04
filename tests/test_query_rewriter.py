import pytest

from app.adapters.llm import FakeLLMGenerator
from app.retrieval.rewriter import QueryRewriter


@pytest.mark.asyncio
async def test_query_rewriter_entity_preservation():
    llm = FakeLLMGenerator()
    rewriter = QueryRewriter(llm_generator=llm)

    chat_history = [
        {"role": "user", "content": "Tell me about the LEARNX platform."},
        {"role": "assistant", "content": "LEARNX is a personalized learning workspace."},
    ]

    res = await rewriter.rewrite(
        user_query="What are its key features?",
        chat_history=chat_history,
        notebook_title="LEARNX Research",
    )

    assert isinstance(res.sub_queries, list)
    assert "What are its key features?" in res.sub_queries  # Original query preserved!
