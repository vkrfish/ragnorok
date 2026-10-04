import asyncio
import json

from pydantic import BaseModel, Field

from app.adapters.llm import LLMGenerator, get_llm_generator
from app.core.config import settings
from app.core.logging import logger
from app.generation.prompts import load_prompt_template


class QueryRewriteResult(BaseModel):
    is_clear: bool = True
    standalone_query: str
    sub_queries: list[str] = Field(default_factory=list)


class QueryRewriter:
    """Query Rewriter implementing LLM JSON understanding, timeout fallback, and entity preservation."""

    def __init__(self, llm_generator: LLMGenerator | None = None):
        self.llm = llm_generator or get_llm_generator()

    async def rewrite(
        self,
        user_query: str,
        chat_history: list[dict[str, str]] | None = None,
        notebook_title: str = "Research Notebook",
        timeout_sec: float = 2.0,
    ) -> QueryRewriteResult:
        """Rewrite user query using conversation history. Retains original user_query in search set."""
        if not settings.ENABLE_QUERY_REWRITE or not chat_history:
            return QueryRewriteResult(
                is_clear=True,
                standalone_query=user_query,
                sub_queries=[user_query],
            )

        # Format chat history safely
        history_lines = []
        for msg in chat_history[-settings.MAX_CHAT_HISTORY_TURNS:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        template = load_prompt_template("query_rewrite_v1")
        prompt = (
            template.replace("{notebook_title}", notebook_title)
            .replace("{chat_history}", history_text)
            .replace("{user_query}", user_query)
        )

        try:
            # Call LLM with timeout (2.0s)
            res = await asyncio.wait_for(self.llm.generate(prompt=prompt), timeout=timeout_sec)
            text = res.text.strip()

            text = text.removeprefix("```json")
            text = text.removesuffix("```")
            text = text.strip()

            data = json.loads(text)
            standalone = data.get("standalone_query", user_query)
            sub_q = data.get("sub_queries", [])

            # Ensure original user_query is retained in search set
            all_queries = [user_query]
            if standalone and standalone != user_query:
                all_queries.append(standalone)
            for sq in sub_q:
                if sq and sq not in all_queries:
                    all_queries.append(sq)

            return QueryRewriteResult(
                is_clear=data.get("is_clear", False),
                standalone_query=standalone,
                sub_queries=all_queries,
            )

        except Exception as e:
            logger.warning(f"Query rewrite fallback ({e}). Using original query.")
            return QueryRewriteResult(
                is_clear=True,
                standalone_query=user_query,
                sub_queries=[user_query],
            )
