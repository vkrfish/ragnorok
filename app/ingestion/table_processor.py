
from app.adapters.llm import LLMGenerator, get_llm_generator
from app.core.logging import logger


class TableProcessor:
    """Non-inferential table summarizer enforcing strict restatement of table contents."""

    def __init__(self, llm_generator: LLMGenerator | None = None):
        self.llm_generator = llm_generator or get_llm_generator()

    async def summarize_table(self, table_content: str, max_part_chars: int = 4000) -> str:
        """Summarize table non-inferentially, handling large multi-page tables in parts if needed."""
        if len(table_content) <= max_part_chars:
            return await self._summarize_part(table_content)

        # Split large table content into parts
        parts = [table_content[i : i + max_part_chars] for i in range(0, len(table_content), max_part_chars)]
        part_summaries = []

        for idx, part in enumerate(parts, start=1):
            logger.info(f"Summarizing large table part {idx}/{len(parts)}")
            summary = await self._summarize_part(part)
            part_summaries.append(f"Part {idx}: {summary}")

        return "\n".join(part_summaries)

    async def _summarize_part(self, content: str) -> str:
        prompt = (
            f"You are a precise table summarizer. Restate ONLY what the table explicitly contains "
            f"(column headers, row categories, and cell values). DO NOT infer trends or make extrapolations.\n\n"
            f"<UNTRUSTED_TABLE_CONTENT>\n{content}\n</UNTRUSTED_TABLE_CONTENT>"
        )

        try:
            res = await self.llm_generator.generate(prompt=prompt)
            return res.text.strip()
        except Exception as e:
            logger.error(f"Table summarization failed: {e}")
            # Fallback: simple header restatement
            lines = [line for line in content.splitlines() if line.strip()]
            headers = lines[0] if lines else "Table data"
            return f"Table containing columns: {headers}"
