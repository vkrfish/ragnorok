import re

from app.domain.schemas import Citation


class CitationResolver:
    """Resolves short handles ([1], [2]) to full Citation models and validates grounding."""

    def resolve_and_validate(
        self,
        llm_text: str,
        retrieved_context_map: dict[str, dict], # e.g. {"[1]": {source_id, page, element_id, bbox, content}}
    ) -> tuple[str, list[Citation], bool]:
        """
        Parses llm_text for handles like [1], [2].
        Returns: (cleaned_text, resolved_citations, is_valid)
        """
        if "REFUSAL" in llm_text or not llm_text.strip():
            return "I am sorry, but the provided sources do not contain relevant information to answer your query.", [], False

        # Find handles like [1], [2] in llm_text
        found_handles = re.findall(r"\[(\d+)\]", llm_text)
        valid_citations: list[Citation] = []
        seen_handles = set()

        for h in found_handles:
            handle_key = f"[{h}]"
            if handle_key in retrieved_context_map and handle_key not in seen_handles:
                ctx = retrieved_context_map[handle_key]
                valid_citations.append(
                    Citation(
                        handle=handle_key,
                        source_id=ctx["source_id"],
                        page=ctx["page"],
                        element_id=ctx.get("element_id"),
                        bbox=ctx.get("bbox"),
                        snippet=ctx["content"][:150],
                    )
                )
                seen_handles.add(handle_key)

        # Refuse if zero valid citations present in generated text
        if not valid_citations:
            return "I am sorry, but the retrieved sources did not support a grounded answer.", [], False

        return llm_text, valid_citations, True
