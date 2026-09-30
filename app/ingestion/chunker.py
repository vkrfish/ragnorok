from typing import List, Optional
from pydantic import BaseModel
from app.domain.models import ElementType
from app.ingestion.parser import ParsedElement


class ChunkData(BaseModel):
    element_index: int
    page_number: int
    content: str
    embeddable_text: str
    token_count: int
    heading_path: Optional[str] = None
    bbox: Optional[dict] = None
    is_embeddable: bool = True


class SectionAwareChunker:
    """Chunker implementing context prepending, table preservation, and figure handling rules."""

    def __init__(self, target_tokens: int = 400, overlap_pct: float = 0.10):
        self.target_tokens = target_tokens
        self.overlap_tokens = int(target_tokens * overlap_pct)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def chunk_elements(self, elements: List[ParsedElement]) -> List[ChunkData]:
        chunks: List[ChunkData] = []
        heading_stack: List[str] = []

        for idx, elem in enumerate(elements):
            if elem.type == ElementType.HEADING:
                heading_stack.append(elem.content)
                heading_path = " > ".join(heading_stack)

                chunks.append(
                    ChunkData(
                        element_index=idx,
                        page_number=elem.page_number,
                        content=elem.content,
                        embeddable_text=f"[Section: {heading_path}]\n{elem.content}",
                        token_count=self._estimate_tokens(elem.content),
                        heading_path=heading_path,
                        bbox=elem.bbox,
                        is_embeddable=True,
                    )
                )

            elif elem.type == ElementType.TABLE:
                heading_path = " > ".join(heading_stack) if heading_stack else None
                prefix = f"[Section: {heading_path}]\n" if heading_path else ""

                # Summary is used for embedding, raw table markdown is stored in content
                summary_text = elem.summary or f"Table on page {elem.page_number}: {elem.content[:100]}"
                embeddable = f"{prefix}{summary_text}"

                chunks.append(
                    ChunkData(
                        element_index=idx,
                        page_number=elem.page_number,
                        content=elem.content,
                        embeddable_text=embeddable,
                        token_count=self._estimate_tokens(elem.content),
                        heading_path=heading_path,
                        bbox=elem.bbox,
                        is_embeddable=True,
                    )
                )

            elif elem.type == ElementType.FIGURE:
                heading_path = " > ".join(heading_stack) if heading_stack else None
                # Non-Negotiable: Do NOT embed placeholder figure captions!
                is_embeddable = bool(elem.summary and elem.caption_status == "completed")
                embeddable = f"[Figure: {elem.summary}]" if is_embeddable else ""

                chunks.append(
                    ChunkData(
                        element_index=idx,
                        page_number=elem.page_number,
                        content=elem.content,
                        embeddable_text=embeddable,
                        token_count=self._estimate_tokens(elem.content),
                        heading_path=heading_path,
                        bbox=elem.bbox,
                        is_embeddable=is_embeddable,
                    )
                )

            else:  # ElementType.TEXT
                heading_path = " > ".join(heading_stack) if heading_stack else None
                prefix = f"[Section: {heading_path}]\n" if heading_path else ""
                tokens = self._estimate_tokens(elem.content)

                if tokens <= self.target_tokens:
                    chunks.append(
                        ChunkData(
                            element_index=idx,
                            page_number=elem.page_number,
                            content=elem.content,
                            embeddable_text=f"{prefix}{elem.content}",
                            token_count=tokens,
                            heading_path=heading_path,
                            bbox=elem.bbox,
                            is_embeddable=True,
                        )
                    )
                else:
                    # Slit text into chunks with ~10% overlap
                    words = elem.content.split()
                    words_per_chunk = self.target_tokens * 3
                    overlap_words = self.overlap_tokens * 3
                    step = max(1, words_per_chunk - overlap_words)

                    for start in range(0, len(words), step):
                        sub_words = words[start : start + words_per_chunk]
                        sub_text = " ".join(sub_words)
                        if not sub_text.strip():
                            continue

                        chunks.append(
                            ChunkData(
                                element_index=idx,
                                page_number=elem.page_number,
                                content=sub_text,
                                embeddable_text=f"{prefix}{sub_text}",
                                token_count=self._estimate_tokens(sub_text),
                                heading_path=heading_path,
                                bbox=elem.bbox,
                                is_embeddable=True,
                            )
                        )

        return chunks
