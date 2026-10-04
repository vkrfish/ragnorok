import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger


class RerankedChunk(BaseModel):
    chunk_id: str
    source_id: str
    element_id: str | None = None
    page_number: int
    score: float
    content: str
    heading_path: str | None = None
    bbox: dict[str, Any] | None = None


class Reranker(ABC):
    """Abstract Cross-Encoder Reranker interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        """Rerank candidates using cross-attention scoring against the ORIGINAL user query.
        Returns: (reranked_candidates, rerank_latency_ms)
        """


class BGEReranker(Reranker):
    """BGE-Reranker-v2-m3 Cross-Encoder implementation with fast fallback."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Initialized CrossEncoder reranker with model '{self.model_name}'")
        except Exception as e:
            logger.info(f"CrossEncoder model '{self.model_name}' not loaded ({e}). Using fast heuristic reranker fallback.")

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        start_time = time.time()
        if not candidates:
            return [], 0.0

        if not self._model:
            # Fast heuristic reranker fallback based on exact word match density & RRF rank
            reranked = self._heuristic_rerank(query, candidates)
            latency_ms = (time.time() - start_time) * 1000
            return reranked[:top_n], latency_ms

        try:
            pairs = [(query, c.get("content", "")) for c in candidates]
            scores = self._model.predict(pairs)

            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)

            candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            latency_ms = (time.time() - start_time) * 1000
            return candidates[:top_n], latency_ms

        except Exception as e:
            logger.error(f"CrossEncoder reranking error: {e}")
            reranked = self._heuristic_rerank(query, candidates)
            latency_ms = (time.time() - start_time) * 1000
            return reranked[:top_n], latency_ms

    def _heuristic_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        for c in candidates:
            text = c.get("content", "").lower()
            overlap = sum(1 for w in query_words if w in text)
            exact_phrase_bonus = 0.5 if query.lower() in text else 0.0
            rrf_base = c.get("rrf_score", 0.5)
            c["rerank_score"] = min(1.0, (overlap / max(1, len(query_words))) * 0.6 + rrf_base * 0.4 + exact_phrase_bonus)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates


class FakeReranker(Reranker):
    """Fake Reranker for deterministic unit testing."""

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        stopwords = {"what", "is", "the", "of", "in", "a", "an", "and", "or", "to", "for"}
        query_words = set(query.lower().replace("?", "").split()) - stopwords
        for c in candidates:
            text = c.get("content", "").lower()
            overlap = sum(1 for w in query_words if w in text)
            if "refusal" in query.lower() or overlap == 0:
                c["rerank_score"] = 0.10
            else:
                c["rerank_score"] = 0.85

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates[:top_n], 1.5


def get_reranker() -> Reranker:
    """Factory function returning configured Reranker."""
    return BGEReranker()
