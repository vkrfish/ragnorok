from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a dense vector."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into dense vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension size."""
        pass


class FakeEmbedder(Embedder):
    """Deterministic Fake Embedder for testing without external models."""

    def __init__(self, dim: int = 1024):
        self._dim = dim

    def _generate_vector(self, text: str) -> List[float]:
        # Hash text to generate deterministic float vector
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        val = sum(h) / (255.0 * len(h))
        vec = [(val + i * 0.001) % 1.0 for i in range(self._dim)]
        return vec

    async def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim
