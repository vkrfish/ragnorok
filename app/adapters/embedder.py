from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a dense vector."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension size."""


class FakeEmbedder(Embedder):
    """Deterministic Fake Embedder for testing without external models."""

    def __init__(self, dim: int = 1024):
        self._dim = dim

    def _generate_vector(self, text: str) -> list[float]:
        # Hash text to generate deterministic float vector
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        val = sum(h) / (255.0 * len(h))
        vec = [(val + i * 0.001) % 1.0 for i in range(self._dim)]
        return vec

    async def embed_text(self, text: str) -> list[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim
