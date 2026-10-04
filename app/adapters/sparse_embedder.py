import hashlib
import re

from pydantic import BaseModel


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbedder:
    """Qdrant Sparse Vector Embedder using BM25 token hashing for exact term matching."""

    def embed_text(self, text: str) -> SparseVector:
        # Tokenize words, exact alphanumeric terms (Q3-2024, SKU-4471)
        tokens = re.findall(r"\b[A-Za-z0-9_\-]+\b", text.lower())
        if not tokens:
            return SparseVector(indices=[], values=[])

        term_freqs: dict[int, float] = {}
        for token in tokens:
            # Map token string to 32-bit positive integer index via SHA256
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
            term_freqs[index] = term_freqs.get(index, 0.0) + 1.0

        # Term frequency normalization
        total_terms = float(len(tokens))
        indices = list(term_freqs.keys())
        values = [tf / total_terms for tf in term_freqs.values()]

        return SparseVector(indices=indices, values=values)

    def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        return [self.embed_text(t) for t in texts]
