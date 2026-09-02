"""Shared embedding-model loader with in-process caching."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it for later questions."""
    return SentenceTransformer(EMBEDDING_MODEL)
