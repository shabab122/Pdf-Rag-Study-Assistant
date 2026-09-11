"""Shared embedding and threshold-based retrieval helpers."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterable
from dotenv import load_dotenv

load_dotenv()


DATABASE_FOLDER = Path(os.getenv("CHROMA_DB_PATH", "chroma_db"))
COLLECTION_NAME = "course_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RESULT_COUNT = 3
DEFAULT_DISTANCE_THRESHOLD = 0.75
DISTANCE_THRESHOLD_ENV = "RAG_DISTANCE_THRESHOLD"
NO_RELEVANT_CONTEXT_MESSAGE = "Dukkhoito, provided PDF-e ei prosner uttor pawa jayni."


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int
    distance: float


@dataclass(frozen=True)
class RetrievalResponse:
    """Accepted chunks plus all candidates returned by Chroma."""

    chunks: tuple[RetrievedChunk, ...]
    candidates: tuple[RetrievedChunk, ...]
    threshold: float

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)

    @property
    def best_candidate(self) -> RetrievedChunk | None:
        return self.candidates[0] if self.candidates else None

    def format_context(self) -> str:
        return "\n\n---\n\n".join(
            f"Source: {chunk.source}, Page: {chunk.page}\n{chunk.text}"
            for chunk in self.chunks
        )


def get_distance_threshold(value: float | None = None) -> float:
    """Return the configured maximum Chroma distance (lower is better)."""
    if value is None:
        raw_value = os.getenv(DISTANCE_THRESHOLD_ENV)
        value = DEFAULT_DISTANCE_THRESHOLD if raw_value is None else float(raw_value)

    threshold = float(value)
    if threshold < 0:
        raise ValueError("RAG distance threshold cannot be negative")
    return threshold


def filter_relevant_chunks(
    candidates: Iterable[RetrievedChunk], threshold: float
) -> tuple[RetrievedChunk, ...]:
    """Keep candidates at or below the maximum distance threshold."""
    threshold = get_distance_threshold(threshold)
    return tuple(chunk for chunk in candidates if chunk.distance <= threshold)


@lru_cache(maxsize=1)
def load_embedding_model() -> Any:
    # Lazy import keeps pure threshold tests usable without ML dependencies.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve(
    question: str,
    result_count: int = DEFAULT_RESULT_COUNT,
    distance_threshold: float | None = None,
    source: str | None = None,
) -> RetrievalResponse:
    """Embed a question, optionally limit it to one PDF, then filter by threshold."""
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if result_count <= 0:
        raise ValueError("result_count must be greater than zero")

    threshold = get_distance_threshold(distance_threshold)

    import chromadb

    model = load_embedding_model()
    question_embedding = model.encode(question).tolist()
    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))
    collection = client.get_collection(COLLECTION_NAME)
    collection_size = collection.count()

    if collection_size == 0:
        return RetrievalResponse((), (), threshold)

    query_options: dict[str, Any] = {
        "query_embeddings": [question_embedding],
        "n_results": min(result_count, collection_size),
        "include": ["documents", "metadatas", "distances"],
    }
    if source:
        query_options["where"] = {"source": source}

    results = collection.query(**query_options)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    candidates = tuple(
        RetrievedChunk(
            text=document,
            source=str(metadata.get("source", "Unknown source")),
            page=int(metadata.get("page", 0)),
            distance=float(distance),
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    )

    return RetrievalResponse(
        chunks=filter_relevant_chunks(candidates, threshold),
        candidates=candidates,
        threshold=threshold,
    )
