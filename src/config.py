"""Shared project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FOLDER = PROJECT_ROOT / "data"
DATABASE_FOLDER = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "course_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
