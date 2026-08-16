from dataclasses import dataclass
from pathlib import Path

from app.core.config import INDEX_DIR, KNOWLEDGE_BASE_FILE, TEXT_DIR, UPLOAD_DIR, VECTOR_INDEX_FILE


@dataclass(frozen=True)
class DocumentStoragePaths:
    upload_dir: Path
    text_dir: Path
    index_dir: Path
    chunks_file: Path
    vectors_file: Path


def get_document_storage_paths() -> DocumentStoragePaths:
    """
    Return filesystem paths used by document APIs and indexes.

    Keeping this behind a function makes API, knowledge-base, and vector-index
    tests easier to isolate without patching module-level config constants.
    """
    return DocumentStoragePaths(
        upload_dir=UPLOAD_DIR,
        text_dir=TEXT_DIR,
        index_dir=INDEX_DIR,
        chunks_file=KNOWLEDGE_BASE_FILE,
        vectors_file=VECTOR_INDEX_FILE,
    )
