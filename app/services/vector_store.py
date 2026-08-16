import json
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.core.config import settings
from app.services.embeddings import Embedding, cosine_similarity, embed_text, embedding_model_for_vector
from app.services.storage_paths import get_document_storage_paths


MATRIX_INDEX_VERSION = 1
FAISS_INDEX_VERSION = 1
QDRANT_PROVIDER = "qdrant"
VECTOR_METADATA_FIELDS = (
    "chunk_id",
    "document_id",
    "filename",
    "position",
    "chunk_index",
    "page_number",
    "content",
    "token_count",
    "embedding_model",
    "created_at",
)


def _vector_metadata_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the source fields that must survive vector storage/retrieval."""
    return {
        field: entry.get(field)
        for field in VECTOR_METADATA_FIELDS
        if field in entry
    }


class VectorSearchBackend(Protocol):
    name: str

    def search(
        self,
        *,
        top_k: int,
        index_path: Path,
        min_score: float,
        query_embedding: Embedding,
        query_embedding_model: str,
        allowed_document_ids: AbstractSet[str],
    ) -> List[Dict[str, Any]] | None:
        ...


class VectorIndexBackend(Protocol):
    name: str

    def upsert_document(
        self,
        *,
        document_id: str,
        entries: List[Dict[str, Any]],
        index_path: Path,
    ) -> List[Dict[str, Any]] | None:
        ...

    def rebuild(
        self,
        *,
        entries: List[Dict[str, Any]],
        index_path: Path,
    ) -> List[Dict[str, Any]] | None:
        ...


def _default_vectors_file() -> Path:
    return get_document_storage_paths().vectors_file


def _resolve_vectors_file(index_path: Path | None) -> Path:
    return index_path or _default_vectors_file()


def _load_vectors(index_path: Path | None = None) -> List[Dict[str, Any]]:
    index_path = _resolve_vectors_file(index_path)
    if not index_path.exists():
        return []

    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_vectors(entries: List[Dict[str, Any]], index_path: Path | None = None) -> None:
    index_path = _resolve_vectors_file(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_matrix_index(entries, index_path)
    _save_faiss_index(entries, index_path)


def _matrix_index_path(index_path: Path) -> Path:
    return index_path.with_suffix(".npz")


def _matrix_metadata_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.metadata.json")


def _faiss_index_path(index_path: Path) -> Path:
    return index_path.with_suffix(".faiss")


def _faiss_metadata_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.faiss.metadata.json")


def _is_dense_embedding(embedding: Any) -> bool:
    return isinstance(embedding, list) and all(isinstance(value, (int, float)) for value in embedding)


def _is_qdrant_configured() -> bool:
    return (
        _is_external_vector_store_configured()
        and settings.vector_store_external_provider.lower() == QDRANT_PROVIDER
    )


def _qdrant_url(path: str) -> str:
    base_url = settings.vector_store_url.rstrip("/")
    return f"{base_url}{path}"


def _qdrant_request(method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.vector_store_api_key:
        headers["api-key"] = settings.vector_store_api_key

    request = Request(
        _qdrant_url(path),
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=settings.vector_store_timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        return None
    except (OSError, URLError, TimeoutError):
        return None

    if not response_body:
        return {}

    try:
        return json.loads(response_body)
    except json.JSONDecodeError:
        return None


def _qdrant_collection_path() -> str:
    collection = quote(settings.vector_store_collection, safe="")
    return f"/collections/{collection}"


def _qdrant_points_path() -> str:
    return f"{_qdrant_collection_path()}/points"


def _qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"enterprise-rag-system/chunk/{chunk_id}"))


def _ensure_qdrant_collection(vector_size: int) -> bool:
    collection_path = _qdrant_collection_path()
    collection = _qdrant_request("GET", collection_path)
    if collection is not None:
        return True

    created = _qdrant_request(
        "PUT",
        collection_path,
        {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        },
    )
    return created is not None


def _qdrant_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    return _vector_metadata_entry(entry)


def _qdrant_result_to_chunk(
    item: Dict[str, Any],
    query_embedding_model: str,
) -> Dict[str, Any] | None:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        return None
    if payload.get("embedding_model") != query_embedding_model:
        return None

    return {
        **_vector_metadata_entry(payload),
        "score": round(float(item.get("score") or 0.0), 4),
        "retrieval_mode": "vector",
        "embedding_model": payload.get("embedding_model"),
        "query_embedding_model": query_embedding_model,
        "vector_store_backend": QDRANT_PROVIDER,
        "created_at": payload.get("created_at"),
    }


def _can_build_matrix_index(entries: List[Dict[str, Any]]) -> bool:
    if not entries:
        return True

    dimensions = None
    for entry in entries:
        embedding = entry.get("embedding")
        if not _is_dense_embedding(embedding):
            return False
        if dimensions is None:
            dimensions = len(embedding)
        elif len(embedding) != dimensions:
            return False
    return True


def _save_matrix_index(entries: List[Dict[str, Any]], index_path: Path) -> None:
    matrix_path = _matrix_index_path(index_path)
    metadata_path = _matrix_metadata_path(index_path)

    if not _can_build_matrix_index(entries):
        matrix_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return

    metadata = _vector_metadata(entries)

    if entries:
        matrix = np.asarray([entry["embedding"] for entry in entries], dtype=np.float32)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)

    np.savez_compressed(
        matrix_path,
        embeddings=matrix,
        version=np.asarray([MATRIX_INDEX_VERSION], dtype=np.int32),
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_faiss_module():
    try:
        import faiss  # type: ignore
    except ImportError:
        return None
    return faiss


def _vector_metadata(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_vector_metadata_entry(entry) for entry in entries]


def _save_faiss_index(entries: List[Dict[str, Any]], index_path: Path) -> None:
    faiss = _load_faiss_module()
    faiss_path = _faiss_index_path(index_path)
    metadata_path = _faiss_metadata_path(index_path)

    if faiss is None or not _can_build_matrix_index(entries):
        faiss_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return

    metadata = _vector_metadata(entries)
    if entries:
        matrix = np.asarray([entry["embedding"] for entry in entries], dtype=np.float32)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)

    if matrix.size == 0:
        faiss_path.unlink(missing_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "version": FAISS_INDEX_VERSION,
                    "dimensions": 0,
                    "entries": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(faiss_path))
    metadata_path.write_text(
        json.dumps(
            {
                "version": FAISS_INDEX_VERSION,
                "dimensions": int(matrix.shape[1]),
                "entries": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_matrix_index(index_path: Path) -> tuple[np.ndarray, List[Dict[str, Any]]] | None:
    matrix_path = _matrix_index_path(index_path)
    metadata_path = _matrix_metadata_path(index_path)
    if not matrix_path.exists() or not metadata_path.exists():
        return None

    try:
        with np.load(matrix_path) as data:
            version = int(data["version"][0])
            if version != MATRIX_INDEX_VERSION:
                return None
            matrix = np.asarray(data["embeddings"], dtype=np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None

    if len(metadata) != len(matrix):
        return None
    return matrix, metadata


def _load_faiss_index(index_path: Path):
    faiss = _load_faiss_module()
    if faiss is None:
        return None

    faiss_path = _faiss_index_path(index_path)
    metadata_path = _faiss_metadata_path(index_path)
    if not faiss_path.exists() or not metadata_path.exists():
        return None

    try:
        index = faiss.read_index(str(faiss_path))
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError):
        return None

    if metadata_payload.get("version") != FAISS_INDEX_VERSION:
        return None

    metadata = metadata_payload.get("entries")
    if not isinstance(metadata, list) or len(metadata) != index.ntotal:
        return None
    return index, metadata


def _search_faiss_index(
    top_k: int,
    index_path: Path,
    min_score: float,
    query_embedding: Embedding,
    query_embedding_model: str,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]] | None:
    if not _is_dense_embedding(query_embedding):
        return None

    loaded = _load_faiss_index(index_path)
    if loaded is None:
        return None

    index, metadata = loaded
    if index.ntotal == 0:
        return []

    query_vector = np.asarray([query_embedding], dtype=np.float32)
    if index.d != query_vector.shape[1]:
        return None

    search_k = min(max(top_k * 4, top_k), index.ntotal)
    while True:
        scores, indexes = index.search(query_vector, search_k)
        scored_chunks = []
        for score, metadata_index in zip(scores[0], indexes[0]):
            if metadata_index < 0:
                continue
            score_value = float(score)
            if score_value <= min_score:
                continue

            entry = metadata[int(metadata_index)]
            document_id = entry.get("document_id")
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id not in allowed_document_ids
            ):
                continue
            if entry.get("embedding_model") != query_embedding_model:
                continue

            scored_chunks.append(
                {
                    **_vector_metadata_entry(entry),
                    "score": round(score_value, 4),
                    "retrieval_mode": "vector",
                    "embedding_model": entry.get("embedding_model"),
                    "query_embedding_model": query_embedding_model,
                    "vector_store_backend": "faiss",
                    "created_at": entry.get("created_at"),
                }
            )
            if len(scored_chunks) >= top_k:
                break
        if len(scored_chunks) >= top_k or search_k >= index.ntotal:
            return scored_chunks
        search_k = min(search_k * 2, index.ntotal)


def _search_matrix_index(
    top_k: int,
    index_path: Path,
    min_score: float,
    query_embedding: Embedding,
    query_embedding_model: str,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]] | None:
    if not _is_dense_embedding(query_embedding):
        return None

    loaded = _load_matrix_index(index_path)
    if loaded is None:
        return None

    matrix, metadata = loaded
    if matrix.size == 0:
        return []

    query_vector = np.asarray(query_embedding, dtype=np.float32)
    if matrix.shape[1] != query_vector.shape[0]:
        return None

    scores = matrix @ query_vector
    candidate_indexes = np.argsort(scores)[::-1]
    scored_chunks = []

    for index in candidate_indexes:
        score = float(scores[index])
        if score <= min_score:
            continue

        entry = metadata[int(index)]
        document_id = entry.get("document_id")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id not in allowed_document_ids
        ):
            continue
        if entry.get("embedding_model") != query_embedding_model:
            continue

        scored_chunks.append(
            {
                **_vector_metadata_entry(entry),
                "score": round(score, 4),
                "retrieval_mode": "vector",
                "embedding_model": entry.get("embedding_model"),
                "query_embedding_model": query_embedding_model,
                "vector_store_backend": "numpy",
                "created_at": entry.get("created_at"),
            }
        )
        if len(scored_chunks) >= top_k:
            break

    return scored_chunks


def _search_json_index(
    top_k: int,
    index_path: Path,
    min_score: float,
    query_embedding: Embedding,
    query_embedding_model: str,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    scored_chunks = []

    for entry in _load_vectors(index_path):
        document_id = entry.get("document_id")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id not in allowed_document_ids
        ):
            continue
        if entry.get("embedding_model") != query_embedding_model:
            continue

        score = cosine_similarity(query_embedding, entry.get("embedding", {}))
        if score > min_score:
            scored_chunks.append(
                {
                    **_vector_metadata_entry(entry),
                    "score": round(score, 4),
                    "retrieval_mode": "vector",
                    "embedding_model": entry.get("embedding_model"),
                    "query_embedding_model": query_embedding_model,
                    "vector_store_backend": "json",
                    "created_at": entry.get("created_at"),
                }
            )

    scored_chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
    return scored_chunks[:top_k]


class FaissVectorSearchBackend:
    name = "faiss"

    def search(
        self,
        *,
        top_k: int,
        index_path: Path,
        min_score: float,
        query_embedding: Embedding,
        query_embedding_model: str,
        allowed_document_ids: AbstractSet[str],
    ) -> List[Dict[str, Any]] | None:
        return _search_faiss_index(
            top_k,
            index_path,
            min_score,
            query_embedding,
            query_embedding_model,
            allowed_document_ids,
        )


class NumpyVectorSearchBackend:
    name = "numpy"

    def search(
        self,
        *,
        top_k: int,
        index_path: Path,
        min_score: float,
        query_embedding: Embedding,
        query_embedding_model: str,
        allowed_document_ids: AbstractSet[str],
    ) -> List[Dict[str, Any]] | None:
        return _search_matrix_index(
            top_k,
            index_path,
            min_score,
            query_embedding,
            query_embedding_model,
            allowed_document_ids,
        )


class JsonVectorSearchBackend:
    name = "json"

    def search(
        self,
        *,
        top_k: int,
        index_path: Path,
        min_score: float,
        query_embedding: Embedding,
        query_embedding_model: str,
        allowed_document_ids: AbstractSet[str],
    ) -> List[Dict[str, Any]]:
        return _search_json_index(
            top_k,
            index_path,
            min_score,
            query_embedding,
            query_embedding_model,
            allowed_document_ids,
        )


class ExternalVectorSearchBackend:
    name = "external"

    def search(
        self,
        *,
        top_k: int,
        index_path: Path,
        min_score: float,
        query_embedding: Embedding,
        query_embedding_model: str,
        allowed_document_ids: AbstractSet[str],
    ) -> List[Dict[str, Any]] | None:
        if not _is_qdrant_configured() or not _is_dense_embedding(query_embedding):
            return None

        response = _qdrant_request(
            "POST",
            f"{_qdrant_points_path()}/search",
            {
                "vector": query_embedding,
                "limit": top_k,
                "with_payload": True,
                "score_threshold": min_score,
                "filter": {
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"any": sorted(allowed_document_ids)},
                        }
                    ]
                },
            },
        )
        if response is None:
            return None

        results = response.get("result")
        if not isinstance(results, list):
            return None

        chunks = []
        for item in results:
            if not isinstance(item, dict):
                continue
            chunk = _qdrant_result_to_chunk(item, query_embedding_model)
            if chunk is None:
                continue
            document_id = chunk.get("document_id")
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id not in allowed_document_ids
            ):
                continue
            chunks.append(chunk)
            if len(chunks) >= top_k:
                break

        return chunks


def _is_external_vector_store_configured() -> bool:
    return bool(
        settings.vector_store_external_provider
        and settings.vector_store_url
        and settings.vector_store_collection
    )


class LocalVectorIndexBackend:
    name = "local"

    def upsert_document(
        self,
        *,
        document_id: str,
        entries: List[Dict[str, Any]],
        index_path: Path,
    ) -> List[Dict[str, Any]]:
        existing_entries = [
            entry for entry in _load_vectors(index_path)
            if entry["document_id"] != document_id
        ]
        _save_vectors(existing_entries + entries, index_path)
        return entries

    def rebuild(
        self,
        *,
        entries: List[Dict[str, Any]],
        index_path: Path,
    ) -> List[Dict[str, Any]]:
        _save_vectors(entries, index_path)
        return entries


class ExternalVectorIndexBackend:
    name = "external"

    def upsert_document(
        self,
        *,
        document_id: str,
        entries: List[Dict[str, Any]],
        index_path: Path,
    ) -> List[Dict[str, Any]] | None:
        if not _is_qdrant_configured() or not _can_build_matrix_index(entries):
            return None
        if not entries:
            return []
        vector_size = len(entries[0]["embedding"])
        if not _ensure_qdrant_collection(vector_size):
            return None

        deleted = _qdrant_request(
            "POST",
            f"{_qdrant_points_path()}/delete?wait=true",
            {
                "filter": {
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"value": document_id},
                        }
                    ]
                }
            },
        )
        if deleted is None:
            return None

        points = [
            {
                "id": _qdrant_point_id(entry["chunk_id"]),
                "vector": entry["embedding"],
                "payload": _qdrant_payload(entry),
            }
            for entry in entries
        ]
        upserted = _qdrant_request(
            "PUT",
            f"{_qdrant_points_path()}?wait=true",
            {"points": points},
        )
        if upserted is None:
            return None
        return entries

    def rebuild(
        self,
        *,
        entries: List[Dict[str, Any]],
        index_path: Path,
    ) -> List[Dict[str, Any]] | None:
        if not _is_qdrant_configured() or not _can_build_matrix_index(entries):
            return None
        if not entries:
            return []
        vector_size = len(entries[0]["embedding"])
        if not _ensure_qdrant_collection(vector_size):
            return None

        points = [
            {
                "id": _qdrant_point_id(entry["chunk_id"]),
                "vector": entry["embedding"],
                "payload": _qdrant_payload(entry),
            }
            for entry in entries
        ]
        upserted = _qdrant_request(
            "PUT",
            f"{_qdrant_points_path()}?wait=true",
            {"points": points},
        )
        if upserted is None:
            return None
        return entries


def get_vector_search_backends(backend: str | None = None) -> List[VectorSearchBackend]:
    selected_backend = (backend or settings.vector_store_backend).lower()

    if selected_backend in {"external", "qdrant", "pgvector", "milvus"}:
        return [
            ExternalVectorSearchBackend(),
            FaissVectorSearchBackend(),
            NumpyVectorSearchBackend(),
            JsonVectorSearchBackend(),
        ]
    if selected_backend == "faiss":
        return [FaissVectorSearchBackend(), NumpyVectorSearchBackend(), JsonVectorSearchBackend()]
    if selected_backend in {"numpy", "matrix"}:
        return [NumpyVectorSearchBackend(), JsonVectorSearchBackend()]
    if selected_backend == "json":
        return [JsonVectorSearchBackend()]
    return [FaissVectorSearchBackend(), NumpyVectorSearchBackend(), JsonVectorSearchBackend()]


def get_vector_index_backends(backend: str | None = None) -> List[VectorIndexBackend]:
    selected_backend = (backend or settings.vector_store_backend).lower()

    if selected_backend in {"external", "qdrant", "pgvector", "milvus"}:
        return [ExternalVectorIndexBackend(), LocalVectorIndexBackend()]
    return [LocalVectorIndexBackend()]


def _build_vector_entries(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entries = []
    for chunk in chunks:
        embedding = embed_text(chunk["content"])
        entries.append(
            {
                **_vector_metadata_entry(chunk),
                "embedding": embedding,
                "embedding_model": embedding_model_for_vector(embedding),
                "created_at": chunk.get("created_at"),
            }
        )
    return entries


def index_vector_chunks(
    document_id: str,
    chunks: List[Dict[str, Any]],
    index_path: Path | None = None,
) -> List[Dict[str, Any]]:
    index_path = _resolve_vectors_file(index_path)
    new_entries = _build_vector_entries(chunks)

    for backend in get_vector_index_backends():
        indexed_entries = backend.upsert_document(
            document_id=document_id,
            entries=new_entries,
            index_path=index_path,
        )
        if indexed_entries is not None:
            return indexed_entries

    return new_entries


def rebuild_vector_index(
    chunks: List[Dict[str, Any]],
    index_path: Path | None = None,
) -> List[Dict[str, Any]]:
    index_path = _resolve_vectors_file(index_path)
    entries = _build_vector_entries(chunks)

    for backend in get_vector_index_backends():
        indexed_entries = backend.rebuild(
            entries=entries,
            index_path=index_path,
        )
        if indexed_entries is not None:
            return indexed_entries

    return entries


def search_vector_chunks(
    question: str,
    top_k: int = 3,
    index_path: Path | None = None,
    min_score: float = 0.0,
    *,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    if not allowed_document_ids:
        return []
    index_path = _resolve_vectors_file(index_path)
    query_embedding = embed_text(question)
    query_embedding_model = embedding_model_for_vector(query_embedding)
    for backend in get_vector_search_backends():
        results = backend.search(
            top_k=top_k,
            index_path=index_path,
            min_score=min_score,
            query_embedding=query_embedding,
            query_embedding_model=query_embedding_model,
            allowed_document_ids=allowed_document_ids,
        )
        if results is not None:
            return results

    return []
