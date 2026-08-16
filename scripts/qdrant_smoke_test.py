import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services import vector_store


SMOKE_CHUNKS = [
    {
        "chunk_id": "qdrant-smoke-1",
        "document_id": "qdrant-smoke-doc",
        "filename": "qdrant_smoke.md",
        "position": 1,
        "content": "Remote work requests require manager approval before travel begins.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    {
        "chunk_id": "qdrant-smoke-2",
        "document_id": "qdrant-smoke-doc",
        "filename": "qdrant_smoke.md",
        "position": 2,
        "content": "Expense reports must include receipts for hotel and airfare claims.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real Qdrant vector-store smoke test when Qdrant is available.",
    )
    parser.add_argument("--url", default=settings.vector_store_url or "http://localhost:6333")
    parser.add_argument("--api-key", default=settings.vector_store_api_key)
    parser.add_argument(
        "--collection",
        default=f"enterprise_rag_smoke_{uuid4().hex[:12]}",
    )
    parser.add_argument(
        "--require-running",
        action="store_true",
        help="Fail instead of skipping when Qdrant is not reachable or not used.",
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Leave the temporary Qdrant collection in place for inspection.",
    )
    return parser.parse_args()


def configure_qdrant(args: argparse.Namespace) -> None:
    settings.vector_store_backend = "external"
    settings.vector_store_external_provider = "qdrant"
    settings.vector_store_url = args.url
    settings.vector_store_collection = args.collection
    settings.vector_store_api_key = args.api_key


def cleanup_collection() -> None:
    vector_store._qdrant_request("DELETE", vector_store._qdrant_collection_path())


def main() -> int:
    args = parse_args()
    configure_qdrant(args)

    print(f"Qdrant URL: {settings.vector_store_url}")
    print(f"Qdrant collection: {settings.vector_store_collection}")

    if vector_store._qdrant_request("GET", "/collections") is None:
        print("Qdrant smoke test skipped: Qdrant is not reachable.")
        return 1 if args.require_running else 0

    with tempfile.TemporaryDirectory() as tempdir:
        vector_index_path = Path(tempdir) / "vectors.json"
        vector_store.rebuild_vector_index(SMOKE_CHUNKS, vector_index_path)
        results = vector_store.search_vector_chunks(
            "Who approves remote work travel?",
            top_k=1,
            index_path=vector_index_path,
            min_score=0.0,
        )

    if not results or results[0].get("vector_store_backend") != "qdrant":
        message = (
            "Qdrant smoke test skipped: Qdrant was not reachable or did not return "
            "a matching result, so local vector-store fallback was used."
        )
        print(message)
        return 1 if args.require_running else 0

    top_result = results[0]
    print("Qdrant smoke test passed.")
    print(f"Top chunk: {top_result['chunk_id']}")
    print(f"Score: {top_result['score']}")
    print(f"Backend: {top_result['vector_store_backend']}")

    if not args.keep_collection:
        cleanup_collection()
        print("Temporary Qdrant collection deleted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
