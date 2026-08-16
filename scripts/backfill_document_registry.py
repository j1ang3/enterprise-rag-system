import argparse
import sys
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_registry import (
    backfill_document_registry,
    build_historical_registrations,
    registered_document_ids,
)
from app.services.knowledge_base import get_all_chunks
from app.services.storage_paths import get_document_storage_paths
from app.services.user_registry import get_user_by_id


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical document metadata for one explicit owner."
    )
    parser.add_argument(
        "--owner-id",
        required=True,
        help="Existing application User UUID that will own inserted documents.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        owner_id = UUID(arguments.owner_id)
    except ValueError as exc:
        raise RuntimeError("--owner-id must be a valid UUID.") from exc
    if get_user_by_id(owner_id) is None:
        raise RuntimeError("--owner-id must reference an existing application User.")
    chunks = get_all_chunks()
    registrations = build_historical_registrations(
        chunks,
        get_document_storage_paths(),
        owner_id=owner_id,
    )
    result = backfill_document_registry(registrations)

    expected_ids = {item.document_id for item in registrations}
    actual_ids = registered_document_ids()
    missing_ids = expected_ids - actual_ids
    unexpected_ids = actual_ids - expected_ids
    if missing_ids or unexpected_ids:
        raise RuntimeError(
            "Registry verification failed: "
            f"missing={len(missing_ids)}, unexpected={len(unexpected_ids)}."
        )

    print(
        "Historical registry backfill verified: "
        f"discovered={result.discovered}, inserted={result.inserted}, "
        f"skipped_existing={result.skipped_existing}, registered={len(actual_ids)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
