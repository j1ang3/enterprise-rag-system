import argparse
import sys
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_registry import (
    assign_document_owners,
    get_document_owner,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an explicit, complete document_id=user_id ownership bootstrap."
        )
    )
    parser.add_argument(
        "--assign",
        action="append",
        required=True,
        metavar="DOCUMENT_ID=USER_ID",
        help="Repeat once for every existing document.",
    )
    return parser.parse_args()


def _mapping(assignments: list[str]) -> dict[str, UUID]:
    mapping: dict[str, UUID] = {}
    for assignment in assignments:
        document_id, separator, raw_user_id = assignment.partition("=")
        if not separator or not document_id or not raw_user_id:
            raise RuntimeError(
                "Every --assign value must use DOCUMENT_ID=USER_ID."
            )
        if document_id in mapping:
            raise RuntimeError("Duplicate document ID in ownership mapping.")
        try:
            mapping[document_id] = UUID(raw_user_id)
        except ValueError as exc:
            raise RuntimeError("Every mapped user ID must be a valid UUID.") from exc
    return mapping


def main() -> int:
    mapping = _mapping(_arguments().assign)
    result = assign_document_owners(mapping, require_complete=True)
    for document_id, owner_id in mapping.items():
        if get_document_owner(document_id) != owner_id:
            raise RuntimeError("Ownership verification failed after bootstrap commit.")
    print(
        "Document ownership bootstrap verified: "
        f"mapped={result.mapped}, assigned={result.assigned}, "
        f"unchanged={result.unchanged}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
