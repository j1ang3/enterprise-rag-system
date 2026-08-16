import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.storage_paths import get_document_storage_paths
from app.services.vector_store import rebuild_vector_index


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the local vector index from chunk JSON.")
    storage_paths = get_document_storage_paths()
    parser.add_argument("--chunks", type=Path, default=storage_paths.chunks_file)
    parser.add_argument("--output", type=Path, default=storage_paths.vectors_file)
    args = parser.parse_args()

    chunks_path = args.chunks if args.chunks.is_absolute() else PROJECT_ROOT / args.chunks
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    chunks = load_chunks(chunks_path)
    entries = rebuild_vector_index(chunks, output_path)

    print(f"Loaded {len(chunks)} chunk(s) from {chunks_path}")
    print(f"Wrote {len(entries)} vector entry(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
