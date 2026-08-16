import argparse
import json
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.core.config import settings
from app.auth.tokens import create_access_token
from app.main import app
from app.services.llm_client import get_llm_runtime_metadata
from app.services.rag_service import answer_question


QUESTION = "How many paid annual leave days do employees receive?"
CONTROLLED_CHUNK = {
    "chunk_id": "ollama-smoke-policy-1",
    "document_id": "ollama-smoke-policy",
    "filename": "ollama_smoke_policy.md",
    "position": 1,
    "chunk_index": 0,
    "page_number": None,
    "content": "Employees receive 15 paid annual leave days per calendar year.",
    "token_count": 10,
    "created_at": "2026-08-05T00:00:00+00:00",
}


def _write_controlled_index(path: Path) -> None:
    path.write_text(
        json.dumps([CONTROLLED_CHUNK], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_smoke(
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
    max_tokens: int,
    user_id: UUID,
) -> Dict[str, Any]:
    """Exercise the real Router/RAG/prompt/client/citation path against Ollama."""
    with tempfile.TemporaryDirectory(prefix="enterprise-rag-ollama-smoke-") as temp_dir:
        index_path = Path(temp_dir) / "chunks.json"
        vector_index_path = Path(temp_dir) / "vectors.json"
        _write_controlled_index(index_path)
        vector_index_path.write_text("[]", encoding="utf-8")

        def isolated_answer_question(
            question: str,
            top_k: int,
            *,
            retrieval_mode: str,
            min_score: float | None,
            user_id: UUID,
        ) -> Dict[str, Any]:
            return answer_question(
                question,
                top_k,
                retrieval_mode=retrieval_mode,
                min_score=min_score,
                user_id=user_id,
                index_path=index_path,
                vector_index_path=vector_index_path,
            )

        with ExitStack() as stack:
            stack.enter_context(patch.object(settings, "llm_provider", "ollama"))
            stack.enter_context(patch.object(settings, "llm_api_key", "ollama"))
            stack.enter_context(patch.object(settings, "llm_base_url", base_url))
            stack.enter_context(patch.object(settings, "llm_model", model))
            stack.enter_context(patch.object(settings, "llm_timeout_seconds", timeout_seconds))
            stack.enter_context(patch.object(settings, "llm_max_tokens", max_tokens))
            stack.enter_context(patch.object(settings, "llm_max_retries", 0))
            stack.enter_context(
                patch(
                    "app.routers.chat.answer_question",
                    side_effect=isolated_answer_question,
                )
            )

            runtime = get_llm_runtime_metadata(resolve_model_identity=True)
            response = TestClient(app).post(
                "/chat/",
                headers={"Authorization": f"Bearer {create_access_token(user_id).value}"},
                json={
                    "question": QUESTION,
                    "top_k": 1,
                    "retrieval_mode": "keyword",
                    "min_score": 0.0,
                },
            )

    response.raise_for_status()
    payload = response.json()
    data = payload["data"]
    if data["answer_mode"] != "llm":
        raise RuntimeError(f"Ollama smoke test used {data['answer_mode']}: {data['llm_error']}")
    if data["model"] != model:
        raise RuntimeError(f"Expected model {model}, received {data['model']}")
    if not data["citations"] or data["citations"][0]["chunk_id"] != CONTROLLED_CHUNK["chunk_id"]:
        raise RuntimeError("Ollama smoke test did not preserve the controlled citation")
    if "15" not in data["answer"]:
        raise RuntimeError("Ollama smoke answer did not contain the controlled fact")

    return {
        "artifact_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "check": "real_chat_rag_smoke",
        "passed": True,
        "runtime": runtime,
        "request": {
            "question": QUESTION,
            "top_k": 1,
            "retrieval_mode": "keyword",
            "min_score": 0.0,
        },
        "response": data,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a controlled real /chat/ smoke test against one configured Ollama model."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL user who can read document_id ollama-smoke-policy.",
    )
    args = parser.parse_args()

    result = run_smoke(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        user_id=args.user_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
