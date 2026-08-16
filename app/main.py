from fastapi import FastAPI

from app.routers import auth, chat, documents, search
from app.core.config import settings
from app.schemas.common import HealthResponse, RootResponse


OPENAPI_TAGS = [
    {
        "name": "system",
        "description": "Application discovery and process health endpoints.",
    },
    {
        "name": "auth",
        "description": "User registration, JWT login, and current-user identity.",
    },
    {
        "name": "documents",
        "description": "Authenticated ingestion, document reads, ownership, and sharing.",
    },
    {
        "name": "search",
        "description": "Permission-aware retrieval without LLM generation.",
    },
    {
        "name": "chat",
        "description": "Permission-aware retrieval-augmented answer generation.",
    },
]


app = FastAPI(
    title=settings.app_name,
    description="Enterprise-level RAG knowledge base system",
    version=settings.app_version,
    openapi_tags=OPENAPI_TAGS,
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["system"],
    summary="Show application status",
    description="Return a lightweight message confirming that the API process is running.",
)
def root():
    return {
        "message": "Enterprise RAG System is running."
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Check process health",
    description="Return process-level health for local, container, and deployment checks.",
)
def health_check():
    return {
        "status": "ok"
    }


app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(auth.router)
