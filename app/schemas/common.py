from pydantic import BaseModel, ConfigDict


class RootResponse(BaseModel):
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"message": "Enterprise RAG System is running."}],
        }
    )


class HealthResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "ok"}]}
    )


class ErrorResponse(BaseModel):
    detail: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"detail": "A safe public error message."}],
        }
    )
