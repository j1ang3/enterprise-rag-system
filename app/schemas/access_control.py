from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class GrantDocumentAccessRequest(BaseModel):
    user_id: UUID

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"user_id": "00000000-0000-0000-0000-000000000202"}]
        }
    )


class DocumentShare(BaseModel):
    document_id: str
    user_id: UUID
    username: str
    created_at: datetime


class GrantDocumentAccessResponse(BaseModel):
    success: bool
    data: DocumentShare
    message: str


class DocumentShareListData(BaseModel):
    document_id: str
    shares: list[DocumentShare]


class DocumentShareListResponse(BaseModel):
    success: bool
    data: DocumentShareListData
    message: str


class RevokeDocumentAccessData(BaseModel):
    document_id: str
    user_id: UUID


class RevokeDocumentAccessResponse(BaseModel):
    success: bool
    data: RevokeDocumentAccessData
    message: str
