from pydantic import BaseModel


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    created_at: str | None = None


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    position: int
    chunk_index: int | None = None
    page_number: int | None = None
    content: str
    token_count: int | None = None
    created_at: str | None = None


class UploadDocumentData(BaseModel):
    document_id: str
    filename: str
    content_type: str | None = None
    saved_path: str
    text_path: str
    chunk_count: int
    preview: str


class UploadDocumentResponse(BaseModel):
    success: bool
    data: UploadDocumentData
    message: str


class DocumentListData(BaseModel):
    documents: list[DocumentSummary]


class DocumentListResponse(BaseModel):
    success: bool
    data: DocumentListData
    message: str


class DocumentPreviewData(BaseModel):
    document_id: str
    preview: str


class DocumentPreviewResponse(BaseModel):
    success: bool
    data: DocumentPreviewData
    message: str


class DocumentChunksData(BaseModel):
    document_id: str
    chunk_count: int
    chunks: list[DocumentChunk]


class DocumentChunksResponse(BaseModel):
    success: bool
    data: DocumentChunksData
    message: str
