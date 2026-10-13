import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DocumentLifecycle = Literal["uploaded", "processed", "ready", "failed"]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    filename: str | None
    source_type: str
    source_url: str | None
    mime_type: str | None
    status: str
    metadata: dict[str, Any] = Field(validation_alias="metadata_json")
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    lifecycle_state: DocumentLifecycle = "uploaded"


class DeleteDocumentResponse(BaseModel):
    id: uuid.UUID
    status: Literal["deleted"] = "deleted"
