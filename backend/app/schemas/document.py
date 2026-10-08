import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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