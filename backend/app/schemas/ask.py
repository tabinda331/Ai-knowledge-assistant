from __future__ import annotations

from typing import Literal
import uuid

from pydantic import BaseModel, Field

from app.schemas.search import SearchRequest


class AskRequest(SearchRequest):
    rerank: bool = True
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    conversation_id: uuid.UUID | None = None


class AskSource(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None


class AskResponse(BaseModel):
    conversation_id: uuid.UUID
    question: str
    answer: str
    sources: list[AskSource]
    grounded: bool
    retrieval_mode: Literal["dense", "bm25", "hybrid"]
    reranked: bool
