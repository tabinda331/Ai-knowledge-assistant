from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
    candidate_k: int = Field(default=20, ge=1, le=50)
    document_id: uuid.UUID | None = None
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "dense"

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_candidate_count(self) -> SearchRequest:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    similarity: float | None = Field(
        default=None,
        description="Cosine similarity; higher scores indicate closer vectors, not probability",
    )
    bm25_score: float | None = None
    hybrid_score: float | None = Field(
        default=None,
        description="Reciprocal Rank Fusion score; higher values indicate a stronger combined rank",
    )
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance score used to order reranked candidates",
    )


class SearchResponse(BaseModel):
    query: str
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "dense"
    reranked: bool = False
    results: list[SearchResult]