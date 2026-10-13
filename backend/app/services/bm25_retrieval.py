from __future__ import annotations

import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.search import SearchResult
from app.services.retrieval_scope import ready_document_ids_statement

BM25_CACHE_TTL_SECONDS = 60
BM25_CACHE_MAX_SCOPES = 32
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class BM25RetrievalError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class BM25Chunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    metadata: dict[str, Any]


@dataclass
class _CachedIndex:
    expires_at: float
    chunks: list[BM25Chunk]
    tokenized_chunks: list[list[str]]
    model: BM25Okapi | None


_index_cache: OrderedDict[uuid.UUID | None, _CachedIndex] = OrderedDict()
_cache_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def _load_chunks(db: Session, document_id: uuid.UUID | None) -> list[BM25Chunk]:
    statement = select(
        DocumentChunk.id,
        DocumentChunk.document_id,
        DocumentChunk.chunk_index,
        DocumentChunk.content,
        DocumentChunk.metadata_json,
    ).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    else:
        statement = statement.where(
            DocumentChunk.document_id.in_(ready_document_ids_statement())
        )

    rows = db.execute(statement).mappings().all()
    return [
        BM25Chunk(
            chunk_id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            metadata=row["metadata_json"] or {},
        )
        for row in rows
    ]


def _get_index(db: Session, document_id: uuid.UUID | None) -> _CachedIndex:
    now = time.monotonic()
    with _cache_lock:
        cached = _index_cache.get(document_id)
        if cached is not None and cached.expires_at > now:
            _index_cache.move_to_end(document_id)
            return cached

    chunks = _load_chunks(db, document_id)
    tokenized_chunks = [tokenize(chunk.content) for chunk in chunks]
    indexed = [
        (chunk, tokens)
        for chunk, tokens in zip(chunks, tokenized_chunks, strict=True)
        if tokens
    ]
    model = BM25Okapi([tokens for _, tokens in indexed]) if indexed else None
    cached_index = _CachedIndex(
        expires_at=now + BM25_CACHE_TTL_SECONDS,
        chunks=[chunk for chunk, _ in indexed],
        tokenized_chunks=[tokens for _, tokens in indexed],
        model=model,
    )

    with _cache_lock:
        existing = _index_cache.get(document_id)
        if existing is not None and existing.expires_at > time.monotonic():
            _index_cache.move_to_end(document_id)
            return existing
        _index_cache[document_id] = cached_index
        _index_cache.move_to_end(document_id)
        while len(_index_cache) > BM25_CACHE_MAX_SCOPES:
            _index_cache.popitem(last=False)
    return cached_index


def invalidate_bm25_index() -> None:
    with _cache_lock:
        _index_cache.clear()


def retrieve_bm25_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
) -> list[SearchResult]:
    if not query.strip():
        raise BM25RetrievalError("Query must not be empty", 422)
    if not 1 <= top_k <= 50:
        raise BM25RetrievalError("Candidate count must be between 1 and 50", 422)

    try:
        if document_id is not None and db.get(Document, document_id) is None:
            raise BM25RetrievalError("Document not found", 404)

        index = _get_index(db, document_id)
        if document_id is not None and not index.chunks:
            raise BM25RetrievalError("Document has no processed chunks", 409)
        if index.model is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = index.model.get_scores(query_tokens)
        query_terms = set(query_tokens)
        ranked = sorted(
            (
                (index.chunks[position], float(score))
                for position, score in enumerate(scores)
                if query_terms.intersection(index.tokenized_chunks[position])
            ),
            key=lambda item: (-item[1], item[0].chunk_index, str(item[0].document_id)),
        )[:top_k]
    except BM25RetrievalError:
        raise
    except SQLAlchemyError:
        raise BM25RetrievalError("Keyword search is unavailable", 503) from None
    except Exception:
        raise BM25RetrievalError("Keyword search could not be completed", 503) from None

    return [
        SearchResult(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            metadata=chunk.metadata,
            bm25_score=score,
            bm25_rank=rank,
        )
        for rank, (chunk, score) in enumerate(ranked, start=1)
    ]
