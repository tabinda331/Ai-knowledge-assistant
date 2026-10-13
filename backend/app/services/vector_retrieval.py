from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.search import SearchResult
from app.services.embedding import EMBEDDING_DIMENSION, EmbeddingGenerationError, embed_texts
from app.services.retrieval_scope import ready_document_ids_statement


class VectorRetrievalError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def retrieve_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
) -> list[SearchResult]:
    if not query.strip():
        raise VectorRetrievalError("Query must not be empty", 422)
    if not 1 <= top_k <= 50:
        raise VectorRetrievalError("Candidate count must be between 1 and 50", 422)

    try:
        if document_id is not None:
            if db.get(Document, document_id) is None:
                raise VectorRetrievalError("Document not found", 404)
            embedded_count = db.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.embedding.is_not(None),
                )
            )
            if not embedded_count:
                raise VectorRetrievalError("Document has no embedded chunks", 409)

        query_vectors = embed_texts([query])
        if len(query_vectors) != 1 or len(query_vectors[0]) != EMBEDDING_DIMENSION:
            raise VectorRetrievalError("Query embedding has an unexpected dimension", 503)

        cosine_distance = DocumentChunk.embedding.cosine_distance(query_vectors[0])
        statement = (
            select(
                DocumentChunk.id.label("chunk_id"),
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.metadata_json.label("metadata"),
                (1.0 - cosine_distance).label("similarity"),
            )
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(cosine_distance.asc())
            .limit(top_k)
        )
        if document_id is not None:
            statement = statement.where(DocumentChunk.document_id == document_id)
        else:
            statement = statement.where(
                DocumentChunk.document_id.in_(ready_document_ids_statement())
            )

        rows = db.execute(statement).mappings().all()
    except VectorRetrievalError:
        raise
    except EmbeddingGenerationError:
        raise VectorRetrievalError("Query embedding is unavailable", 503) from None
    except SQLAlchemyError:
        raise VectorRetrievalError("Vector search is unavailable", 503) from None
    except Exception:
        raise VectorRetrievalError("Vector search could not be completed", 503) from None

    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            metadata=row["metadata"] or {},
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]
