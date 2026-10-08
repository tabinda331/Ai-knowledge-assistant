from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_embedding_model_name
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.embedding import (
    EMBEDDING_DIMENSION,
    EmbeddingGenerationError,
    embed_texts,
)


class DocumentEmbeddingError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def embed_document(db: Session, document_id: uuid.UUID) -> dict[str, Any]:
    try:
        document = db.query(Document).filter_by(id=document_id).first()
        if document is None:
            raise DocumentEmbeddingError("Document not found", 404)

        chunks = (
            db.query(DocumentChunk)
            .filter_by(document_id=document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        if not chunks:
            raise DocumentEmbeddingError("Document has no processed chunks", 409)

        pending_chunks = [chunk for chunk in chunks if chunk.embedding is None]
        if pending_chunks:
            vectors = embed_texts([chunk.content for chunk in pending_chunks])
            for chunk, vector in zip(pending_chunks, vectors, strict=True):
                chunk.embedding = vector
            db.commit()
    except DocumentEmbeddingError:
        raise
    except EmbeddingGenerationError:
        db.rollback()
        raise DocumentEmbeddingError("Embedding generation is unavailable", 503) from None
    except SQLAlchemyError:
        db.rollback()
        raise DocumentEmbeddingError("Document embeddings could not be saved", 503) from None
    except Exception:
        db.rollback()
        raise DocumentEmbeddingError("Document embeddings could not be generated", 503) from None

    return {
        "document_id": str(document_id),
        "status": "completed",
        "chunks_embedded": sum(chunk.embedding is not None for chunk in chunks),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "model": get_embedding_model_name(),
    }