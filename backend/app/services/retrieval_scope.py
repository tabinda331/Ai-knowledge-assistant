from __future__ import annotations

from sqlalchemy import func, select

from app.models.document import Document
from app.models.document_chunk import DocumentChunk


def ready_document_ids_statement():
    """Select completed documents that have a vector for every stored chunk."""
    return (
        select(Document.id)
        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.status == "completed")
        .group_by(Document.id)
        .having(
            func.count(DocumentChunk.id) > 0,
            func.count(DocumentChunk.embedding) == func.count(DocumentChunk.id),
        )
    )
