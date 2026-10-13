from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.bm25_retrieval import invalidate_bm25_index
from app.services.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from app.services.document_storage import resolve_stored_pdf_path
from app.services.pdf_extraction import PDFExtractionError, extract_pdf_pages


class DocumentProcessingError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _update_failed(db: Session, document: Document, error_description: str) -> None:
    document.status = "failed"
    document.updated_at = datetime.now(timezone.utc)
    document.metadata_json = {
        **(document.metadata_json or {}),
        "processing_error": error_description,
    }
    try:
        db.commit()
        invalidate_bm25_index()
    except SQLAlchemyError:
        db.rollback()
        raise DocumentProcessingError("Document processing could not be saved", 503) from None


def process_document(db: Session, document: Document) -> dict[str, Any]:
    metadata = document.metadata_json or {}
    storage_key = metadata.get("storage_key")
    if not isinstance(storage_key, str) or not storage_key:
        _update_failed(db, document, "Stored PDF file is missing")
        raise DocumentProcessingError("Stored PDF file is missing", 404)

    pdf_path = resolve_stored_pdf_path(storage_key)
    if pdf_path is None or not pdf_path.is_file():
        _update_failed(db, document, "Stored PDF file is missing")
        raise DocumentProcessingError("Stored PDF file is missing", 404)

    document.status = "processing"
    document.updated_at = datetime.now(timezone.utc)
    metadata.pop("processing_error", None)
    document.metadata_json = metadata
    try:
        db.commit()
        invalidate_bm25_index()
    except SQLAlchemyError:
        db.rollback()
        raise DocumentProcessingError("Document processing could not be started", 503) from None

    try:
        pages = extract_pdf_pages(pdf_path)
        chunks = chunk_pages(pages)
    except PDFExtractionError as error:
        _update_failed(db, document, str(error))
        raise DocumentProcessingError(str(error), 422) from None
    except Exception:
        _update_failed(db, document, "PDF processing failed")
        raise DocumentProcessingError("PDF processing failed", 422) from None

    try:
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        for chunk in chunks:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    metadata_json={
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    },
                )
            )

        characters_extracted = sum(len(page.text) for page in pages)
        document.status = "completed"
        document.updated_at = datetime.now(timezone.utc)
        document.metadata_json = {
            **metadata,
            "processing_stats": {
                "pages_processed": len(pages),
                "characters_extracted": characters_extracted,
                "chunks_created": len(chunks),
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            },
        }
        db.commit()
        invalidate_bm25_index()
    except SQLAlchemyError:
        db.rollback()
        _update_failed(db, document, "Chunk storage failed")
        raise DocumentProcessingError("Document processing could not be saved", 503) from None

    return {
        "document_id": str(document.id),
        "status": document.status,
        "pages_processed": len(pages),
        "characters_extracted": characters_extracted,
        "chunks_created": len(chunks),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }
