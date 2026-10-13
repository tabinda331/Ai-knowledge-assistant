from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.document import DeleteDocumentResponse, DocumentResponse
from app.services.document_embedding import DocumentEmbeddingError, embed_document
from app.services.document_processing import DocumentProcessingError, process_document
from app.services.document_storage import (
    DocumentStorageError,
    InvalidPDFError,
    PDFTooLargeError,
    remove_stored_pdf,
    stage_stored_pdf_removal,
    restore_staged_pdf,
    finalize_staged_pdf_removal,
    sanitize_filename,
    store_pdf,
    validate_pdf_content_type,
)
from app.services.bm25_retrieval import invalidate_bm25_index

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=500),
    db: Session = Depends(get_db_session),
) -> DocumentResponse:
    try:
        filename = sanitize_filename(file.filename)
        validate_pdf_content_type(file.content_type)
        prefix = file.file.read(1024)
        file.file.seek(0)
        if b"%PDF-" not in prefix:
            raise InvalidPDFError("The uploaded file does not have a PDF signature")
    except InvalidPDFError as error:
        raise HTTPException(status_code=415, detail=str(error)) from None
    except OSError:
        raise HTTPException(status_code=400, detail="Unable to read uploaded file") from None

    document_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    document = Document(
        id=document_id,
        title=(title or filename[:-4]).strip() or filename[:-4],
        filename=filename,
        source_type="upload",
        mime_type="application/pdf",
        status="processing",
        metadata_json={"original_filename": filename},
        created_at=now,
        updated_at=now,
    )
    db.add(document)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=503, detail="Document service unavailable") from None

    try:
        stored_pdf = store_pdf(file.file, document_id)
    except PDFTooLargeError:
        document.status = "failed"
        document.metadata_json = {**document.metadata_json, "failure_code": "file_too_large"}
        _commit_failure_state(db)
        raise HTTPException(status_code=413, detail="PDF exceeds the 25 MiB upload limit") from None
    except InvalidPDFError as error:
        document.status = "failed"
        document.metadata_json = {**document.metadata_json, "failure_code": "invalid_pdf"}
        _commit_failure_state(db)
        raise HTTPException(status_code=415, detail=str(error)) from None
    except DocumentStorageError:
        document.status = "failed"
        document.metadata_json = {**document.metadata_json, "failure_code": "storage_failed"}
        _commit_failure_state(db)
        raise HTTPException(status_code=500, detail="Unable to store uploaded document") from None

    document.status = "completed"
    document.metadata_json = {
        **document.metadata_json,
        "storage_key": stored_pdf.storage_key,
        "size_bytes": stored_pdf.size_bytes,
        "sha256": stored_pdf.sha256,
    }
    document.updated_at = datetime.now(timezone.utc)
    try:
        db.commit()
        db.refresh(document)
    except SQLAlchemyError:
        db.rollback()
        remove_stored_pdf(stored_pdf.storage_key)
        raise HTTPException(status_code=503, detail="Document service unavailable") from None
    return _document_response(db, document)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_session),
) -> list[DocumentResponse]:
    try:
        documents = (
            db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return _document_responses(db, documents)
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Document service unavailable") from None


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: uuid.UUID, db: Session = Depends(get_db_session)
) -> DocumentResponse:
    try:
        document = db.query(Document).filter_by(id=document_id).first()
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return _document_response(db, document)
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Document service unavailable") from None


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: uuid.UUID, db: Session = Depends(get_db_session)
) -> DeleteDocumentResponse:
    try:
        document = db.query(Document).filter_by(id=document_id).first()
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Document service unavailable") from None
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_key = (document.metadata_json or {}).get("storage_key")
    try:
        staged_removal = stage_stored_pdf_removal(storage_key)
    except DocumentStorageError:
        raise HTTPException(status_code=503, detail="Document could not be deleted") from None

    try:
        db.delete(document)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        try:
            restore_staged_pdf(staged_removal)
        except DocumentStorageError:
            raise HTTPException(status_code=503, detail="Document deletion could not be completed") from None
        raise HTTPException(status_code=503, detail="Document could not be deleted") from None

    invalidate_bm25_index()

    try:
        finalize_staged_pdf_removal(staged_removal)
    except DocumentStorageError:
        # The database deletion has committed; the staged file may need manual cleanup.
        raise HTTPException(status_code=503, detail="Document was deleted but file cleanup failed") from None
    return DeleteDocumentResponse(id=document_id)


@router.post("/{document_id}/process")
def process_document_endpoint(
    document_id: uuid.UUID, db: Session = Depends(get_db_session)
) -> dict[str, object]:
    try:
        document = db.query(Document).filter_by(id=document_id).first()
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Document service unavailable") from None
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        return process_document(db, document)
    except DocumentProcessingError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None


@router.post("/{document_id}/embed")
def embed_document_endpoint(
    document_id: uuid.UUID, db: Session = Depends(get_db_session)
) -> dict[str, object]:
    try:
        return embed_document(db, document_id)
    except DocumentEmbeddingError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None


def _commit_failure_state(db: Session) -> None:
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()


def _document_response(db: Session, document: Document) -> DocumentResponse:
    return _document_responses(db, [document])[0]


def _document_responses(db: Session, documents: list[Document]) -> list[DocumentResponse]:
    if not documents:
        return []
    document_ids = [document.id for document in documents]
    chunk_counts = (
        db.query(
            DocumentChunk.document_id,
            func.count(DocumentChunk.id),
            func.count(DocumentChunk.embedding),
        )
        .filter(DocumentChunk.document_id.in_(document_ids))
        .group_by(DocumentChunk.document_id)
        .all()
    )
    counts: dict[uuid.UUID, list[int]] = {document_id: [0, 0] for document_id in document_ids}
    for document_id, chunk_count, embedded_chunk_count in chunk_counts:
        counts[document_id] = [chunk_count, embedded_chunk_count]

    responses = []
    for document in documents:
        chunk_count, embedded_chunk_count = counts[document.id]
        metadata = document.metadata_json or {}
        if document.status == "failed" or metadata.get("processing_error"):
            lifecycle_state = "failed"
        elif chunk_count == 0:
            lifecycle_state = "uploaded"
        elif embedded_chunk_count == chunk_count:
            lifecycle_state = "ready"
        else:
            lifecycle_state = "processed"
        response = DocumentResponse.model_validate(document)
        responses.append(
            response.model_copy(
                update={
                    "chunk_count": chunk_count,
                    "embedded_chunk_count": embedded_chunk_count,
                    "lifecycle_state": lifecycle_state,
                }
            )
        )
    return responses
