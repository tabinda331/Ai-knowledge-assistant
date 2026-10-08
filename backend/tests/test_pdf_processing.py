from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services import document_storage
from app.services.chunking import chunk_pages
from app.services.pdf_extraction import (
    ExtractedPage,
    PDFExtractionError,
    extract_pdf_pages,
    normalize_text,
)


def make_pdf(text: str | None) -> bytes:
    if text:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    else:
        stream = b"BT /F1 12 Tf ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


class FakeQuery:
    def __init__(self, documents: list[Document]):
        self.documents = documents

    def filter_by(self, **values: Any) -> FakeQuery:
        self.documents = [
            document
            for document in self.documents
            if all(getattr(document, key) == value for key, value in values.items())
        ]
        return self

    def first(self) -> Document | None:
        return self.documents[0] if self.documents else None


class FakeSession:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.chunks: list[DocumentChunk] = []

    def query(self, _model: type[Document]) -> FakeQuery:
        return FakeQuery(self.documents.copy())

    def execute(self, _statement: Any) -> None:
        self.chunks.clear()

    def add(self, chunk: DocumentChunk) -> None:
        self.chunks.append(chunk)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


@pytest.fixture
def processing_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = FakeSession()
    monkeypatch.setattr(document_storage, "get_document_storage_root", lambda: tmp_path)
    app.dependency_overrides[get_db_session] = lambda: session
    with TestClient(app) as client:
        yield client, session, tmp_path
    app.dependency_overrides.clear()


def add_stored_document(session: FakeSession, storage_root: Path, content: bytes) -> Document:
    document_id = uuid.uuid4()
    relative_path = Path("documents") / str(document_id)[:2] / f"{document_id}.pdf"
    stored_path = storage_root / relative_path
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)
    document = Document(
        id=document_id,
        title="Test source",
        filename="source.pdf",
        source_type="upload",
        source_url=None,
        mime_type="application/pdf",
        status="completed",
        metadata_json={"storage_key": relative_path.as_posix()},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.documents.append(document)
    return document


def test_extracts_text_from_normal_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "text.pdf"
    pdf_path.write_bytes(make_pdf("A reliable text extraction test."))

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "A reliable text extraction test." in pages[0].text


@pytest.mark.parametrize("pdf_bytes", [make_pdf(None), b"not a valid PDF"])
def test_empty_or_unreadable_pdf_raises_controlled_error(
    tmp_path: Path, pdf_bytes: bytes
) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(pdf_bytes)

    with pytest.raises(PDFExtractionError):
        extract_pdf_pages(pdf_path)


def test_normalization_collapses_whitespace_and_preserves_paragraphs() -> None:
    raw_text = "  First\tline   continues.  \n\tSecond line.\n\n\n Third   paragraph.  "

    assert normalize_text(raw_text) == "First line continues. Second line.\n\nThird paragraph."


def test_chunk_size_overlap_and_sequential_indexes_are_deterministic() -> None:
    pages = [ExtractedPage(1, "a" * 2200)]

    first_run = chunk_pages(pages, chunk_size=1000, overlap=150)
    second_run = chunk_pages(pages, chunk_size=1000, overlap=150)

    assert first_run == second_run
    assert [chunk.chunk_index for chunk in first_run] == list(range(len(first_run)))
    assert [len(chunk.content) for chunk in first_run] == [1000, 1000, 500]
    assert first_run[0].content[-150:] == first_run[1].content[:150]
    assert first_run[1].content[-150:] == first_run[2].content[:150]


def test_chunk_page_ranges_follow_source_pages() -> None:
    chunks = chunk_pages(
        [ExtractedPage(1, "one " * 40), ExtractedPage(2, "two " * 40)],
        chunk_size=120,
        overlap=20,
    )

    assert chunks
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)


def test_process_success_persists_related_sequential_chunks(
    processing_client: tuple[TestClient, FakeSession, Path]
) -> None:
    client, session, storage_root = processing_client
    document = add_stored_document(session, storage_root, make_pdf("First paragraph.\n\nSecond paragraph."))

    response = client.post(f"/documents/{document.id}/process")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["pages_processed"] == 1
    assert result["chunks_created"] >= 1
    assert document.status == "completed"
    assert [chunk.chunk_index for chunk in session.chunks] == list(range(len(session.chunks)))
    assert all(chunk.document_id == document.id for chunk in session.chunks)
    assert session.chunks[0].metadata_json["page_start"] == 1


def test_processing_failure_sets_failed_status_and_safe_description(
    processing_client: tuple[TestClient, FakeSession, Path]
) -> None:
    client, session, storage_root = processing_client
    document = add_stored_document(session, storage_root, make_pdf(None))

    response = client.post(f"/documents/{document.id}/process")

    assert response.status_code == 422
    assert response.json()["detail"] == "No usable text could be extracted from the PDF"
    assert document.status == "failed"
    assert document.metadata_json["processing_error"] == response.json()["detail"]


def test_missing_document_returns_404(
    processing_client: tuple[TestClient, FakeSession, Path]
) -> None:
    client, _, _ = processing_client

    response = client.post(f"/documents/{uuid.uuid4()}/process")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_missing_stored_file_marks_document_failed(
    processing_client: tuple[TestClient, FakeSession, Path]
) -> None:
    client, session, _ = processing_client
    document = Document(
        id=uuid.uuid4(),
        title="Missing file",
        filename="missing.pdf",
        source_type="upload",
        source_url=None,
        mime_type="application/pdf",
        status="completed",
        metadata_json={"storage_key": "documents/aa/missing.pdf"},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.documents.append(document)

    response = client.post(f"/documents/{document.id}/process")

    assert response.status_code == 404
    assert response.json()["detail"] == "Stored PDF file is missing"
    assert document.status == "failed"
    assert document.metadata_json["processing_error"] == "Stored PDF file is missing"


def test_repeat_processing_replaces_instead_of_duplicating_chunks(
    processing_client: tuple[TestClient, FakeSession, Path]
) -> None:
    client, session, storage_root = processing_client
    document = add_stored_document(session, storage_root, make_pdf("Repeat processing safely."))

    first_response = client.post(f"/documents/{document.id}/process")
    first_chunks = [(chunk.chunk_index, chunk.content) for chunk in session.chunks]
    second_response = client.post(f"/documents/{document.id}/process")

    assert first_response.status_code == second_response.status_code == 200
    assert [(chunk.chunk_index, chunk.content) for chunk in session.chunks] == first_chunks
    assert len(session.chunks) == first_response.json()["chunks_created"]