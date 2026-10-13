from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import documents as documents_route
from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services import document_storage


class FakeQuery:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.offset_value = 0
        self.limit_value: int | None = None

    def order_by(self, *_expressions: Any) -> FakeQuery:
        self.documents.sort(key=lambda document: document.created_at, reverse=True)
        return self

    def offset(self, value: int) -> FakeQuery:
        self.offset_value = value
        return self

    def limit(self, value: int) -> FakeQuery:
        self.limit_value = value
        return self

    def all(self) -> list[Document]:
        documents = self.documents[self.offset_value :]
        return documents if self.limit_value is None else documents[: self.limit_value]

    def filter_by(self, **values: Any) -> FakeQuery:
        self.documents = [
            document
            for document in self.documents
            if all(getattr(document, key) == value for key, value in values.items())
        ]
        return self

    def first(self) -> Document | None:
        return self.documents[0] if self.documents else None

    def filter(self, *_conditions: Any) -> FakeQuery:
        return self

    def group_by(self, *_expressions: Any) -> FakeQuery:
        return self


class FakeSession:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.chunks: list[DocumentChunk] = []
        self.pending_delete: Document | None = None
        self.fail_next_commit = False

    def add(self, document: Document) -> None:
        self.documents.append(document)

    def commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            from sqlalchemy.exc import SQLAlchemyError

            raise SQLAlchemyError("simulated commit failure")
        if self.pending_delete is not None:
            document = self.pending_delete
            self.documents = [item for item in self.documents if item.id != document.id]
            self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document.id]
            self.pending_delete = None

    def delete(self, document: Document) -> None:
        self.pending_delete = document

    def refresh(self, _document: Document) -> None:
        pass

    def rollback(self) -> None:
        self.pending_delete = None

    def query(self, *entities: Any) -> FakeQuery:
        model = entities[0]
        if model is Document:
            return FakeQuery(self.documents.copy())
        if model is DocumentChunk:
            return FakeQuery(self.chunks.copy())
        if model is DocumentChunk.document_id:
            # The lifecycle query projects document_id plus two aggregate counts.
            grouped: dict[uuid.UUID, list[int]] = {}
            for chunk in self.chunks:
                counts = grouped.setdefault(chunk.document_id, [0, 0])
                counts[0] += 1
                if chunk.embedding is not None:
                    counts[1] += 1
            return FakeQuery(
                [(document_id, values[0], values[1]) for document_id, values in grouped.items()]
            )
        raise AssertionError(f"Unexpected model queried: {model}")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_session = FakeSession()
    monkeypatch.setattr(document_storage, "get_document_storage_root", lambda: tmp_path)
    app.dependency_overrides[get_db_session] = lambda: fake_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_upload_pdf_stores_file_and_metadata_safely(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/documents",
        files={"file": ("../../private/knowledge.pdf", b"%PDF-1.7\nexample", "application/pdf")},
        data={"title": "Knowledge source"},
    )

    assert response.status_code == 201
    document = response.json()
    assert document["title"] == "Knowledge source"
    assert document["filename"] == "knowledge.pdf"
    assert document["status"] == "completed"
    assert document["metadata"]["original_filename"] == "knowledge.pdf"
    storage_key = Path(document["metadata"]["storage_key"])
    assert not storage_key.is_absolute()
    assert ".." not in storage_key.parts
    stored_path = tmp_path / storage_key
    assert stored_path.is_file()
    assert stored_path.read_bytes() == b"%PDF-1.7\nexample"
    assert document["metadata"]["size_bytes"] == len(b"%PDF-1.7\nexample")
    assert len(document["metadata"]["sha256"]) == 64
    assert document["chunk_count"] == 0
    assert document["embedded_chunk_count"] == 0
    assert document["lifecycle_state"] == "uploaded"


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_status"),
    [
        ("notes.txt", b"%PDF-1.7", "application/pdf", 415),
        ("notes.pdf", b"not a PDF", "application/pdf", 415),
        ("notes.pdf", b"%PDF-1.7", "text/plain", 415),
    ],
)
def test_rejects_non_pdf_uploads(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str,
    expected_status: int,
) -> None:
    response = client.post(
        "/documents",
        files={"file": (filename, content, content_type)},
    )

    assert response.status_code == expected_status


def test_oversized_pdf_is_marked_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(document_storage, "MAX_PDF_SIZE_BYTES", 16)
    response = client.post(
        "/documents",
        files={"file": ("large.pdf", b"%PDF-1.7" + b"x" * 32, "application/pdf")},
    )

    assert response.status_code == 413
    fake_session = app.dependency_overrides[get_db_session]()
    assert fake_session.documents[0].status == "failed"
    assert fake_session.documents[0].metadata_json["failure_code"] == "file_too_large"


def test_lists_and_retrieves_document_metadata(client: TestClient) -> None:
    upload_response = client.post(
        "/documents",
        files={"file": ("reference.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    document_id = upload_response.json()["id"]

    list_response = client.get("/documents", params={"limit": 10, "offset": 0})
    detail_response = client.get(f"/documents/{document_id}")
    missing_response = client.get(f"/documents/{uuid.uuid4()}")

    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == document_id
    assert detail_response.json()["status"] == "completed"
    assert missing_response.status_code == 404


def test_document_readiness_counts_unembedded_partial_and_all_chunks(client: TestClient) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("readiness.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    document_id = uuid.UUID(upload.json()["id"])
    session = app.dependency_overrides[get_db_session]()
    session.chunks = [
        DocumentChunk(document_id=document_id, chunk_index=index, content=f"chunk {index}",
                      metadata_json={}, embedding=([0.0] * 1024 if index == 1 else None))
        for index in range(3)
    ]

    processed = client.get(f"/documents/{document_id}").json()
    assert (processed["chunk_count"], processed["embedded_chunk_count"]) == (3, 1)
    assert processed["lifecycle_state"] == "processed"

    session.chunks[1].embedding = None
    unembedded = client.get(f"/documents/{document_id}").json()
    assert (unembedded["chunk_count"], unembedded["embedded_chunk_count"]) == (3, 0)
    assert unembedded["lifecycle_state"] == "processed"

    for chunk in session.chunks:
        chunk.embedding = [0.0] * 1024
    ready = client.get(f"/documents/{document_id}").json()
    assert (ready["chunk_count"], ready["embedded_chunk_count"]) == (3, 3)
    assert ready["lifecycle_state"] == "ready"


def test_processing_failure_is_exposed_as_failed_lifecycle(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("failed.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    document_id = uuid.UUID(response.json()["id"])
    session = app.dependency_overrides[get_db_session]()
    session.documents[0].metadata_json["processing_error"] = "No usable text"

    result = client.get(f"/documents/{document_id}").json()

    assert result["lifecycle_state"] == "failed"


def test_delete_document_removes_database_rows_and_stored_pdf(
    client: TestClient, tmp_path: Path
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("delete.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    data = upload.json()
    document_id = uuid.UUID(data["id"])
    stored_path = tmp_path / data["metadata"]["storage_key"]
    session = app.dependency_overrides[get_db_session]()
    session.chunks.append(DocumentChunk(document_id=document_id, chunk_index=0, content="chunk",
                                       metadata_json={}, embedding=[0.0] * 1024))

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {"id": str(document_id), "status": "deleted"}
    assert not stored_path.exists()
    assert not any(document.id == document_id for document in session.documents)
    assert not any(chunk.document_id == document_id for chunk in session.chunks)


def test_successful_delete_invalidates_bm25_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("invalidate.pdf", b"%PDF-1.7\\ncontent", "application/pdf")},
    )
    document_id = upload.json()["id"]
    invalidations: list[bool] = []
    monkeypatch.setattr(documents_route, "invalidate_bm25_index", lambda: invalidations.append(True))

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert invalidations == [True]


def test_delete_missing_document_returns_404(client: TestClient) -> None:
    response = client.delete(f"/documents/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_delete_commit_failure_restores_pdf_and_keeps_document(
    client: TestClient, tmp_path: Path
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("rollback.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    data = upload.json()
    document_id = uuid.UUID(data["id"])
    stored_path = tmp_path / data["metadata"]["storage_key"]
    session = app.dependency_overrides[get_db_session]()
    session.fail_next_commit = True

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 503
    assert stored_path.is_file()
    assert any(document.id == document_id for document in session.documents)


def test_delete_staging_failure_does_not_delete_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("stage.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    document_id = uuid.UUID(upload.json()["id"])
    session = app.dependency_overrides[get_db_session]()
    monkeypatch.setattr(documents_route, "stage_stored_pdf_removal",
                        lambda _key: (_ for _ in ()).throw(document_storage.DocumentStorageError("internal")))

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 503
    assert any(document.id == document_id for document in session.documents)


def test_delete_cleanup_failure_is_not_reported_as_success(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("cleanup.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )
    data = upload.json()
    document_id = uuid.UUID(data["id"])
    original_path = tmp_path / data["metadata"]["storage_key"]
    session = app.dependency_overrides[get_db_session]()
    staged = document_storage.stage_stored_pdf_removal
    staged_removals = []

    def capture_staged_removal(key: str | None):
        removal = staged(key)
        staged_removals.append(removal)
        return removal

    monkeypatch.setattr(documents_route, "stage_stored_pdf_removal", capture_staged_removal)
    monkeypatch.setattr(
        documents_route,
        "finalize_staged_pdf_removal",
        lambda _removal: (_ for _ in ()).throw(document_storage.DocumentStorageError("internal")),
    )

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 503
    assert "internal" not in response.text
    assert not original_path.exists()
    assert staged_removals[0] is not None
    assert staged_removals[0].staged_path.is_file()
    assert not any(document.id == document_id for document in session.documents)


def test_delete_cors_preflight_allows_delete() -> None:
    with TestClient(app) as cors_client:
        response = cors_client.options(
            "/documents/00000000-0000-0000-0000-000000000001",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "DELETE",
            },
        )

    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]
