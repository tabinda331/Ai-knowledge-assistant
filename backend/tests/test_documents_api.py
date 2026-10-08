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


class FakeSession:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add(self, document: Document) -> None:
        self.documents.append(document)

    def commit(self) -> None:
        pass

    def refresh(self, _document: Document) -> None:
        pass

    def rollback(self) -> None:
        pass

    def query(self, _model: type[Document]) -> FakeQuery:
        return FakeQuery(self.documents.copy())


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