from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services import embedding


class FakeQuery:
    def __init__(self, rows: list[Any]):
        self.rows = rows

    def filter_by(self, **values: Any) -> FakeQuery:
        self.rows = [
            row for row in self.rows
            if all(getattr(row, key) == value for key, value in values.items())
        ]
        return self

    def order_by(self, *_expressions: Any) -> FakeQuery:
        self.rows.sort(key=lambda row: row.chunk_index)
        return self

    def first(self) -> Any | None:
        return self.rows[0] if self.rows else None

    def all(self) -> list[Any]:
        return self.rows


class FakeSession:
    def __init__(self, documents: list[Document], chunks: list[DocumentChunk]):
        self.documents = documents
        self.chunks = chunks
        self.commits = 0
        self.rollbacks = 0

    def query(self, model: type[Any]) -> FakeQuery:
        if model is Document:
            return FakeQuery(self.documents.copy())
        if model is DocumentChunk:
            return FakeQuery(self.chunks.copy())
        raise AssertionError(f"Unexpected model queried: {model}")

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def make_document(document_id: uuid.UUID) -> Document:
    now = datetime.now(timezone.utc)
    return Document(
        id=document_id,
        title="Embedding test",
        filename="embedding.pdf",
        source_type="upload",
        source_url=None,
        mime_type="application/pdf",
        status="completed",
        metadata_json={},
        created_at=now,
        updated_at=now,
    )


def make_chunk(document_id: uuid.UUID, index: int, vector: list[float] | None = None) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=index,
        content=f"Chunk {index}",
        metadata_json={},
        embedding=vector,
    )


def fake_vector_encoder(texts: list[str]) -> list[list[float]]:
    return [[float(index)] * embedding.EMBEDDING_DIMENSION for index, _ in enumerate(texts)]


def test_embedding_service_returns_1024_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def encode(self, texts: list[str], **_kwargs: Any) -> Any:
            class Encoded:
                def tolist(self) -> list[list[float]]:
                    return fake_vector_encoder(texts)

            return Encoded()

    monkeypatch.setattr(embedding, "_load_model", lambda _name: FakeModel())

    vectors = embedding.embed_texts(["one", "two"])

    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)


def test_embedding_service_handles_empty_input_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(_name: str) -> Any:
        raise AssertionError("The model should not load for empty input")

    monkeypatch.setattr(embedding, "_load_model", unexpected_load)

    assert embedding.embed_texts([]) == []


@pytest.fixture
def embedding_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeSession]:
    document_id = uuid.uuid4()
    document = make_document(document_id)
    chunks = [make_chunk(document_id, index) for index in range(3)]
    session = FakeSession([document], chunks)
    monkeypatch.setattr(embedding, "embed_texts", fake_vector_encoder)
    app.dependency_overrides[get_db_session] = lambda: session
    with TestClient(app) as client:
        yield client, session
    app.dependency_overrides.clear()


def test_document_chunks_are_embedded_and_persisted(
    embedding_client: tuple[TestClient, FakeSession],
) -> None:
    client, session = embedding_client
    document_id = session.documents[0].id

    response = client.post(f"/documents/{document_id}/embed")

    assert response.status_code == 200
    assert response.json()["chunks_embedded"] == 3
    assert response.json()["embedding_dimension"] == 1024
    assert response.json()["model"] == "BAAI/bge-m3"
    assert all(chunk.embedding is not None for chunk in session.chunks)
    assert all(len(chunk.embedding or []) == 1024 for chunk in session.chunks)
    assert session.commits == 1


def test_rerunning_embedding_skips_existing_vectors(
    embedding_client: tuple[TestClient, FakeSession],
) -> None:
    client, session = embedding_client
    document_id = session.documents[0].id

    first = client.post(f"/documents/{document_id}/embed")
    vector_snapshot = [chunk.embedding for chunk in session.chunks]
    second = client.post(f"/documents/{document_id}/embed")

    assert first.status_code == second.status_code == 200
    assert len(session.chunks) == 3
    assert [chunk.embedding for chunk in session.chunks] == vector_snapshot
    assert session.commits == 1


def test_missing_document_returns_404(embedding_client: tuple[TestClient, FakeSession]) -> None:
    client, _ = embedding_client

    response = client.post(f"/documents/{uuid.uuid4()}/embed")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_document_without_chunks_returns_409(embedding_client: tuple[TestClient, FakeSession]) -> None:
    client, session = embedding_client
    session.chunks.clear()

    response = client.post(f"/documents/{session.documents[0].id}/embed")

    assert response.status_code == 409
    assert response.json()["detail"] == "Document has no processed chunks"