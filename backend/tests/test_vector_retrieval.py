from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.main import app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services import vector_retrieval
from app.services.embedding import EMBEDDING_DIMENSION, EmbeddingGenerationError


def vector(first: float, second: float = 0.0) -> list[float]:
    values = [0.0] * EMBEDDING_DIMENSION
    values[0] = first
    values[1] = second
    return values


def make_document(document_id: uuid.UUID) -> Document:
    now = datetime.now(timezone.utc)
    return Document(
        id=document_id,
        title="Search test",
        filename="search.pdf",
        source_type="upload",
        source_url=None,
        mime_type="application/pdf",
        status="completed",
        metadata_json={},
        created_at=now,
        updated_at=now,
    )


def make_chunk(
    document_id: uuid.UUID,
    index: int,
    values: list[float] | None,
) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=index,
        content=f"Evidence passage {index}",
        metadata_json={"page_start": index + 1, "page_end": index + 2},
        embedding=values,
    )


class FakeMappings:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def all(self) -> list[dict[str, Any]]:
        return self.rows


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def mappings(self) -> FakeMappings:
        return FakeMappings(self.rows)


class FakeSession:
    def __init__(self, documents: list[Document], chunks: list[DocumentChunk]):
        self.documents = {document.id: document for document in documents}
        self.chunks = chunks
        self.commits = 0
        self.executes = 0
        self.statements: list[Any] = []

    def get(self, model: type[Any], key: uuid.UUID) -> Any | None:
        assert model is Document
        return self.documents.get(key)

    def scalar(self, statement: Any) -> int:
        params = statement.compile().params.values()
        document_id = next(value for value in params if isinstance(value, uuid.UUID))
        return sum(
            chunk.document_id == document_id and chunk.embedding is not None
            for chunk in self.chunks
        )

    def execute(self, statement: Any) -> FakeResult:
        self.executes += 1
        self.statements.append(statement)
        params = list(statement.compile().params.values())
        requested_document = next(
            (value for value in params if isinstance(value, uuid.UUID)), None
        )
        query_vector = next(
            value
            for value in params
            if isinstance(value, list) and len(value) == EMBEDDING_DIMENSION
        )
        limit = statement._limit_clause.value
        rows = []
        ready_document_ids = {
            document_id
            for document_id, document in self.documents.items()
            if document.status == "completed"
            and (document_chunks := [chunk for chunk in self.chunks if chunk.document_id == document_id])
            and all(chunk.embedding is not None for chunk in document_chunks)
        }
        for chunk in self.chunks:
            if chunk.embedding is None:
                continue
            if requested_document is not None and chunk.document_id != requested_document:
                continue
            if requested_document is None and chunk.document_id not in ready_document_ids:
                continue
            numerator = sum(left * right for left, right in zip(chunk.embedding, query_vector))
            left_norm = math.sqrt(sum(value * value for value in chunk.embedding))
            right_norm = math.sqrt(sum(value * value for value in query_vector))
            similarity = numerator / (left_norm * right_norm)
            rows.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": chunk.metadata_json,
                    "similarity": similarity,
                }
            )
        rows.sort(key=lambda row: row["similarity"], reverse=True)
        return FakeResult(rows[:limit])


@pytest.fixture
def search_setup(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeSession, list[list[str]]]:
    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    documents = [make_document(first_id), make_document(second_id)]
    chunks = [
        make_chunk(first_id, 0, vector(1.0)),
        make_chunk(first_id, 1, vector(0.5, math.sqrt(0.75))),
        make_chunk(first_id, 2, vector(-1.0)),
        make_chunk(second_id, 0, vector(0.0, 1.0)),
        make_chunk(second_id, 1, None),
    ]
    session = FakeSession(documents, chunks)
    embedded_queries: list[list[str]] = []

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        embedded_queries.append(texts)
        return [vector(1.0) for _ in texts]

    monkeypatch.setattr(vector_retrieval, "embed_texts", fake_embed_texts)
    app.dependency_overrides[get_db_session] = lambda: session
    with TestClient(app) as client:
        yield client, session, embedded_queries
    app.dependency_overrides.clear()


def test_search_ranks_results_limits_top_k_and_preserves_metadata(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, _session, embedded_queries = search_setup

    response = client.post(
        "/search",
        json={"query": "  traditional RAG limitations  ", "top_k": 2},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["query"] == "traditional RAG limitations"
    assert len(body["results"]) == 2
    assert [result["similarity"] for result in body["results"]] == pytest.approx([1.0, 0.5])
    assert body["results"][0]["content"] == "Evidence passage 0"
    assert body["results"][0]["metadata"] == {"page_start": 1, "page_end": 2}
    assert embedded_queries == [["traditional RAG limitations"]]


def test_search_filters_by_document_id(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, session, _ = search_setup
    requested_id = str(session.chunks[3].document_id)

    response = client.post(
        "/search",
        json={"query": "limitations", "document_id": requested_id, "top_k": 5},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
    assert response.json()["results"][0]["document_id"] == requested_id


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"query": ""}, 422),
        ({"query": "   "}, 422),
        ({"query": "valid", "top_k": 0}, 422),
        ({"query": "valid", "top_k": 21}, 422),
        ({"query": "valid", "document_id": "not-a-uuid"}, 422),
    ],
)
def test_search_rejects_invalid_request(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
    payload: dict[str, Any],
    expected_status: int,
) -> None:
    client, _session, _ = search_setup

    response = client.post("/search", json=payload)

    assert response.status_code == expected_status


def test_search_returns_controlled_errors_for_missing_or_unembedded_documents(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, session, _ = search_setup

    missing = client.post(
        "/search",
        json={"query": "test", "document_id": str(uuid.uuid4())},
    )
    unembedded_id = uuid.uuid4()
    session.documents[unembedded_id] = make_document(unembedded_id)
    session.chunks.append(make_chunk(unembedded_id, 0, None))
    unembedded = client.post(
        "/search",
        json={"query": "test", "document_id": str(unembedded_id)},
    )

    assert missing.status_code == 404
    assert missing.json()["detail"] == "Document not found"
    assert unembedded.status_code == 409
    assert unembedded.json()["detail"] == "Document has no embedded chunks"


def test_search_without_matching_embedded_chunks_returns_empty_results(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, session, _ = search_setup
    session.chunks = [make_chunk(chunk.document_id, chunk.chunk_index, None) for chunk in session.chunks]

    response = client.post("/search", json={"query": "no vectors"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_global_dense_search_includes_only_fully_embedded_completed_documents(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, session, _ = search_setup
    non_ready_document = session.documents[next(iter(session.documents))]
    ready_document_id = next(
        document.id for document in session.documents.values() if document.id != non_ready_document.id
    )
    non_ready_document.status = "completed"
    session.chunks = [
        make_chunk(ready_document_id, 0, vector(1.0)),
        make_chunk(non_ready_document.id, 0, vector(1.0)),
        make_chunk(non_ready_document.id, 1, None),
    ]

    response = client.post("/search", json={"query": "evidence", "retrieval_mode": "dense"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["document_id"] == str(ready_document_id)
    sql = str(session.statements[0].compile()).lower()
    assert "document_chunks.document_id in (select" in sql
    assert "documents.status" in sql


def test_explicit_dense_search_keeps_partial_document_behavior(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, session, _ = search_setup
    requested_document_id = next(iter(session.documents))
    session.chunks = [
        make_chunk(requested_document_id, 0, vector(1.0)),
        make_chunk(requested_document_id, 1, None),
    ]

    response = client.post(
        "/search",
        json={"query": "evidence", "document_id": str(requested_document_id)},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_search_is_read_only_and_query_embedding_is_1024_dimensions(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
) -> None:
    client, session, _ = search_setup
    before = [
        (chunk.id, chunk.embedding, dict(chunk.metadata_json), chunk.content)
        for chunk in session.chunks
    ]

    response = client.post("/search", json={"query": "read only", "top_k": 1})

    after = [
        (chunk.id, chunk.embedding, dict(chunk.metadata_json), chunk.content)
        for chunk in session.chunks
    ]
    assert response.status_code == 200
    assert len(vector_retrieval.embed_texts(["dimension check"])[0]) == EMBEDDING_DIMENSION
    assert session.commits == 0
    assert before == after


def test_embedding_failure_returns_controlled_service_error(
    search_setup: tuple[TestClient, FakeSession, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session, _ = search_setup

    def fail_embedding(_texts: list[str]) -> list[list[float]]:
        raise EmbeddingGenerationError("private model failure")

    monkeypatch.setattr(vector_retrieval, "embed_texts", fail_embedding)

    response = client.post("/search", json={"query": "test"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Query embedding is unavailable"
