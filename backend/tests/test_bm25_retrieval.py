from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from app.models.document import Document
from app.schemas.search import SearchResult
from app.services import bm25_retrieval
from app.services.bm25_retrieval import BM25Chunk, BM25RetrievalError


class FakeSession:
    def __init__(self, document_ids: set[uuid.UUID]):
        self.document_ids = document_ids
        self.commits = 0

    def get(self, model: type[Any], key: uuid.UUID) -> Document | None:
        assert model is Document
        if key not in self.document_ids:
            return None
        now = datetime.now(timezone.utc)
        return Document(
            id=key,
            title="BM25 test",
            filename="bm25.pdf",
            source_type="upload",
            source_url=None,
            mime_type="application/pdf",
            status="completed",
            metadata_json={},
            created_at=now,
            updated_at=now,
        )


def make_chunk(document_id: uuid.UUID, index: int, content: str) -> BM25Chunk:
    return BM25Chunk(
        chunk_id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=index,
        content=content,
        metadata={"page_start": index + 1, "page_end": index + 1},
    )


@pytest.fixture(autouse=True)
def clear_bm25_cache() -> None:
    bm25_retrieval.invalidate_bm25_index()


def test_tokenizer_normalizes_case_and_splits_punctuation() -> None:
    assert bm25_retrieval.tokenize("BGE-M3, RAG's v2.0") == [
        "bge", "m3", "rag", "s", "v2", "0"
    ]


def test_bm25_finds_lexical_matches_limits_top_k_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.uuid4()
    chunks = [
        make_chunk(document_id, 0, "Agentic RAG uses a planner and retrieval tools."),
        make_chunk(document_id, 1, "Traditional retrieval has static workflows."),
        make_chunk(document_id, 2, "Agentic RAG can use multi-step retrieval."),
    ]
    monkeypatch.setattr(bm25_retrieval, "_load_chunks", lambda _db, _scope: chunks)

    results = bm25_retrieval.retrieve_bm25_chunks(
        FakeSession({document_id}), "agentic retrieval", top_k=1
    )

    assert len(results) == 1
    assert results[0].chunk_index in {0, 2}
    assert results[0].bm25_score is not None and results[0].bm25_score > 0
    assert results[0].bm25_rank == 1
    assert results[0].metadata["page_start"] == results[0].chunk_index + 1


def test_bm25_applies_document_filter_before_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_id = uuid.uuid4()
    other_id = uuid.uuid4()
    requested_chunks = [
        make_chunk(requested_id, 0, "Distinctive identifier ZXQ-481."),
        make_chunk(requested_id, 1, "A separate discussion of retrieval architecture."),
        make_chunk(requested_id, 2, "Another unrelated discussion about evaluation."),
    ]
    loaded_scopes: list[uuid.UUID | None] = []

    def load_chunks(_db: Any, scope: uuid.UUID | None) -> list[BM25Chunk]:
        loaded_scopes.append(scope)
        return requested_chunks if scope == requested_id else [
            *requested_chunks,
            make_chunk(other_id, 1, "Distinctive identifier ZXQ-481."),
        ]

    monkeypatch.setattr(bm25_retrieval, "_load_chunks", load_chunks)

    results = bm25_retrieval.retrieve_bm25_chunks(
        FakeSession({requested_id}), "ZXQ-481", document_id=requested_id
    )

    assert loaded_scopes == [requested_id]
    assert results and all(result.document_id == requested_id for result in results)


def test_bm25_index_is_cached_and_can_be_invalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.uuid4()
    chunks = [make_chunk(document_id, 0, "agentic retrieval")]
    load_count = 0

    def load_chunks(_db: Any, _scope: uuid.UUID | None) -> list[BM25Chunk]:
        nonlocal load_count
        load_count += 1
        return chunks

    monkeypatch.setattr(bm25_retrieval, "_load_chunks", load_chunks)
    session = FakeSession({document_id})

    bm25_retrieval.retrieve_bm25_chunks(session, "agentic")
    bm25_retrieval.retrieve_bm25_chunks(session, "retrieval")
    assert load_count == 1

    bm25_retrieval.invalidate_bm25_index()
    bm25_retrieval.retrieve_bm25_chunks(session, "agentic")
    assert load_count == 2


def test_bm25_keeps_exact_match_when_small_corpus_idf_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.uuid4()
    chunk = make_chunk(document_id, 0, "Exact identifier ZXQ-481")
    monkeypatch.setattr(bm25_retrieval, "_load_chunks", lambda _db, _scope: [chunk])

    results = bm25_retrieval.retrieve_bm25_chunks(
        FakeSession({document_id}), "ZXQ-481", document_id=document_id
    )

    assert len(results) == 1
    assert results[0].chunk_index == 0


def test_bm25_empty_query_missing_document_and_empty_document_are_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.uuid4()
    session = FakeSession({document_id})
    monkeypatch.setattr(bm25_retrieval, "_load_chunks", lambda _db, _scope: [])

    with pytest.raises(BM25RetrievalError, match="Query must not be empty"):
        bm25_retrieval.retrieve_bm25_chunks(session, "  ")
    with pytest.raises(BM25RetrievalError, match="Document not found") as missing:
        bm25_retrieval.retrieve_bm25_chunks(
            FakeSession(set()), "term", document_id=document_id
        )
    assert missing.value.status_code == 404
    with pytest.raises(BM25RetrievalError, match="no processed chunks") as empty:
        bm25_retrieval.retrieve_bm25_chunks(session, "term", document_id=document_id)
    assert empty.value.status_code == 409


def test_bm25_does_not_commit_database_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    document_id = uuid.uuid4()
    session = FakeSession({document_id})
    monkeypatch.setattr(
        bm25_retrieval,
        "_load_chunks",
        lambda _db, _scope: [make_chunk(document_id, 0, "agentic retrieval")],
    )

    bm25_retrieval.retrieve_bm25_chunks(session, "agentic")

    assert session.commits == 0


def test_search_result_can_represent_bm25_only_fields() -> None:
    result = SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        content="BM25 passage",
        metadata={"page_start": 1},
        bm25_score=1.25,
        bm25_rank=1,
    )

    assert result.similarity is None
    assert result.bm25_score == 1.25
