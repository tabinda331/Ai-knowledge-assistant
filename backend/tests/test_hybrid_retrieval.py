from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.main import app
from app.schemas.search import SearchResult
from app.services import bm25_retrieval, vector_retrieval
from app.services.hybrid_retrieval import fuse_ranked_results
from app.services.vector_retrieval import VectorRetrievalError
from app.services.bm25_retrieval import BM25RetrievalError


def result(
    chunk_index: int,
    *,
    similarity: float | None = None,
    bm25_score: float | None = None,
    bm25_rank: int | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.UUID(int=chunk_index + 1),
        document_id=uuid.UUID(int=100),
        chunk_index=chunk_index,
        content=f"Evidence {chunk_index}",
        metadata={"page_start": chunk_index + 1, "page_end": chunk_index + 1},
        similarity=similarity,
        bm25_score=bm25_score,
        bm25_rank=bm25_rank,
    )


def test_rrf_calculation_order_and_one_sided_results() -> None:
    dense = [result(0, similarity=0.9), result(1, similarity=0.8)]
    bm25 = [result(0, bm25_score=4.2, bm25_rank=1), result(2, bm25_score=2.1, bm25_rank=2)]

    fused = fuse_ranked_results(dense, bm25, top_k=3, rrf_constant=60)

    assert fused[0].chunk_index == 0
    assert fused[0].hybrid_score == pytest.approx((1 / 61) + (1 / 61))
    assert fused[0].dense_rank == fused[0].bm25_rank == 1
    dense_only = next(item for item in fused if item.chunk_index == 1)
    bm25_only = next(item for item in fused if item.chunk_index == 2)
    assert dense_only.hybrid_score == pytest.approx(1 / 62)
    assert dense_only.dense_rank == 2 and dense_only.bm25_rank is None
    assert bm25_only.hybrid_score == pytest.approx(1 / 62)
    assert bm25_only.dense_rank is None and bm25_only.bm25_rank == 2
    assert bm25_only.metadata["page_start"] == 3


def test_rrf_top_k_limits_fused_results() -> None:
    dense = [result(index, similarity=1 - index / 10) for index in range(4)]
    bm25 = [result(index, bm25_score=4 - index, bm25_rank=index + 1) for index in range(4)]

    assert len(fuse_ranked_results(dense, bm25, top_k=2)) == 2


class FakeReadOnlySession:
    def __init__(self) -> None:
        self.commits = 0


@pytest.fixture
def hybrid_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeReadOnlySession]:
    session = FakeReadOnlySession()
    app.dependency_overrides[get_db_session] = lambda: session
    with TestClient(app) as client:
        yield client, session
    app.dependency_overrides.clear()


def test_hybrid_mode_runs_dense_and_bm25_with_same_document_filter(
    hybrid_client: tuple[TestClient, FakeReadOnlySession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = hybrid_client
    document_id = uuid.uuid4()
    calls: list[tuple[str, int, uuid.UUID | None]] = []

    def dense(
        _db: Any,
        query: str,
        top_k: int,
        document_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        assert query == "traditional RAG limitations"
        calls.append(("dense", top_k, document_id))
        return [result(0, similarity=0.9)]

    def bm25(
        _db: Any,
        query: str,
        top_k: int,
        document_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        assert query == "traditional RAG limitations"
        calls.append(("bm25", top_k, document_id))
        return [result(0, bm25_score=2.0, bm25_rank=1)]

    monkeypatch.setattr("app.services.search_pipeline.retrieve_chunks", dense)
    monkeypatch.setattr("app.services.search_pipeline.retrieve_bm25_chunks", bm25)
    response = client.post(
        "/search",
        json={
            "query": "traditional RAG limitations",
            "top_k": 1,
            "document_id": str(document_id),
            "retrieval_mode": "hybrid",
        },
    )

    assert response.status_code == 200
    assert calls == [("dense", 1, document_id), ("bm25", 1, document_id)]
    assert response.json()["retrieval_mode"] == "hybrid"
    assert len(response.json()["results"]) == 1
    assert response.json()["results"][0]["metadata"]["page_start"] == 1
    assert session.commits == 0


def test_global_hybrid_retrieval_passes_global_ready_scope_to_both_methods(
    hybrid_client: tuple[TestClient, FakeReadOnlySession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = hybrid_client
    calls: list[tuple[str, uuid.UUID | None]] = []
    eligible = result(0, similarity=0.9, bm25_score=2.0, bm25_rank=1)

    def dense(db: Any, query: str, top_k: int, document_id: uuid.UUID | None) -> list[SearchResult]:
        assert db is _session and query == "ready evidence" and top_k == 5
        calls.append(("dense", document_id))
        return [eligible.model_copy(update={"bm25_score": None, "bm25_rank": None})]

    def bm25(db: Any, query: str, top_k: int, document_id: uuid.UUID | None) -> list[SearchResult]:
        assert db is _session and query == "ready evidence" and top_k == 5
        calls.append(("bm25", document_id))
        return [eligible.model_copy(update={"similarity": None})]

    monkeypatch.setattr("app.services.search_pipeline.retrieve_chunks", dense)
    monkeypatch.setattr("app.services.search_pipeline.retrieve_bm25_chunks", bm25)

    response = client.post(
        "/search",
        json={"query": "ready evidence", "retrieval_mode": "hybrid", "rerank": False},
    )

    assert response.status_code == 200
    assert calls == [("dense", None), ("bm25", None)]
    assert len(response.json()["results"]) == 1
    assert response.json()["results"][0]["hybrid_score"] is not None


def test_bm25_mode_calls_only_keyword_retrieval(
    hybrid_client: tuple[TestClient, FakeReadOnlySession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = hybrid_client
    calls: list[str] = []

    def bm25(
        _db: Any,
        query: str,
        top_k: int,
        document_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        assert query == "exact term" and top_k == 5 and document_id is None
        calls.append("bm25")
        return [result(0, bm25_score=1.8, bm25_rank=1)]

    def unexpected_dense(*_args: Any, **_kwargs: Any) -> list[SearchResult]:
        pytest.fail("Dense retrieval should not run in BM25 mode")

    monkeypatch.setattr("app.services.search_pipeline.retrieve_bm25_chunks", bm25)
    monkeypatch.setattr("app.services.search_pipeline.retrieve_chunks", unexpected_dense)
    response = client.post("/search", json={"query": "exact term", "retrieval_mode": "bm25"})

    assert response.status_code == 200
    assert response.json()["results"][0]["bm25_score"] == pytest.approx(1.8)
    assert "similarity" not in response.json()["results"][0]
    assert calls == ["bm25"]


def test_hybrid_propagates_controlled_retrieval_errors(
    hybrid_client: tuple[TestClient, FakeReadOnlySession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = hybrid_client

    def fail_dense(*_args: Any, **_kwargs: Any) -> list[SearchResult]:
        raise VectorRetrievalError("Document not found", 404)

    monkeypatch.setattr("app.services.search_pipeline.retrieve_chunks", fail_dense)
    monkeypatch.setattr(
        "app.services.search_pipeline.retrieve_bm25_chunks",
        lambda *_args, **_kwargs: [],
    )

    response = client.post("/search", json={"query": "test", "retrieval_mode": "hybrid"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"
