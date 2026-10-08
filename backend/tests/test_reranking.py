from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.main import app
from app.schemas.search import SearchResult
from app.services import reranking
from app.services.reranking import RerankingError


def make_result(index: int) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.UUID(int=index + 1),
        document_id=uuid.UUID(int=100),
        chunk_index=index,
        content=f"candidate passage {index}",
        metadata={"page_start": index + 1, "page_end": index + 2},
        similarity=0.7 - index / 100,
        bm25_score=2.0 - index / 10,
        hybrid_score=0.03 - index / 1000,
        dense_rank=index + 1,
        bm25_rank=index + 1,
    )


def test_reranker_scores_candidates_orders_them_and_preserves_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [make_result(0), make_result(1), make_result(2)]

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
            assert pairs == [("query", candidate.content) for candidate in candidates]
            assert kwargs["batch_size"] == reranking.RERANK_BATCH_SIZE
            assert kwargs["show_progress_bar"] is False
            return np.array([0.1, 0.9, -0.2])

    monkeypatch.setattr(reranking, "_load_model", lambda _name: FakeCrossEncoder())

    ranked = reranking.rerank_candidates("query", candidates)

    assert [candidate.chunk_index for candidate in ranked] == [1, 0, 2]
    assert [candidate.rerank_score for candidate in ranked] == pytest.approx([0.9, 0.1, -0.2])
    assert ranked[0].metadata == candidates[1].metadata
    assert ranked[0].content == candidates[1].content
    assert ranked[0].similarity == candidates[1].similarity
    assert ranked[0].bm25_score == candidates[1].bm25_score
    assert ranked[0].hybrid_score == candidates[1].hybrid_score
    assert all(candidate.rerank_score is not None for candidate in ranked)


def test_empty_candidates_do_not_load_the_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reranking,
        "_load_model",
        lambda _name: pytest.fail("Model should not load for no candidates"),
    )

    assert reranking.rerank_candidates("query", []) == []


def test_model_loader_caches_cross_encoder_and_selects_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sentence_transformers

    instances: list[object] = []

    def fake_cross_encoder(model_name: str, *, device: str) -> object:
        instance = object()
        instances.append((model_name, device, instance))
        return instance

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_cross_encoder)
    reranking._load_model.cache_clear()
    try:
        first = reranking._load_model("BAAI/bge-reranker-v2-m3")
        second = reranking._load_model("BAAI/bge-reranker-v2-m3")
    finally:
        reranking._load_model.cache_clear()

    assert first is second
    assert instances == [("BAAI/bge-reranker-v2-m3", "cpu", first)]


def test_reranker_rejects_a_score_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoder:
        def predict(self, _pairs: list[tuple[str, str]], **_kwargs: Any) -> list[float]:
            return [0.1]

    monkeypatch.setattr(reranking, "_load_model", lambda _name: FakeCrossEncoder())

    with pytest.raises(RerankingError, match="unexpected number of scores"):
        reranking.rerank_candidates("query", [make_result(0), make_result(1)])


class FakeReadOnlySession:
    def __init__(self) -> None:
        self.commits = 0


@pytest.fixture
def rerank_client() -> tuple[TestClient, FakeReadOnlySession]:
    session = FakeReadOnlySession()
    app.dependency_overrides[get_db_session] = lambda: session
    with TestClient(app) as client:
        yield client, session
    app.dependency_overrides.clear()


def test_search_reranks_expanded_hybrid_candidates_and_returns_final_top_k(
    rerank_client: tuple[TestClient, FakeReadOnlySession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = rerank_client
    retrieval_calls: list[tuple[str, int, uuid.UUID | None]] = []
    rerank_candidate_count: list[int] = []
    document_id = uuid.uuid4()

    def dense(
        _db: Any,
        query: str,
        top_k: int,
        document_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        assert query == "query"
        retrieval_calls.append(("dense", top_k, document_id))
        return [make_result(index) for index in range(20)]

    def bm25(
        _db: Any,
        query: str,
        top_k: int,
        document_id: uuid.UUID | None,
    ) -> list[SearchResult]:
        assert query == "query"
        retrieval_calls.append(("bm25", top_k, document_id))
        return [make_result(index + 20) for index in range(20)]

    def fake_rerank(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        assert query == "query"
        rerank_candidate_count.append(len(candidates))
        return [
            candidate.model_copy(update={"rerank_score": float(position)})
            for position, candidate in reversed(list(enumerate(candidates)))
        ]

    monkeypatch.setattr("app.services.search_pipeline.retrieve_chunks", dense)
    monkeypatch.setattr("app.services.search_pipeline.retrieve_bm25_chunks", bm25)
    monkeypatch.setattr("app.services.search_pipeline.rerank_candidates", fake_rerank)

    response = client.post(
        "/search",
        json={
            "query": "query",
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "rerank": True,
            "candidate_k": 20,
            "document_id": str(document_id),
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert retrieval_calls == [("dense", 20, document_id), ("bm25", 20, document_id)]
    assert rerank_candidate_count == [20]
    assert body["reranked"] is True
    assert len(body["results"]) == 5
    assert body["results"][0]["rerank_score"] == 19.0
    assert body["results"][0]["metadata"]["page_start"] is not None
    assert session.commits == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "query", "top_k": 5, "candidate_k": 4},
        {"query": "query", "top_k": 5, "candidate_k": 51},
        {"query": "query", "top_k": 21, "candidate_k": 21},
    ],
)
def test_search_validates_candidate_k_and_top_k(
    rerank_client: tuple[TestClient, FakeReadOnlySession],
    payload: dict[str, Any],
) -> None:
    client, _ = rerank_client

    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_search_rerank_false_keeps_dense_mode_and_does_not_call_reranker(
    rerank_client: tuple[TestClient, FakeReadOnlySession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = rerank_client
    results = [make_result(index) for index in range(7)]
    calls: list[int] = []

    monkeypatch.setattr("app.services.search_pipeline.retrieve_chunks", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        "app.services.search_pipeline.rerank_candidates",
        lambda *_args, **_kwargs: calls.append(1),
    )

    response = client.post("/search", json={"query": "query", "retrieval_mode": "dense"})

    assert response.status_code == 200
    assert response.json()["reranked"] is False
    assert len(response.json()["results"]) == 5
    assert "rerank_score" not in response.json()["results"][0]
    assert calls == []