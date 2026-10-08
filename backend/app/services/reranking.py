from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_reranker_model_name
from app.schemas.search import SearchResult

RERANK_BATCH_SIZE = 16


class RerankingError(Exception):
    pass


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def rerank_candidates(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    if not candidates:
        return []

    try:
        pairs = [(query, candidate.content) for candidate in candidates]
        raw_scores = _load_model(get_reranker_model_name()).predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(candidates):
            raise RerankingError("Reranker returned an unexpected number of scores")
    except RerankingError:
        raise
    except Exception:
        raise RerankingError("Reranking is unavailable") from None

    scored_candidates = [
        (candidate.model_copy(update={"rerank_score": score}), position)
        for position, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
    ]
    scored_candidates.sort(key=lambda item: (-float(item[0].rerank_score), item[1]))
    return [candidate for candidate, _ in scored_candidates]